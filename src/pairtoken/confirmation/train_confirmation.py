#!/usr/bin/env python3
"""Fit the frozen N02/C02 forecasts from outcome-blinded fold artifacts.

The production data-preparation adapter is intentionally a separate key.  Each
input bundle contains train/validation labels and test features, but it has no
test label, return, funding, cost, IC, or portfolio-outcome member.  This
module imports only the frozen lower-level model implementation; it never calls
the development runner, whose training loop computes development ICs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from confirmation_governance import (
    AUXILIARY_DIMENSION,
    CONTRACT_PATH,
    FORECAST_SCHEMA,
    MODEL_IDS,
    PAIR_COUNT,
    PATH_DIMENSION,
    RAW_DIMENSION,
    REPRESENTATIONS,
    SEEDS,
    ConfirmationGateError,
    ForecastSpec,
    atomic_json,
    axis_sha256,
    expected_artifact_path,
    expected_registry_path,
    fold_contract,
    hourly_clock_ns,
    load_contract,
    load_json,
    pair_ids,
    read_forecast_artifact,
    register_forecast,
    sha256_file,
    verify_source_freeze,
    write_forecast_artifact,
)


HERE = Path(__file__).resolve().parent
PARENT_MODEL_DIRECTORY = (HERE.parent / "development").resolve()
PARENT_MODELS_PATH = PARENT_MODEL_DIRECTORY / "paired_neural_models.py"
INPUT_SCHEMA = "pairtoken_1m_confirmation_blinded_fold_v1"

INPUT_MEMBERS = {
    "metadata_json",
    "pair_ids",
    "pair_leg_indices",
    "clock_source_ns",
    "train_rows",
    "validation_rows",
    "test_rows",
    "coin_minute_returns",
    "pair_weights",
    "pair_path_endpoints",
    "feature_available",
    "target_cashflows_train",
    "target_cashflows_validation",
    "target_available_train",
    "target_available_validation",
}
FORBIDDEN_MEMBER_FRAGMENTS = (
    "target_cashflows_test",
    "target_available_test",
    "test_return",
    "test_funding",
    "test_cost",
    "test_ic",
    "test_metric",
    "test_sharpe",
    "test_drawdown",
    "test_turnover",
)


def _array_family_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash factorized arrays exactly as the preparation key does."""

    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class BlindedFoldInput:
    fold: str
    representation: str
    pair_ids: tuple[str, ...]
    train_clock_ns: np.ndarray
    validation_clock_ns: np.ndarray
    test_clock_ns: np.ndarray
    raw_train: np.ndarray
    raw_validation: np.ndarray
    raw_test: np.ndarray
    feature_train: np.ndarray
    feature_validation: np.ndarray
    feature_test: np.ndarray
    target_train: np.ndarray
    target_validation: np.ndarray
    label_train: np.ndarray
    label_validation: np.ndarray
    artifact_sha256: str
    factorized_content_sha256: str = ""


def _expected_partition_clocks(fold: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hour = 3_600_000_000_000
    day = 24 * hour
    test = hourly_clock_ns(fold["test_start"], fold["test_end_exclusive"])
    test_start = int(test[0])
    validation_end = test_start - hour
    validation_start = validation_end - 90 * day
    training_end = validation_start - hour
    training_start = training_end - 365 * day
    return (
        np.arange(training_start, training_end, hour, dtype=np.int64),
        np.arange(validation_start, validation_end, hour, dtype=np.int64),
        test,
    )


def _input_path(root: Path, fold: str, representation: str) -> Path:
    return Path(root) / fold / f"{representation}.blinded.npz"


def materialize_factorized_rows(
    *,
    source_clock: np.ndarray,
    selected: np.ndarray,
    legs: np.ndarray,
    coin_minute: np.ndarray,
    weights: np.ndarray,
    endpoints: np.ndarray,
    source_feature: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one partition with the exact frozen parent arithmetic."""

    signal = legs[:, 0]
    hedge = legs[:, 1]
    episode = selected - 1
    paths = (
        weights[episode, :, 0, None] * coin_minute[episode[:, None], signal[None, :], :]
        + weights[episode, :, 1, None] * coin_minute[episode[:, None], hedge[None, :], :]
    )
    mask = source_feature[selected]
    paths = np.where(mask[:, :, None], paths, 0.0).astype(np.float32)
    if representation == "causal_15m_control":
        blocks = paths.reshape(*paths.shape[:2], 4, 15)
        original_endpoints = blocks.sum(axis=-1, dtype=np.float64)
        paths = np.broadcast_to(
            blocks.mean(axis=-1, keepdims=True, dtype=np.float64), blocks.shape
        ).reshape(paths.shape).astype(np.float32)
        controlled_endpoints = paths.reshape(*paths.shape[:2], 4, 15).sum(
            axis=-1, dtype=np.float64
        )
        if not np.allclose(
            original_endpoints, controlled_endpoints, atol=2e-9, rtol=2e-6
        ):
            raise ConfirmationGateError("control materialization changed a 15-minute endpoint")
    elif representation != "native_1m":
        raise ConfirmationGateError("unknown path representation")
    history = np.stack(
        [endpoints[row - 168 : row].T for row in selected], axis=0
    ).astype(np.float32)
    current_weights = weights[selected]
    hours = pd.to_datetime(source_clock[selected], utc=True).hour.to_numpy(dtype=float)
    phase = 2.0 * np.pi * hours / 24.0
    time = np.column_stack((np.sin(phase), np.cos(phase))).astype(np.float32)
    time = np.broadcast_to(time[:, None, :], (len(selected), PAIR_COUNT, 2))
    raw = np.concatenate((paths, history, current_weights, time), axis=2).astype(np.float32)
    if raw.shape != (len(selected), PAIR_COUNT, RAW_DIMENSION):
        raise ConfirmationGateError("materialized raw feature axes differ")
    raw = np.where(mask[:, :, None], raw, 0.0).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ConfirmationGateError("materialized raw features are nonfinite")
    return raw, mask


def load_blinded_fold_input(
    path: Path,
    *,
    fold_name: str,
    representation: str,
    source_freeze_path: Path,
) -> BlindedFoldInput:
    contract = load_contract()
    freeze = verify_source_freeze(source_freeze_path)
    try:
        with np.load(path, allow_pickle=False) as stored:
            names = set(stored.files)
            if names != INPUT_MEMBERS:
                leaked = sorted(
                    name for name in names
                    if any(fragment in name.lower() for fragment in FORBIDDEN_MEMBER_FRAGMENTS)
                )
                if leaked:
                    raise ConfirmationGateError(
                        f"test-outcome members are forbidden in the forecast stage: {leaked}"
                    )
                raise ConfirmationGateError(f"blinded input member family differs: {path}")
            metadata = json.loads(str(stored["metadata_json"].item()))
            shared_arrays = {
                name: np.asarray(stored[name]) for name in INPUT_MEMBERS - {"metadata_json"}
            }
            pairs = tuple(shared_arrays["pair_ids"].astype(str))
            source_clock = np.asarray(shared_arrays["clock_source_ns"], dtype=np.int64)
            train_rows = np.asarray(shared_arrays["train_rows"], dtype=np.int64)
            validation_rows = np.asarray(shared_arrays["validation_rows"], dtype=np.int64)
            test_rows = np.asarray(shared_arrays["test_rows"], dtype=np.int64)
            legs = np.asarray(shared_arrays["pair_leg_indices"], dtype=np.int64)
            coin_minute = np.asarray(shared_arrays["coin_minute_returns"], dtype=np.float32)
            # Match the frozen component path: hedge weights and training labels
            # retain float64 precision until raw tokens/model transforms cast.
            weights = np.asarray(shared_arrays["pair_weights"], dtype=np.float64)
            endpoints = np.asarray(shared_arrays["pair_path_endpoints"], dtype=np.float32)
            source_feature = np.asarray(shared_arrays["feature_available"], dtype=bool)
            target_train = np.asarray(shared_arrays["target_cashflows_train"], dtype=np.float64)
            target_validation = np.asarray(
                shared_arrays["target_cashflows_validation"], dtype=np.float64
            )
            label_train = np.asarray(shared_arrays["target_available_train"], dtype=bool)
            label_validation = np.asarray(
                shared_arrays["target_available_validation"], dtype=bool
            )
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ConfirmationGateError(f"invalid blinded fold input: {path}") from exc
    required_metadata = {
        "schema": INPUT_SCHEMA,
        "status": "complete_factorized_train_validation_labels_and_test_features_only",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "fold": fold_name,
        "representation": representation,
        "pair_count": PAIR_COUNT,
        "raw_dimension": RAW_DIMENSION,
        "auxiliary_dimension": AUXILIARY_DIMENSION,
        "auxiliary_values": "all_zeros_created_by_trainer",
        "test_target_labels_included": False,
        "causal_test_history_features_included": True,
        "confirmation_metrics_computed": False,
        "factorized_storage": True,
    }
    if not isinstance(metadata, dict):
        raise ConfirmationGateError("blinded-input metadata must be a mapping")
    for key, value in required_metadata.items():
        if metadata.get(key) != value:
            raise ConfirmationGateError(f"blinded-input metadata differs for {key}")
    factorized_hash = str(metadata.get("factorized_content_sha256", ""))
    if len(factorized_hash) != 64 or any(character not in "0123456789abcdef" for character in factorized_hash):
        raise ConfirmationGateError("factorized content hash is invalid")
    if _array_family_sha256(shared_arrays) != factorized_hash:
        raise ConfirmationGateError("factorized content does not match its cryptographic hash")
    forbidden_metadata = (
        "test_target", "test_return", "test_funding", "test_cost", "test_ic",
        "test_sharpe", "test_drawdown", "test_turnover", "test_metric",
    )
    governed_disclosures = {
        "test_target_labels_included", "causal_test_history_features_included"
    }
    if any(
        key not in governed_disclosures
        and any(fragment in str(key).lower() for fragment in forbidden_metadata)
        for key in metadata
    ):
        raise ConfirmationGateError("blinded-input metadata contains a test-outcome key")
    expected_pairs = pair_ids(contract["data"]["symbols_in_order"])
    if pairs != expected_pairs or axis_sha256(pairs) != metadata.get("pair_axis_sha256"):
        raise ConfirmationGateError("blinded-input pair axis differs")
    fold = fold_contract(contract, fold_name)
    expected_train, expected_validation, expected_test = _expected_partition_clocks(fold)
    if (
        source_clock.ndim != 1
        or len(source_clock) < 170
        or not np.all(np.diff(source_clock) == 3_600_000_000_000)
    ):
        raise ConfirmationGateError("factorized source clock differs")
    for label, rows in (
        ("train", train_rows), ("validation", validation_rows), ("test", test_rows)
    ):
        if rows.ndim != 1 or len(rows) == 0 or rows.min() < 168 or rows.max() >= len(source_clock):
            raise ConfirmationGateError(f"factorized {label} row axis differs")
        if not np.all(np.diff(rows) == 1):
            raise ConfirmationGateError(f"factorized {label} rows are not contiguous")
    train_clock = source_clock[train_rows]
    validation_clock = source_clock[validation_rows]
    test_clock = source_clock[test_rows]
    for label, observed, expected in (
        ("train", train_clock, expected_train),
        ("validation", validation_clock, expected_validation),
        ("test", test_clock, expected_test),
    ):
        if not np.array_equal(observed, expected):
            raise ConfirmationGateError(f"{label} clock differs for {fold_name}")
    rows_count = len(source_clock)
    if legs.shape != (PAIR_COUNT, 2) or (legs < 0).any() or (legs >= 24).any():
        raise ConfirmationGateError("factorized pair-leg axis differs")
    if coin_minute.shape != (rows_count, 24, 60):
        raise ConfirmationGateError("factorized minute-return cube differs")
    if weights.shape != (rows_count, PAIR_COUNT, 2):
        raise ConfirmationGateError("factorized pair-weight tensor differs")
    if endpoints.shape != (rows_count, PAIR_COUNT) or source_feature.shape != endpoints.shape:
        raise ConfirmationGateError("factorized endpoint/mask axes differ")
    if not np.isfinite(coin_minute).all() or not np.isfinite(endpoints).all():
        raise ConfirmationGateError("factorized path arrays are nonfinite")

    materializer = lambda selected: materialize_factorized_rows(
        source_clock=source_clock,
        selected=selected,
        legs=legs,
        coin_minute=coin_minute,
        weights=weights,
        endpoints=endpoints,
        source_feature=source_feature,
        representation=representation,
    )
    raw_train, feature_train = materializer(train_rows)
    raw_validation, feature_validation = materializer(validation_rows)
    raw_test, feature_test = materializer(test_rows)
    for label, target, available, rows, feature in (
        ("train", target_train, label_train, len(train_clock), feature_train),
        ("validation", target_validation, label_validation, len(validation_clock), feature_validation),
    ):
        if target.shape != (rows, PAIR_COUNT) or available.shape != target.shape:
            raise ConfirmationGateError(f"{label} target axes differ")
        if np.any(available & ~feature) or not np.isfinite(target[available]).all():
            raise ConfirmationGateError(f"{label} target availability or values differ")
        if np.any(target[~available] != 0.0):
            raise ConfirmationGateError(f"{label} unavailable targets must be zero")
    return BlindedFoldInput(
        fold_name, representation, pairs,
        train_clock, validation_clock, test_clock,
        raw_train, raw_validation, raw_test,
        feature_train, feature_validation, feature_test,
        target_train, target_validation, label_train, label_validation,
        sha256_file(path), factorized_hash,
    )


def paired_input_audit(native: BlindedFoldInput, control: BlindedFoldInput) -> None:
    if native.fold != control.fold or native.pair_ids != control.pair_ids:
        raise ConfirmationGateError("paired input identities differ")
    for partition in ("train", "validation", "test"):
        native_raw = getattr(native, f"raw_{partition}")
        control_raw = getattr(control, f"raw_{partition}")
        native_mask = getattr(native, f"feature_{partition}")
        control_mask = getattr(control, f"feature_{partition}")
        if not np.array_equal(native_mask, control_mask):
            raise ConfirmationGateError(f"paired {partition} feature masks differ")
        if not np.array_equal(native_raw[..., PATH_DIMENSION:], control_raw[..., PATH_DIMENSION:]):
            raise ConfirmationGateError(f"paired {partition} common tails differ")
        native_blocks = native_raw[..., :PATH_DIMENSION].reshape(
            *native_raw.shape[:2], 4, 15
        ).sum(axis=-1)
        control_blocks = control_raw[..., :PATH_DIMENSION].reshape(
            *control_raw.shape[:2], 4, 15
        ).sum(axis=-1)
        if not np.allclose(native_blocks, control_blocks, atol=2e-9, rtol=2e-6):
            raise ConfirmationGateError(f"paired {partition} fifteen-minute endpoints differ")
    for name in ("target_train", "target_validation", "label_train", "label_validation"):
        if not np.array_equal(getattr(native, name), getattr(control, name)):
            raise ConfirmationGateError(f"paired training/validation outcome differs: {name}")


def load_frozen_models(source_freeze_path: Path) -> Any:
    verify_source_freeze(source_freeze_path)
    if str(PARENT_MODEL_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(PARENT_MODEL_DIRECTORY))
    name = "pairtoken_confirmation_frozen_paired_models"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PARENT_MODELS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(PARENT_MODELS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if tuple(module.FIXED_SEEDS) != SEEDS:
        raise ConfirmationGateError("frozen model seed family differs")
    return module


def runtime_identity(device: str) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise ConfirmationGateError("CUDA was requested but is unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _runtime_sha256(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bind_runtime_manifest(
    output_root: Path, runtime: Mapping[str, Any], source_freeze_path: Path
) -> None:
    payload = {
        "schema": "pairtoken_1m_confirmation_runtime_v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "runtime_sha256": _runtime_sha256(runtime),
        "runtime": dict(runtime),
    }
    path = Path(output_root) / "runtime_manifest.json"
    if path.exists():
        if load_json(path, label="runtime manifest") != payload:
            raise ConfirmationGateError("forecast runtime changed inside the sealed run")
    else:
        atomic_json(path, payload)


def _panel_and_fold(bundle: BlindedFoldInput, models: Any) -> tuple[Any, Any]:
    raw = np.concatenate((bundle.raw_train, bundle.raw_validation, bundle.raw_test), axis=0)
    feature = np.concatenate(
        (bundle.feature_train, bundle.feature_validation, bundle.feature_test), axis=0
    )
    target = np.concatenate((
        bundle.target_train,
        bundle.target_validation,
        np.zeros(bundle.feature_test.shape, dtype=np.float32),
    ))
    label = np.concatenate((
        bundle.label_train,
        bundle.label_validation,
        np.zeros(bundle.feature_test.shape, dtype=bool),
    ))
    clock = pd.to_datetime(np.concatenate((
        bundle.train_clock_ns, bundle.validation_clock_ns, bundle.test_clock_ns
    )), utc=True)
    auxiliary = np.zeros((*raw.shape[:2], AUXILIARY_DIMENSION), dtype=np.float32)
    panel = models.MaterializedPanel(
        clock=clock,
        pair_ids=bundle.pair_ids,
        raw_features=raw,
        zero_auxiliary=auxiliary,
        feature_available=feature,
        target_cashflows=target,
        target_available=label,
    )
    n_train = len(bundle.train_clock_ns)
    n_validation = len(bundle.validation_clock_ns)
    fold = models.cycle.PilotFold(
        bundle.fold,
        np.arange(n_train, dtype=np.int64),
        np.arange(n_train, n_train + n_validation, dtype=np.int64),
        np.arange(n_train + n_validation, len(clock), dtype=np.int64),
    )
    if panel.label_mask[fold.test_indices].any() or np.any(
        panel.target_cashflows[fold.test_indices] != 0.0
    ):
        raise ConfirmationGateError("test outcomes reached the forecast-stage panel")
    return panel, fold


def _diagnostics(
    *,
    spec: ForecastSpec,
    input_hash: str,
    source_freeze_path: Path,
    runtime: Mapping[str, Any],
    epochs: int | None = None,
    training_loss: float | None = None,
    validation_loss: float | None = None,
) -> dict[str, Any]:
    freeze = verify_source_freeze(source_freeze_path)
    result: dict[str, Any] = {
        "schema": FORECAST_SCHEMA,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "input_artifact_sha256": input_hash,
        "runtime_sha256": _runtime_sha256(runtime),
        "fold": spec.fold,
        "representation": spec.representation,
        "model_id": spec.model_id,
        "seed": spec.seed,
        "role": spec.role,
        "target_horizon_hours": 1,
        "confirmation_metrics_computed": False,
        "test_outcomes_loaded": False,
    }
    if epochs is not None:
        result.update({
            "epochs": int(epochs),
            "training_loss": float(training_loss),
            "validation_loss": float(validation_loss),
        })
    if spec.role == "seed_mean":
        result.update({
            "seeds": list(SEEDS),
            "aggregation": "arithmetic_mean_never_best_seed",
        })
    return result


def train_one_bundle(
    bundle: BlindedFoldInput,
    *,
    models: Any,
    output_root: Path,
    source_freeze_path: Path,
    device: str,
) -> None:
    panel, fold = _panel_and_fold(bundle, models)
    runtime = runtime_identity(device)
    _bind_runtime_manifest(output_root, runtime, source_freeze_path)
    predictions: dict[int, np.ndarray] = {}
    test = fold.test_indices
    for seed in SEEDS:
        spec = ForecastSpec(bundle.fold, bundle.representation, str(seed), "individual_seed")
        path = expected_artifact_path(output_root, spec)
        diagnostics = _diagnostics(
            spec=spec,
            input_hash=bundle.artifact_sha256,
            source_freeze_path=source_freeze_path,
            runtime=runtime,
        )
        if path.exists():
            loaded = read_forecast_artifact(path, spec, load_contract())
            for key, value in diagnostics.items():
                if key not in {"epochs", "training_loss", "validation_loss"} and loaded["diagnostics"].get(key) != value:
                    raise ConfirmationGateError(f"existing forecast identity differs: {path}")
            prediction = loaded["predictions"]
        else:
            fitted = models.fit_itransformer(
                panel,
                fold,
                config=models.transformer_config(seed=seed, device=device),
            )
            prediction = fitted.predict(
                panel.raw_features[test],
                panel.zero_auxiliary[test],
                panel.feature_available[test],
            )
            full_diagnostics = _diagnostics(
                spec=spec,
                input_hash=bundle.artifact_sha256,
                source_freeze_path=source_freeze_path,
                runtime=runtime,
                epochs=int(fitted.epochs_trained),
                training_loss=float(fitted.training_losses[-1]),
                validation_loss=float(fitted.validation_losses[-1]),
            )
            write_forecast_artifact(
                path,
                clock_ns=bundle.test_clock_ns,
                pair_axis=bundle.pair_ids,
                feature_available=bundle.feature_test,
                predictions=prediction,
                diagnostics=full_diagnostics,
                pair_targets=None,
            )
            del fitted
        predictions[seed] = np.asarray(prediction, dtype=np.float32)
        registry = expected_registry_path(output_root, spec)
        if not registry.exists():
            register_forecast(
                output_root,
                spec=spec,
                artifact_path=path,
                input_sha256=bundle.artifact_sha256,
                source_freeze_path=source_freeze_path,
            )
    ensemble = models.mean_seed_predictions(predictions)
    pair_targets = models.cycle.rank_pair_targets(
        ensemble,
        bundle.feature_test,
        long_pairs=8,
        short_pairs=8,
        absolute_pair_target=0.05,
    ).astype(np.float32)
    ensemble_spec = ForecastSpec(
        bundle.fold, bundle.representation, "all3", "seed_mean"
    )
    ensemble_path = expected_artifact_path(output_root, ensemble_spec)
    diagnostics = _diagnostics(
        spec=ensemble_spec,
        input_hash=bundle.artifact_sha256,
        source_freeze_path=source_freeze_path,
        runtime=runtime,
    )
    if ensemble_path.exists():
        loaded = read_forecast_artifact(ensemble_path, ensemble_spec, load_contract())
        if not np.array_equal(loaded["predictions"], ensemble, equal_nan=True):
            raise ConfirmationGateError("existing ensemble does not rebuild from all seeds")
        if not np.array_equal(loaded["pair_targets"], pair_targets):
            raise ConfirmationGateError("existing pair targets do not rebuild")
    else:
        write_forecast_artifact(
            ensemble_path,
            clock_ns=bundle.test_clock_ns,
            pair_axis=bundle.pair_ids,
            feature_available=bundle.feature_test,
            predictions=ensemble,
            diagnostics=diagnostics,
            pair_targets=pair_targets,
        )
    registry = expected_registry_path(output_root, ensemble_spec)
    if not registry.exists():
        register_forecast(
            output_root,
            spec=ensemble_spec,
            artifact_path=ensemble_path,
            input_sha256=bundle.artifact_sha256,
            source_freeze_path=source_freeze_path,
        )


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fold", action="append", dest="folds")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    contract = load_contract()
    verify_source_freeze(args.source_freeze)
    allowed_folds = tuple(row["name"] for row in contract["refits"])
    selected = tuple(args.folds) if args.folds else allowed_folds
    if len(set(selected)) != len(selected) or any(fold not in allowed_folds for fold in selected):
        raise ConfirmationGateError("requested fold family is invalid")
    models = load_frozen_models(args.source_freeze)
    completed: list[str] = []
    for fold in selected:
        shared_factorized_hash: str | None = None
        for representation in REPRESENTATIONS:
            bundle = load_blinded_fold_input(
                _input_path(args.input_root, fold, representation),
                fold_name=fold,
                representation=representation,
                source_freeze_path=args.source_freeze,
            )
            if shared_factorized_hash is None:
                shared_factorized_hash = bundle.factorized_content_sha256
            elif bundle.factorized_content_sha256 != shared_factorized_hash:
                raise ConfirmationGateError("N02/C02 factorized fold contents differ")
            train_one_bundle(
                bundle,
                models=models,
                output_root=args.output_root,
                source_freeze_path=args.source_freeze,
                device=args.device,
            )
            completed.append(f"{fold}/{MODEL_IDS[representation]}")
            del bundle
    print(json.dumps({
        "status": "requested_forecasts_registered_no_confirmation_metric_computed",
        "completed": completed,
        "test_outcomes_loaded": False,
        "confirmation_metrics_computed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
