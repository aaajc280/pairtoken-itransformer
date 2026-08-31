#!/usr/bin/env python3
"""Train and evaluate the frozen 13-quarter historical robustness family.

This adapter intentionally does not call the parent's six-fold runner or its
six-fold evaluator.  It imports their hash-bound feature/model and audited
portfolio primitives, derives all family counts from this package's contract,
stitches quarterly forecasts, and executes one continuous ledger per
series/cost.  Nothing at or after the July-2025 firewall is addressable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from history_core import (
    CONTRACT_PATH,
    EXPECTED_FOLDS,
    HARD_CUTOFF_EXCLUSIVE,
    canonical_json,
    contribution_concentration,
    expected_decision_clock,
    expected_fold_clocks,
    fold_for_dates,
    leave_one_fold_out_metrics,
    load_contract,
    moving_block_bootstrap,
    performance_metrics,
    sha256_file,
    stability_summary,
    utc,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
FROZEN_DIR = HERE.parent / "development"
if str(FROZEN_DIR) not in sys.path:
    sys.path.insert(0, str(FROZEN_DIR))

import paired_neural_core as frozen_core  # noqa: E402
import paired_neural_models as frozen_models  # noqa: E402
import portfolio_evaluation as frozen_evaluation  # noqa: E402


FORECAST_SCHEMA = "crypto_pairtoken_1m_history_forecast_v1"
RUN_SCHEMA = "crypto_pairtoken_1m_history_forecast_run_v1"
EVALUATION_SCHEMA = "crypto_pairtoken_1m_history_evaluation_v1"
REPRESENTATIONS = ("native_1m", "causal_15m_control")
SERIES = {
    "B00": ("endpoint_ridge_shared", "shared_endpoint", "endpoint_ridge", "none"),
    "N02": ("native_1m_itransformer_seed_mean", "native_1m", "itransformer_seed_mean", "all3"),
    "C02": ("causal_15m_itransformer_seed_mean", "causal_15m_control", "itransformer_seed_mean", "all3"),
}
IDENTITY_KEYS = frozenset(
    {
        "schema",
        "history_contract_sha256",
        "component_sha256",
        "pair_axis_sha256",
        "frozen_source_bundle_sha256",
        "history_source_bundle_sha256",
        "runtime_environment_sha256",
        "fold",
        "representation",
        "model",
        "seed",
        "role",
        "confirmation_parsed",
    }
)


def frozen_source_paths() -> dict[str, Path]:
    return {
        "contract.json": FROZEN_DIR / "contract.json",
        "paired_neural_core.py": FROZEN_DIR / "paired_neural_core.py",
        "paired_neural_models.py": FROZEN_DIR / "paired_neural_models.py",
        "run_paired_neural.py": FROZEN_DIR / "run_paired_neural.py",
        "portfolio_evaluation.py": FROZEN_DIR / "portfolio_evaluation.py",
        "parent_cycle_pilot.py": HERE.parent / "model" / "pairtoken_model.py",
    }


def validate_frozen_sources(contract: Mapping[str, Any]) -> dict[str, str]:
    expected = contract["frozen_parent"]["source_sha256"]
    paths = frozen_source_paths()
    if set(expected) != set(paths):
        raise ValueError("frozen source family differs")
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        mismatched = sorted(name for name in observed if observed[name] != expected[name])
        raise ValueError(f"frozen paired implementation changed: {mismatched}")
    return observed


def history_source_hashes() -> dict[str, str]:
    return {
        "contract.json": sha256_file(CONTRACT_PATH),
        "history_core.py": sha256_file(HERE / "history_core.py"),
        "run_history.py": sha256_file(Path(__file__)),
    }


def bundle_sha256(values: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json(dict(values))).hexdigest()


def pair_axis_sha256(pair_ids: Sequence[str]) -> str:
    payload = (json.dumps(list(map(str, pair_ids)), separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def runtime_environment(device: str) -> dict[str, Any]:
    requested = str(device)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "requested_device": requested,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if requested == "cuda" else None,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _atomic_create(path: Path, payload: bytes) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_create(path, canonical_json(dict(value)))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_create(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _forecast_path(output: Path, *, fold: str, representation: str, model: str, seed: str | int) -> Path:
    return output / "forecasts" / fold / f"{representation}__{model}__seed-{seed}.npz"


def _identity(
    *,
    contract_hash: str,
    component_hash: str,
    pair_hash: str,
    frozen_bundle_hash: str,
    history_bundle_hash: str,
    environment_hash: str,
    fold: str,
    representation: str,
    model: str,
    seed: str | int,
    role: str,
) -> dict[str, Any]:
    return {
        "schema": FORECAST_SCHEMA,
        "history_contract_sha256": contract_hash,
        "component_sha256": component_hash,
        "pair_axis_sha256": pair_hash,
        "frozen_source_bundle_sha256": frozen_bundle_hash,
        "history_source_bundle_sha256": history_bundle_hash,
        "runtime_environment_sha256": environment_hash,
        "fold": str(fold),
        "representation": str(representation),
        "model": str(model),
        "seed": str(seed),
        "role": str(role),
        "confirmation_parsed": False,
    }


def _save_forecast(
    path: Path,
    *,
    clock: pd.DatetimeIndex,
    predictions: np.ndarray,
    pair_targets: np.ndarray | None,
    diagnostics: Mapping[str, Any],
) -> None:
    targets = np.empty((0, 0), dtype=np.float32) if pair_targets is None else np.asarray(pair_targets, dtype=np.float32)
    _atomic_npz(
        path,
        clock_ns=clock.view("int64"),
        predictions=np.asarray(predictions, dtype=np.float32),
        pair_targets=targets,
        diagnostics_json=np.asarray(json.dumps(dict(diagnostics), sort_keys=True, separators=(",", ":"))),
    )


def _load_forecast(
    path: Path,
    *,
    expected_clock: pd.DatetimeIndex,
    expected_mask: np.ndarray,
    expected_identity: Mapping[str, Any],
    require_pair_targets: bool,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {"clock_ns", "predictions", "pair_targets", "diagnostics_json"}:
                raise ValueError("forecast NPZ members differ")
            clock = pd.to_datetime(stored["clock_ns"], utc=True)
            predictions = np.asarray(stored["predictions"], dtype=np.float32)
            pair_targets = np.asarray(stored["pair_targets"], dtype=np.float32)
            diagnostics = json.loads(str(stored["diagnostics_json"].item()))
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid history forecast artifact: {path}") from error
    shape = tuple(np.asarray(expected_mask).shape)
    if not clock.equals(expected_clock) or predictions.shape != shape:
        raise ValueError(f"forecast axes differ: {path}")
    mask = np.asarray(expected_mask, dtype=bool)
    if np.isinf(predictions).any() or not np.array_equal(np.isfinite(predictions), mask):
        raise ValueError(f"forecast finite mask differs: {path}")
    if not isinstance(diagnostics, dict) or not IDENTITY_KEYS.issubset(diagnostics):
        raise ValueError(f"forecast diagnostics are incomplete: {path}")
    if {key: diagnostics[key] for key in IDENTITY_KEYS} != dict(expected_identity):
        raise ValueError(f"forecast identity differs: {path}")
    if require_pair_targets:
        if pair_targets.shape != shape or not np.isfinite(pair_targets).all():
            raise ValueError(f"forecast pair-target axes differ: {path}")
        rebuilt = frozen_models.cycle.rank_pair_targets(
            predictions,
            mask,
            long_pairs=int(contract["portfolio"]["long_pairs"]),
            short_pairs=int(contract["portfolio"]["short_pairs"]),
            absolute_pair_target=float(contract["portfolio"]["absolute_pair_target"]),
        ).astype(np.float32)
        if not np.array_equal(pair_targets, rebuilt):
            raise ValueError(f"forecast pair targets do not rebuild: {path}")
        return predictions, pair_targets, diagnostics
    if pair_targets.shape != (0, 0):
        raise ValueError(f"individual seed artifact unexpectedly contains targets: {path}")
    return predictions, None, diagnostics


def _load_components(path: Path, contract: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    expected_hash = str(contract["frozen_parent"]["component_artifact_sha256"])
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError("component artifact hash differs from the frozen development input")
    components, metadata = frozen_core.load_components(path)
    if components.clock.max() >= HARD_CUTOFF_EXCLUSIVE:
        raise PermissionError("component cache reaches confirmation")
    provenance = metadata.get("provenance", {})
    if provenance.get("paired_contract_sha256") != contract["frozen_parent"]["source_sha256"]["contract.json"]:
        raise ValueError("component provenance is not bound to the frozen paired contract")
    if provenance.get("economic_artifact_sha256") != contract["frozen_parent"]["economic_artifact_sha256"]:
        raise ValueError("component provenance economic artifact differs")
    return components, metadata


def _validate_fold_partitions(components: Any, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    fold_clocks = expected_fold_clocks(contract)
    rows: list[dict[str, Any]] = []
    for spec in contract["folds"]:
        fold = frozen_models.make_walk_forward_fold(
            components,
            name=spec["name"],
            test_start=spec["test_start"],
            test_end_exclusive=spec["test_end_exclusive"],
            training_days=int(contract["walk_forward"]["training_days"]),
            validation_days=int(contract["walk_forward"]["validation_days"]),
            purge_hours=int(contract["walk_forward"]["target_purge_hours"]),
        )
        observed_test = components.clock.take(fold.test_indices)
        source_indices = np.concatenate(
            (fold.train_indices, fold.validation_indices, fold.test_indices)
        ).astype(np.int64)
        endpoint_floor = int(frozen_core.ENDPOINT_HISTORY_HOURS)
        if len(source_indices) == 0 or int(source_indices.min()) < endpoint_floor:
            raise ValueError(
                f"{spec['name']} source begins before the frozen "
                f"{endpoint_floor}-hour endpoint-history floor"
            )
        if len(fold.train_indices) != 365 * 24 or len(fold.validation_indices) != 90 * 24:
            raise ValueError(f"{spec['name']} lacks the exact formation windows")
        if not observed_test.equals(fold_clocks[str(spec["name"])]):
            raise ValueError(f"{spec['name']} test clock differs")
        rows.append(
            {
                "fold": str(spec["name"]),
                "minimum_source_index": int(source_indices.min()),
                "required_endpoint_history_hours": endpoint_floor,
                "training_hours": int(len(fold.train_indices)),
                "validation_hours": int(len(fold.validation_indices)),
                "test_hours": int(len(fold.test_indices)),
                "training_start": components.clock[fold.train_indices[0]].isoformat(),
                "training_end": components.clock[fold.train_indices[-1]].isoformat(),
                "validation_start": components.clock[fold.validation_indices[0]].isoformat(),
                "validation_end": components.clock[fold.validation_indices[-1]].isoformat(),
                "test_start": observed_test[0].isoformat(),
                "test_end": observed_test[-1].isoformat(),
            }
        )
    return rows


def preflight_inputs(
    *, components_path: Path, economic_inputs: Path | None = None, economic_manifest: Path | None = None
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    contract = load_contract()
    frozen_hashes = validate_frozen_sources(contract)
    components, metadata = _load_components(components_path, contract)
    partitions = _validate_fold_partitions(components, contract)
    clock = expected_decision_clock(contract)
    if (components.clock.get_indexer(clock) < 0).any():
        raise ValueError("component cache does not contain the continuous history clock")
    report: dict[str, Any] = {
        "schema": "crypto_pairtoken_1m_history_preflight_v1",
        "status": "valid_development_only_inputs",
        "sample_role": contract["claim_boundary"]["sample_role"],
        "not_confirmatory": True,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "component_sha256": sha256_file(components_path),
        "pair_axis_sha256": pair_axis_sha256(components.pair_ids),
        "frozen_source_sha256": frozen_hashes,
        "fold_partitions": partitions,
        "decision_rows": int(len(clock)),
        "first_decision": clock[0].isoformat(),
        "last_decision": clock[-1].isoformat(),
        "terminal_zero_target": contract["data"]["boundary_policy"]["terminal_zero_target"],
        "confirmation_parsed": False,
    }
    if economic_inputs is not None:
        if economic_manifest is None:
            raise ValueError("economic manifest is required with economic inputs")
        if sha256_file(economic_inputs) != contract["frozen_parent"]["economic_artifact_sha256"]:
            raise ValueError("economic artifact hash differs")
        if sha256_file(economic_manifest) != contract["frozen_parent"]["economic_manifest_sha256"]:
            raise ValueError("economic manifest hash differs")
        economic = frozen_evaluation.load_economic_inputs(economic_inputs, economic_manifest)
        if (
            components.symbols != economic.symbols
            or components.pair_ids != economic.pair_ids
            or not components.clock.equals(economic.clock)
            or not np.array_equal(components.pair_leg_indices, economic.pair_leg_indices)
        ):
            raise ValueError("component and economic axes differ")
        terminal = utc(contract["data"]["boundary_policy"]["terminal_zero_target"])
        if economic.clock.get_indexer([terminal])[0] < 0 or terminal >= HARD_CUTOFF_EXCLUSIVE:
            raise ValueError("economic artifact lacks the development terminal row")
        report["economic_artifact_sha256"] = sha256_file(economic_inputs)
        report["economic_manifest_sha256"] = sha256_file(economic_manifest)
    return components, metadata, report


def command_validate_contract(_args: argparse.Namespace) -> int:
    contract = load_contract()
    validate_frozen_sources(contract)
    print(json.dumps({"status": "contract_valid", "folds": list(EXPECTED_FOLDS), "confirmation_parsed": False}, sort_keys=True))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    _components, _metadata, report = preflight_inputs(
        components_path=args.components,
        economic_inputs=args.economic_inputs,
        economic_manifest=args.economic_manifest,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def command_train(args: argparse.Namespace) -> int:
    contract = load_contract()
    components, _metadata, preflight = preflight_inputs(components_path=args.components)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_names = ("forecast_manifest.csv", "fold_metrics.csv", "seed_metrics.csv", "ic_by_decision.csv", "attempt_ledger.csv")
    if (output / "run_manifest.json").exists():
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != RUN_SCHEMA or manifest.get("confirmation_parsed") is not False:
            raise ValueError("completed history run manifest differs")
        for name, digest in manifest.get("summary_sha256", {}).items():
            if name not in summary_names or sha256_file(output / name) != digest:
                raise ValueError(f"completed history summary differs: {name}")
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if any((output / name).exists() for name in summary_names):
        raise FileExistsError("partial history summaries exist without a completion manifest")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    frozen_models.cycle.set_reproducible_seed(int(contract["model"]["fixed_seeds"][0]))
    environment = runtime_environment(str(args.device))
    environment_hash = hashlib.sha256(canonical_json(environment)).hexdigest()
    contract_hash = sha256_file(CONTRACT_PATH)
    component_hash = sha256_file(args.components)
    pair_hash = pair_axis_sha256(components.pair_ids)
    frozen_hashes = validate_frozen_sources(contract)
    history_hashes = history_source_hashes()
    frozen_bundle_hash = bundle_sha256(frozen_hashes)
    history_bundle_hash = bundle_sha256(history_hashes)
    fold_clocks = expected_fold_clocks(contract)
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []

    for spec in contract["folds"]:
        fold = frozen_models.make_walk_forward_fold(
            components,
            name=spec["name"],
            test_start=spec["test_start"],
            test_end_exclusive=spec["test_end_exclusive"],
            training_days=365,
            validation_days=90,
            purge_hours=1,
        )
        expected_clock = fold_clocks[str(spec["name"])]
        panel, local_fold, _local = frozen_models.materialize_fold(components, fold, representation="native_1m")
        test = local_fold.test_indices
        endpoint_identity = _identity(
            contract_hash=contract_hash,
            component_hash=component_hash,
            pair_hash=pair_hash,
            frozen_bundle_hash=frozen_bundle_hash,
            history_bundle_hash=history_bundle_hash,
            environment_hash=environment_hash,
            fold=fold.name,
            representation="shared_endpoint",
            model="endpoint_ridge",
            seed="none",
            role="descriptive_endpoint_only_baseline",
        )
        endpoint_path = _forecast_path(output, fold=fold.name, representation="shared_endpoint", model="endpoint_ridge", seed="none")
        if not endpoint_path.exists():
            fitted_endpoint = frozen_models.fit_endpoint_ridge(panel, local_fold, ridge_penalty=10.0)
            endpoint_prediction = fitted_endpoint.predict(panel.raw_features[test], panel.feature_available[test])
            endpoint_targets = frozen_models.cycle.rank_pair_targets(
                endpoint_prediction,
                panel.feature_available[test],
                long_pairs=8,
                short_pairs=8,
                absolute_pair_target=0.05,
            )
            _save_forecast(endpoint_path, clock=expected_clock, predictions=endpoint_prediction, pair_targets=endpoint_targets, diagnostics=endpoint_identity)
        endpoint_prediction, _endpoint_targets, _ = _load_forecast(
            endpoint_path,
            expected_clock=expected_clock,
            expected_mask=panel.feature_available[test],
            expected_identity=endpoint_identity,
            require_pair_targets=True,
            contract=contract,
        )
        endpoint_ic = frozen_models.decision_ic(endpoint_prediction, panel.target_cashflows[test], panel.label_mask[test])
        metric_rows.append({"fold": fold.name, "series_id": "B00", "representation": "shared_endpoint", "model": "endpoint_ridge", "mean_ic": float(np.nanmean(endpoint_ic)), "finite_ic_decisions": int(np.isfinite(endpoint_ic).sum())})
        ic_rows.extend({"timestamp": timestamp.isoformat(), "fold": fold.name, "series_id": "B00", "representation": "shared_endpoint", "model": "endpoint_ridge", "ic": float(value)} for timestamp, value in zip(expected_clock, endpoint_ic, strict=True))
        manifest_rows.append({"fold": fold.name, "representation": "shared_endpoint", "model": "endpoint_ridge", "seed": "none", "path": os.path.relpath(endpoint_path, output), "sha256": sha256_file(endpoint_path)})
        del panel
        gc.collect()

        for representation, series_id in (("native_1m", "N02"), ("causal_15m_control", "C02")):
            panel, local_fold, _local = frozen_models.materialize_fold(components, fold, representation=representation)
            test = local_fold.test_indices
            predictions_by_seed: dict[int, np.ndarray] = {}
            for seed in map(int, contract["model"]["fixed_seeds"]):
                seed_identity = _identity(
                    contract_hash=contract_hash,
                    component_hash=component_hash,
                    pair_hash=pair_hash,
                    frozen_bundle_hash=frozen_bundle_hash,
                    history_bundle_hash=history_bundle_hash,
                    environment_hash=environment_hash,
                    fold=fold.name,
                    representation=representation,
                    model="itransformer",
                    seed=seed,
                    role="individual_fixed_seed_forecast",
                )
                path = _forecast_path(output, fold=fold.name, representation=representation, model="itransformer", seed=seed)
                if not path.exists():
                    fitted = frozen_models.fit_itransformer(
                        panel,
                        local_fold,
                        config=frozen_models.transformer_config(seed=seed, device=str(args.device)),
                    )
                    prediction = fitted.predict(panel.raw_features[test], panel.zero_auxiliary[test], panel.feature_available[test])
                    diagnostics = {
                        **seed_identity,
                        "epochs": int(fitted.epochs_trained),
                        "training_loss": float(fitted.training_losses[-1]),
                        "validation_loss": float(fitted.validation_losses[-1]),
                        "best_validation_loss": float(fitted.best_validation_loss),
                    }
                    _save_forecast(path, clock=expected_clock, predictions=prediction, pair_targets=None, diagnostics=diagnostics)
                    del fitted
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                prediction, _targets, diagnostics = _load_forecast(
                    path,
                    expected_clock=expected_clock,
                    expected_mask=panel.feature_available[test],
                    expected_identity=seed_identity,
                    require_pair_targets=False,
                    contract=contract,
                )
                if int(diagnostics.get("epochs", 0)) < 1 or not np.isfinite([diagnostics.get("training_loss", np.nan), diagnostics.get("validation_loss", np.nan)]).all():
                    raise ValueError(f"invalid seed diagnostics: {path}")
                predictions_by_seed[seed] = prediction
                seed_ic = frozen_models.decision_ic(prediction, panel.target_cashflows[test], panel.label_mask[test])
                seed_rows.append({"fold": fold.name, "series_id": series_id, "representation": representation, "model": "itransformer", "seed": seed, "mean_ic": float(np.nanmean(seed_ic)), "finite_ic_decisions": int(np.isfinite(seed_ic).sum()), "epochs": int(diagnostics["epochs"]), "training_loss": float(diagnostics["training_loss"]), "validation_loss": float(diagnostics["validation_loss"]), "best_validation_loss": float(diagnostics["best_validation_loss"])})
                manifest_rows.append({"fold": fold.name, "representation": representation, "model": "itransformer", "seed": str(seed), "path": os.path.relpath(path, output), "sha256": sha256_file(path)})

            ensemble = frozen_models.mean_seed_predictions(predictions_by_seed)
            ensemble_identity = _identity(
                contract_hash=contract_hash,
                component_hash=component_hash,
                pair_hash=pair_hash,
                frozen_bundle_hash=frozen_bundle_hash,
                history_bundle_hash=history_bundle_hash,
                environment_hash=environment_hash,
                fold=fold.name,
                representation=representation,
                model="itransformer_seed_mean",
                seed="all3",
                role="fixed_three_seed_arithmetic_mean",
            )
            ensemble_path = _forecast_path(output, fold=fold.name, representation=representation, model="itransformer_seed_mean", seed="all3")
            if not ensemble_path.exists():
                targets = frozen_models.cycle.rank_pair_targets(
                    ensemble,
                    panel.feature_available[test],
                    long_pairs=8,
                    short_pairs=8,
                    absolute_pair_target=0.05,
                )
                _save_forecast(
                    ensemble_path,
                    clock=expected_clock,
                    predictions=ensemble,
                    pair_targets=targets,
                    diagnostics={**ensemble_identity, "seeds": list(frozen_models.FIXED_SEEDS), "aggregation": "arithmetic_mean_never_best_seed"},
                )
            ensemble, _targets, diagnostics = _load_forecast(
                ensemble_path,
                expected_clock=expected_clock,
                expected_mask=panel.feature_available[test],
                expected_identity=ensemble_identity,
                require_pair_targets=True,
                contract=contract,
            )
            if diagnostics.get("seeds") != list(frozen_models.FIXED_SEEDS) or diagnostics.get("aggregation") != "arithmetic_mean_never_best_seed":
                raise ValueError("ensemble aggregation diagnostics differ")
            ensemble_ic = frozen_models.decision_ic(ensemble, panel.target_cashflows[test], panel.label_mask[test])
            metric_rows.append({"fold": fold.name, "series_id": series_id, "representation": representation, "model": "itransformer", "mean_ic": float(np.nanmean(ensemble_ic)), "finite_ic_decisions": int(np.isfinite(ensemble_ic).sum())})
            ic_rows.extend({"timestamp": timestamp.isoformat(), "fold": fold.name, "series_id": series_id, "representation": representation, "model": "itransformer", "ic": float(value)} for timestamp, value in zip(expected_clock, ensemble_ic, strict=True))
            manifest_rows.append({"fold": fold.name, "representation": representation, "model": "itransformer_seed_mean", "seed": "all3", "path": os.path.relpath(ensemble_path, output), "sha256": sha256_file(ensemble_path)})
            del panel, predictions_by_seed, ensemble
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fold_count = len(contract["folds"])
    if len(manifest_rows) != fold_count * 9 or len(metric_rows) != fold_count * 3 or len(seed_rows) != fold_count * 6:
        raise AssertionError("history forecast family counts differ")
    frames = {
        "forecast_manifest.csv": pd.DataFrame(manifest_rows).sort_values(["fold", "representation", "model", "seed"]),
        "fold_metrics.csv": pd.DataFrame(metric_rows).sort_values(["fold", "series_id"]),
        "seed_metrics.csv": pd.DataFrame(seed_rows).sort_values(["fold", "series_id", "seed"]),
        "ic_by_decision.csv": pd.DataFrame(ic_rows).sort_values(["timestamp", "series_id"]),
        "attempt_ledger.csv": pd.DataFrame([
            {"series_id": "B00", "role": "descriptive_endpoint_benchmark", "status": "forecast_complete"},
            {"series_id": "N02", "role": "primary_native_1m_itransformer", "status": "forecast_complete"},
            {"series_id": "C02", "role": "matched_causal_15m_control", "status": "forecast_complete"},
        ]),
    }
    for name, frame in frames.items():
        _atomic_csv(output / name, frame)
    manifest = {
        "schema": RUN_SCHEMA,
        "status": "complete_retrospective_forecasts_evaluation_pending",
        "sample_role": contract["claim_boundary"]["sample_role"],
        "not_confirmatory": True,
        "history_contract_sha256": contract_hash,
        "component_sha256": component_hash,
        "pair_axis_sha256": pair_hash,
        "frozen_source_sha256": frozen_hashes,
        "frozen_source_bundle_sha256": frozen_bundle_hash,
        "history_source_sha256": history_hashes,
        "history_source_bundle_sha256": history_bundle_hash,
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_hash,
        "fold_count": fold_count,
        "series": list(SERIES),
        "fixed_seeds": list(frozen_models.FIXED_SEEDS),
        "seed_selection": False,
        "declared_fit_count": fold_count * 7,
        "forecast_artifact_count": len(manifest_rows),
        "fold_metric_rows": len(metric_rows),
        "seed_metric_rows": len(seed_rows),
        "preflight": preflight,
        "summary_sha256": {name: sha256_file(output / name) for name in frames},
        "confirmation_parsed": False,
    }
    _atomic_json(output / "run_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _load_evaluation_forecasts(
    *, forecast_output: Path, components: Any, component_path: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(forecast_output).resolve()
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("schema") != RUN_SCHEMA or run_manifest.get("confirmation_parsed") is not False:
        raise ValueError("history forecast run manifest differs")
    if run_manifest.get("history_contract_sha256") != sha256_file(CONTRACT_PATH) or run_manifest.get("component_sha256") != sha256_file(component_path):
        raise ValueError("history forecast run input binding differs")
    current_frozen = validate_frozen_sources(contract)
    current_history = history_source_hashes()
    if run_manifest.get("frozen_source_sha256") != current_frozen or run_manifest.get("frozen_source_bundle_sha256") != bundle_sha256(current_frozen):
        raise ValueError("history forecast run frozen-source binding differs")
    if run_manifest.get("history_source_sha256") != current_history or run_manifest.get("history_source_bundle_sha256") != bundle_sha256(current_history):
        raise ValueError("history forecast run local-source binding differs")
    environment = run_manifest.get("runtime_environment")
    if not isinstance(environment, dict) or run_manifest.get("runtime_environment_sha256") != hashlib.sha256(canonical_json(environment)).hexdigest():
        raise ValueError("history forecast runtime binding differs")
    expected_counts = {
        "fold_count": len(contract["folds"]),
        "declared_fit_count": len(contract["folds"]) * 7,
        "forecast_artifact_count": len(contract["folds"]) * 9,
        "fold_metric_rows": len(contract["folds"]) * 3,
        "seed_metric_rows": len(contract["folds"]) * 6,
    }
    if any(run_manifest.get(field) != value for field, value in expected_counts.items()):
        raise ValueError("history forecast run family counts differ")
    if run_manifest.get("series") != list(SERIES) or run_manifest.get("fixed_seeds") != list(frozen_models.FIXED_SEEDS) or run_manifest.get("seed_selection") is not False:
        raise ValueError("history forecast series or seed policy differs")
    expected_summaries = {"forecast_manifest.csv", "fold_metrics.csv", "seed_metrics.csv", "ic_by_decision.csv", "attempt_ledger.csv"}
    summary_hashes = run_manifest.get("summary_sha256")
    if not isinstance(summary_hashes, dict) or set(summary_hashes) != expected_summaries:
        raise ValueError("history forecast summary family differs")
    for name, digest in summary_hashes.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"history forecast summary hash differs: {name}")
    manifest = pd.read_csv(root / "forecast_manifest.csv", dtype=str, keep_default_na=False)
    if manifest.duplicated(["fold", "representation", "model", "seed"]).any() or len(manifest) != len(contract["folds"]) * 9:
        raise ValueError("history forecast manifest family differs")
    indexed = manifest.set_index(["fold", "representation", "model", "seed"])
    expected_keys: set[tuple[str, str, str, str]] = set()
    for spec in contract["folds"]:
        fold = str(spec["name"])
        expected_keys.add((fold, "shared_endpoint", "endpoint_ridge", "none"))
        for representation in REPRESENTATIONS:
            expected_keys.update(
                (fold, representation, "itransformer", str(seed))
                for seed in frozen_models.FIXED_SEEDS
            )
            expected_keys.add((fold, representation, "itransformer_seed_mean", "all3"))
    observed_keys = set(map(tuple, indexed.index.tolist()))
    if observed_keys != expected_keys or manifest["path"].duplicated().any():
        raise ValueError("history forecast manifest is not the exact declared family")
    for key, row in indexed.iterrows():
        path = (root / str(row["path"])).resolve()
        expected_path = _forecast_path(
            root,
            fold=str(key[0]),
            representation=str(key[1]),
            model=str(key[2]),
            seed=str(key[3]),
        )
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("history forecast path escapes output") from error
        if path != expected_path or not path.is_file() or sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"history forecast manifest artifact differs: {key}")
    fold_clocks = expected_fold_clocks(contract)
    component_lookup = pd.Series(np.arange(len(components.clock), dtype=np.int64), index=components.clock)
    stitched: dict[str, list[np.ndarray]] = {series_id: [] for series_id in SERIES}
    consumed: set[tuple[str, str, str, str]] = set()
    for spec in contract["folds"]:
        fold = str(spec["name"])
        clock = fold_clocks[fold]
        locations = component_lookup.reindex(clock)
        if locations.isna().any():
            raise ValueError(f"component cache lacks forecast clock: {fold}")
        mask = components.feature_available[locations.to_numpy(dtype=np.int64)]
        for series_id, (_series, representation, model, seed) in SERIES.items():
            key = (fold, representation, model, seed)
            if key not in indexed.index:
                raise ValueError(f"history forecast manifest lacks {key}")
            row = indexed.loc[key]
            path = (root / str(row["path"])).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("history forecast path escapes output") from error
            if not path.is_file() or sha256_file(path) != str(row["sha256"]):
                raise ValueError(f"history forecast hash differs: {path}")
            consumed.add(key)
            role = "descriptive_endpoint_only_baseline" if series_id == "B00" else "fixed_three_seed_arithmetic_mean"
            identity = _identity(
                contract_hash=run_manifest["history_contract_sha256"],
                component_hash=run_manifest["component_sha256"],
                pair_hash=run_manifest["pair_axis_sha256"],
                frozen_bundle_hash=run_manifest["frozen_source_bundle_sha256"],
                history_bundle_hash=run_manifest["history_source_bundle_sha256"],
                environment_hash=run_manifest["runtime_environment_sha256"],
                fold=fold,
                representation=representation,
                model=model,
                seed=seed,
                role=role,
            )
            _predictions, targets, _diagnostics = _load_forecast(
                path,
                expected_clock=clock,
                expected_mask=mask,
                expected_identity=identity,
                require_pair_targets=True,
                contract=contract,
            )
            assert targets is not None
            stitched[series_id].append(targets)
    if len(consumed) != len(contract["folds"]) * 3:
        raise AssertionError("history evaluation did not consume all declared ensemble folds")
    result = {series_id: np.concatenate(parts, axis=0) for series_id, parts in stitched.items()}
    expected_shape = (len(expected_decision_clock(contract)), len(components.pair_ids))
    if any(values.shape != expected_shape for values in result.values()):
        raise ValueError("stitched history forecast axes differ")
    return result, run_manifest


def _continuous_daily_attribution(ledger: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily compounding for a terminal row that remains inside its UTC day."""
    bars = ledger.bars.copy()
    clock = pd.DatetimeIndex(bars.index).tz_convert("UTC")
    keys = clock.normalize()
    prior_growth = np.empty(len(bars), dtype=float)
    running = 1.0
    previous: pd.Timestamp | None = None
    for row, (date, net) in enumerate(zip(keys, bars["net_return"], strict=True)):
        if previous is None or date != previous:
            running = 1.0
        prior_growth[row] = running
        running *= 1.0 + float(net)
        previous = date
    hourly = pd.DataFrame(
        {
            "price_return_contribution": bars["price_return"].to_numpy(dtype=float) * prior_growth,
            "funding_return_contribution": bars["funding_return"].to_numpy(dtype=float) * prior_growth,
            "transaction_cost_contribution": -bars["transaction_cost"].to_numpy(dtype=float) * prior_growth,
        },
        index=clock,
    )
    hourly["net_return_contribution"] = hourly.sum(axis=1)
    daily = hourly.groupby(keys).sum()
    daily["net_return"] = (1.0 + bars["net_return"]).groupby(keys).prod() - 1.0
    daily["gross_return"] = (1.0 + bars["gross_return"]).groupby(keys).prod() - 1.0
    daily["turnover"] = bars["turnover"].groupby(keys).sum()
    daily["trading_cost"] = bars["transaction_cost"].groupby(keys).sum()
    daily.index.name = "date"
    if not np.allclose(daily["net_return"], daily["net_return_contribution"], rtol=1e-12, atol=1e-14):
        raise ValueError("continuous daily return attribution does not reconcile")
    coin = ledger.coin_ledger.reset_index().copy()
    locations = clock.get_indexer(pd.DatetimeIndex(coin["timestamp"]))
    if (locations < 0).any():
        raise ValueError("coin ledger timestamp lies outside continuous clock")
    coin["date"] = keys.take(locations).to_numpy()
    coin["price_return_contribution"] = coin["price_return"].to_numpy(dtype=float) * prior_growth[locations]
    coin["funding_return_contribution"] = coin["funding_return"].to_numpy(dtype=float) * prior_growth[locations]
    coin["transaction_cost_contribution"] = -coin["transaction_cost"].to_numpy(dtype=float) * prior_growth[locations]
    coin["net_return_contribution"] = coin[["price_return_contribution", "funding_return_contribution", "transaction_cost_contribution"]].sum(axis=1)
    daily_coin = coin.groupby(["date", "coin"], as_index=False)[["price_return_contribution", "funding_return_contribution", "transaction_cost_contribution", "net_return_contribution"]].sum()
    reconciled = daily_coin.groupby("date")["net_return_contribution"].sum()
    if not np.allclose(reconciled.reindex(daily.index), daily["net_return"], rtol=1e-12, atol=1e-14):
        raise ValueError("continuous coin attribution does not reconcile")
    return daily, daily_coin


def command_evaluate(args: argparse.Namespace) -> int:
    contract = load_contract()
    components, _component_metadata, preflight = preflight_inputs(
        components_path=args.components,
        economic_inputs=args.economic_inputs,
        economic_manifest=args.economic_manifest,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("history evaluation output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    pair_targets_by_series, forecast_manifest = _load_evaluation_forecasts(
        forecast_output=args.forecast_output,
        components=components,
        component_path=args.components,
        contract=contract,
    )
    economic = frozen_evaluation.load_economic_inputs(args.economic_inputs, args.economic_manifest)
    dependencies = frozen_evaluation.install_audited_dependencies(args.pairs_research_root)
    decision_clock = expected_decision_clock(contract)
    terminal = utc(contract["data"]["boundary_policy"]["terminal_zero_target"])
    clock = decision_clock.append(pd.DatetimeIndex([terminal]))
    if clock[-1] >= HARD_CUTOFF_EXCLUSIVE or not clock.equals(pd.date_range(clock[0], clock[-1], freq="1h")):
        raise PermissionError("continuous execution clock is not development-only and contiguous")
    economic_lookup = pd.Series(np.arange(len(economic.clock), dtype=np.int64), index=economic.clock)
    economic_locations = economic_lookup.reindex(clock)
    if economic_locations.isna().any():
        raise ValueError("economic artifact lacks continuous execution clock")
    economic_locations_array = economic_locations.to_numpy(dtype=np.int64)
    returns = pd.DataFrame(economic.coin_open_to_open_returns[economic_locations_array], index=clock, columns=economic.symbols)
    availability = pd.DataFrame(economic.coin_decision_available[economic_locations_array], index=clock, columns=economic.symbols)
    # The final development row has no permitted t+1 return.  It applies known
    # boundary funding, is marked unavailable, and carries a zero return so the
    # audited ledger closes every position without fabricating a shortened hour.
    returns.iloc[-1] = 0.0
    availability.iloc[-1] = False
    funding = pd.DataFrame(economic.funding_at_boundary[economic_locations_array], index=clock, columns=economic.symbols)
    opens = pd.DataFrame(economic.coin_open_prices, index=economic.clock, columns=economic.symbols)
    risk = dependencies.estimate_daily_risk_inputs(
        opens,
        start=decision_clock[0],
        end_exclusive=terminal,
        lookback_days=frozen_evaluation.RISK_LOOKBACK_DAYS,
        minimum_complete_observations=frozen_evaluation.RISK_MINIMUM_COMPLETE_HOURS,
    )
    beta = risk.beta_frame(clock).reindex(columns=list(components.symbols))
    covariance = frozen_evaluation._asof_covariance(risk.covariance_matrices, clock, components.symbols)
    portfolio = contract["portfolio"]
    constraints = dependencies.PortfolioConstraints(
        maximum_total_gross=float(portfolio["maximum_total_gross"]),
        maximum_absolute_coin_weight=float(portfolio["maximum_absolute_coin_weight"]),
        maximum_absolute_dollar_net=float(portfolio["maximum_absolute_dollar_net"]),
        maximum_absolute_btc_beta=float(portfolio["maximum_absolute_btc_beta"]),
        target_annualized_volatility=float(portfolio["target_annualized_volatility"]),
    )
    daily_parts: list[pd.DataFrame] = []
    coin_parts: list[pd.DataFrame] = []
    risk_rows: list[dict[str, Any]] = []
    hourly_parts: list[pd.DataFrame] = []
    ledgers = 0
    for series_id, (series_name, _representation, _model, _seed) in SERIES.items():
        decision_targets = pd.DataFrame(pair_targets_by_series[series_id], index=decision_clock, columns=components.pair_ids, dtype=float)
        pair_targets = decision_targets.reindex(clock, fill_value=0.0)
        if not np.array_equal(pair_targets.iloc[-1].to_numpy(dtype=float), np.zeros(len(components.pair_ids))):
            raise AssertionError("history terminal pair target is not zero")
        if np.any((np.abs(pair_targets_by_series[series_id]) > 0.0) & ~economic.pair_decision_available[economic_locations_array[:-1]]):
            raise ValueError("history forecast trades outside economic availability")
        hedge = frozen_evaluation.economic_hedge_tensor(economic, economic_locations_array, pair_targets.to_numpy(dtype=float))
        for cost in map(float, portfolio["cost_bps"]):
            ledger = dependencies.backtest_multicoin_pair_ledger(
                pair_targets,
                hedge,
                returns,
                funding,
                transaction_cost_bps=cost,
                coin_symbols=tuple(components.symbols),
                availability=availability,
                constraints=constraints,
                btc_beta_loadings=beta,
                covariance_matrices=risk.covariance_matrices,
                liquidate_at_end=False,
            )
            ledgers += 1
            bars = ledger.bars.copy()
            frozen_evaluation._validate_ledger_attribution(bars, cost)
            terminal_close = ledger.coin_ledger.reset_index().loc[lambda frame: frame["timestamp"].eq(terminal), "close_notional"]
            if not np.allclose(terminal_close.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-12):
                raise ValueError("continuous history ledger is not flat at the firewall")
            daily, daily_coin = _continuous_daily_attribution(ledger)
            daily = daily.reset_index().rename(columns={"index": "date"})
            daily["fold"] = fold_for_dates(daily["date"], contract)
            daily.insert(0, "cost_bps", cost)
            daily.insert(0, "series", series_name)
            daily.insert(0, "series_id", series_id)
            daily["nav"] = (1.0 + daily["net_return"]).cumprod()
            daily["sample_role"] = contract["claim_boundary"]["sample_role"]
            daily_parts.append(daily)
            daily_coin["fold"] = fold_for_dates(daily_coin["date"], contract)
            daily_coin.insert(0, "cost_bps", cost)
            daily_coin.insert(0, "series", series_name)
            daily_coin.insert(0, "series_id", series_id)
            coin_parts.append(daily_coin)
            audit, summary = frozen_evaluation._risk_audit(
                projection=ledger.projection,
                bars=bars,
                beta=beta,
                covariance=covariance,
                constraints=constraints,
            )
            risk_rows.append({"scope": "aggregate", "series_id": series_id, "series": series_name, "cost_bps": cost, **summary})
            if np.isclose(cost, float(portfolio["primary_cost_bps"]), rtol=0.0, atol=1e-12):
                audit.insert(0, "fold", fold_for_dates(audit["timestamp"], contract))
                audit.insert(0, "cost_bps", cost)
                audit.insert(0, "series", series_name)
                audit.insert(0, "series_id", series_id)
                hourly_parts.append(audit)
    if ledgers != len(SERIES) * len(portfolio["cost_bps"]):
        raise AssertionError("history evaluation ledger count differs")

    daily = pd.concat(daily_parts, ignore_index=True).sort_values(["series_id", "cost_bps", "date"]).reset_index(drop=True)
    coin = pd.concat(coin_parts, ignore_index=True).sort_values(["series_id", "cost_bps", "date", "coin"]).reset_index(drop=True)
    hourly = pd.concat(hourly_parts, ignore_index=True).sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    if daily.duplicated(["series_id", "cost_bps", "date"]).any():
        raise ValueError("continuous daily keys are duplicated")
    expected_dates = pd.date_range(decision_clock[0].normalize(), terminal.normalize(), freq="1D")
    for (_series, _cost), group in daily.groupby(["series_id", "cost_bps"]):
        if not pd.DatetimeIndex(group.sort_values("date")["date"]).equals(expected_dates):
            raise ValueError("continuous daily path has a calendar gap")

    quarterly_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for (series_id, cost), group in daily.groupby(["series_id", "cost_bps"], sort=True):
        series_name = SERIES[str(series_id)][0]
        group = group.sort_values("date")
        aggregate_rows.append({
            "scope": "aggregate",
            "series_id": str(series_id),
            "series": series_name,
            "fold": "ALL",
            "cost_bps": float(cost),
            **performance_metrics(group["net_return"], turnover=float(group["turnover"].sum()), gross_sum=float(group["gross_return"].sum())),
            "total_price_return_contribution": float(group["price_return_contribution"].sum()),
            "total_funding_return_contribution": float(group["funding_return_contribution"].sum()),
            "total_transaction_cost_contribution": float(group["transaction_cost_contribution"].sum()),
        })
        for fold, chosen in group.groupby("fold", sort=False):
            quarterly_rows.append({
                "scope": "fold",
                "series_id": str(series_id),
                "series": series_name,
                "fold": str(fold),
                "cost_bps": float(cost),
                **performance_metrics(chosen["net_return"], turnover=float(chosen["turnover"].sum()), gross_sum=float(chosen["gross_return"].sum())),
                "total_price_return_contribution": float(chosen["price_return_contribution"].sum()),
                "total_funding_return_contribution": float(chosen["funding_return_contribution"].sum()),
                "total_transaction_cost_contribution": float(chosen["transaction_cost_contribution"].sum()),
            })
    quarterly = pd.DataFrame(quarterly_rows).sort_values(["series_id", "cost_bps", "fold"]).reset_index(drop=True)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["series_id", "cost_bps"]).reset_index(drop=True)
    loo = leave_one_fold_out_metrics(daily)
    stability = stability_summary(quarterly, loo)

    forecast_ic = pd.read_csv(Path(args.forecast_output) / "fold_metrics.csv")
    quarterly_ic = forecast_ic[["fold", "series_id", "representation", "model", "mean_ic", "finite_ic_decisions"]].copy()
    native_ic = quarterly_ic.loc[quarterly_ic["series_id"].eq("N02")].set_index("fold")["mean_ic"]
    control_ic = quarterly_ic.loc[quarterly_ic["series_id"].eq("C02")].set_index("fold")["mean_ic"]
    if tuple(native_ic.index) != EXPECTED_FOLDS or not native_ic.index.equals(control_ic.index):
        raise ValueError("quarterly IC family differs")
    ic_delta = native_ic - control_ic
    stability["positive_native_minus_control_ic_quarters"] = int((ic_delta > 0.0).sum())
    stability["median_native_minus_control_ic"] = float(ic_delta.median())

    paired_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_spec = contract["diagnostics"]["bootstrap"]
    for cost in map(float, portfolio["cost_bps"]):
        native = daily.loc[daily["series_id"].eq("N02") & np.isclose(daily["cost_bps"], cost, rtol=0.0, atol=1e-12)].set_index("date")
        control = daily.loc[daily["series_id"].eq("C02") & np.isclose(daily["cost_bps"], cost, rtol=0.0, atol=1e-12)].set_index("date")
        if not native.index.equals(control.index):
            raise ValueError("paired continuous daily clocks differ")
        frame = pd.DataFrame({
            "date": native.index,
            "fold": native["fold"].to_numpy(),
            "cost_bps": cost,
            "native_net_return": native["net_return"].to_numpy(dtype=float),
            "control_net_return": control["net_return"].to_numpy(dtype=float),
        })
        frame["difference"] = frame["native_net_return"] - frame["control_net_return"]
        paired_rows.append(frame)
        bootstrap_rows.append({
            "contrast": "N02_native_1m_minus_C02_causal_15m",
            "cost_bps": cost,
            **moving_block_bootstrap(
                frame[["date", "difference"]],
                replications=int(bootstrap_spec["replications"]),
                block_days=int(bootstrap_spec["block_days"]),
                seed=int(str(bootstrap_spec["seed"]), 0),
            ),
        })
    paired = pd.concat(paired_rows, ignore_index=True).sort_values(["cost_bps", "date"]).reset_index(drop=True)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values("cost_bps").reset_index(drop=True)

    primary = float(portfolio["primary_cost_bps"])
    risk_concentration_rows: list[dict[str, Any]] = []
    primary_hourly = hourly.loc[np.isclose(hourly["cost_bps"], primary, rtol=0.0, atol=1e-12)]
    for (series_id, fold), group in primary_hourly.groupby(["series_id", "fold"], sort=True):
        risk_concentration_rows.append({
            "scope": "fold_weight_concentration",
            "series_id": str(series_id),
            "cost_bps": primary,
            "fold": str(fold),
            "mean_effective_coin_count": float(group["effective_coin_count"].mean()),
            "median_effective_coin_count": float(group["effective_coin_count"].median()),
            "mean_weight_concentration": float(group["weight_concentration"].mean()),
            "maximum_weight_concentration": float(group["weight_concentration"].max()),
            "mean_total_gross": float(group["total_gross"].mean()),
            "maximum_absolute_coin_weight": float(group["maximum_absolute_coin_weight"].max()),
        })
    for series_id, group in primary_hourly.groupby("series_id", sort=True):
        risk_concentration_rows.append({
            "scope": "aggregate_weight_concentration",
            "series_id": str(series_id),
            "cost_bps": primary,
            "fold": "ALL",
            "mean_effective_coin_count": float(group["effective_coin_count"].mean()),
            "median_effective_coin_count": float(group["effective_coin_count"].median()),
            "mean_weight_concentration": float(group["weight_concentration"].mean()),
            "maximum_weight_concentration": float(group["weight_concentration"].max()),
            "mean_total_gross": float(group["total_gross"].mean()),
            "maximum_absolute_coin_weight": float(group["maximum_absolute_coin_weight"].max()),
        })
    risk_concentration = pd.DataFrame(risk_concentration_rows)
    contribution = contribution_concentration(coin)
    contribution.insert(0, "scope", "aggregate_coin_contribution_concentration")
    contribution["fold"] = "ALL"
    concentration = pd.concat([risk_concentration, contribution], ignore_index=True, sort=False).sort_values(["series_id", "cost_bps", "scope", "fold"]).reset_index(drop=True)
    risk_summary = pd.DataFrame(risk_rows).sort_values(["series_id", "cost_bps"]).reset_index(drop=True)
    violations = int(risk_summary["all_constraint_violation_count"].sum())
    if violations != 0:
        raise ValueError(f"history evaluation contains {violations} constraint violations")

    output_frames = {
        "continuous_daily_returns.csv": daily,
        "quarterly_metrics.csv": quarterly,
        "aggregate_metrics.csv": aggregate,
        "quarterly_ic.csv": quarterly_ic,
        "stability_summary.csv": stability,
        "leave_one_quarter_out_metrics.csv": loo,
        "paired_daily_differences.csv": paired,
        "paired_bootstrap.csv": bootstrap,
        "risk_audit_summary.csv": risk_summary,
        "concentration_summary.csv": concentration,
        "coin_contributions.csv": coin,
    }
    for name, frame in output_frames.items():
        _atomic_csv(output / name, frame)
    hourly_path = output / "hourly_constraint_audit.parquet"
    hourly.to_parquet(hourly_path, index=False)
    output_hashes = {name: sha256_file(output / name) for name in output_frames}
    output_hashes[hourly_path.name] = sha256_file(hourly_path)
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete_retrospective_continuous_history_evaluation",
        "sample_role": contract["claim_boundary"]["sample_role"],
        "not_confirmatory": True,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "components_sha256": sha256_file(args.components),
        "economic_inputs_sha256": sha256_file(args.economic_inputs),
        "economic_manifest_sha256": sha256_file(args.economic_manifest),
        "forecast_run_manifest_sha256": sha256_file(Path(args.forecast_output) / "run_manifest.json"),
        "forecast_runtime_environment": forecast_manifest["runtime_environment"],
        "audited_dependency_source_hashes": dict(dependencies.source_hashes),
        "family": {"series": list(SERIES), "folds": list(EXPECTED_FOLDS), "cost_bps": list(map(float, portfolio["cost_bps"])), "continuous_ledger_count": ledgers, "quarter_boundary_liquidations": 0},
        "clock": {"first_decision": decision_clock[0].isoformat(), "last_decision": decision_clock[-1].isoformat(), "terminal_zero_target": terminal.isoformat(), "daily_start": expected_dates[0].isoformat(), "daily_end": expected_dates[-1].isoformat()},
        "terminal_policy": "apply permitted boundary funding, set every coin unavailable with zero forward return, and close the zero target; no shortened pseudo-hour",
        "constraint_violation_count": violations,
        "bootstrap_interpretation": "descriptive_after_model_selection_not_confirmatory",
        "preflight": preflight,
        "output_sha256": output_hashes,
        "confirmation_authorized": False,
        "confirmation_parsed": False,
    }
    _atomic_json(output / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-contract")
    validate.set_defaults(function=command_validate_contract)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--components", type=Path, required=True)
    preflight.add_argument("--economic-inputs", type=Path, required=True)
    preflight.add_argument("--economic-manifest", type=Path, required=True)
    preflight.set_defaults(function=command_preflight)
    train = subparsers.add_parser("train")
    train.add_argument("--components", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.set_defaults(function=command_train)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--components", type=Path, required=True)
    evaluate.add_argument("--economic-inputs", type=Path, required=True)
    evaluate.add_argument("--economic-manifest", type=Path, required=True)
    evaluate.add_argument("--forecast-output", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--pairs-research-root", type=Path, default=frozen_evaluation.DEFAULT_AUDITED_ROOT)
    evaluate.set_defaults(function=command_evaluate)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
