#!/usr/bin/env python3
"""Prepare, smoke-test, and train the frozen paired neural experiment.

The training stage intentionally stops after writing fixed pair targets.  Those
targets are the input to the already audited portfolio/risk ledger.  It never
implements a second, subtly different constraint engine inside this package.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from paired_neural_core import (
    DEVELOPMENT_END_EXCLUSIVE,
    build_components_from_closes,
    load_acquisition_manifest,
    load_components,
    load_development_closes,
    load_hourly_economic_inputs,
    paired_feature_audit,
    save_components,
    sha256_file,
)
from paired_neural_models import (
    FIXED_SEEDS,
    LSTMConfig,
    cycle,
    decision_ic,
    fit_endpoint_ridge,
    fit_itransformer,
    fit_lstm,
    make_walk_forward_fold,
    materialize_fold,
    mean_seed_predictions,
    transformer_config,
)


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contract.json"
CORE_PATH = HERE / "paired_neural_core.py"
MODELS_PATH = HERE / "paired_neural_models.py"
RUNNER_PATH = Path(__file__).resolve()
PARENT_CONTRACT_PATH = HERE.parent / "contracts" / "universe.json"
EXPECTED_PARENT_CONTRACT_SHA256 = (
    "45727a9548600a0f259701a53ab1880106359322a2470ddc176054ad98b021b1"
)
REPRESENTATIONS = ("native_1m", "causal_15m_control")
NEURAL_MODELS = ("lstm", "itransformer")
FORECAST_SCHEMA = "paired_neural_forecast_v2"
COMPONENT_SCHEMA = "paired_1m_factorized_components_v2"
FORECAST_IDENTITY_KEYS = {
    "schema",
    "contract_sha256",
    "component_sha256",
    "pair_axis_sha256",
    "core_source_sha256",
    "models_source_sha256",
    "runner_source_sha256",
    "source_bundle_sha256",
    "runtime_environment_sha256",
    "economic_artifact_sha256",
    "economic_manifest_sha256",
    "fold",
    "representation",
    "model",
    "seed",
    "role",
}


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def pair_axis_sha256(pair_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        (json.dumps(list(map(str, pair_ids)), separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def current_source_hashes() -> dict[str, str]:
    """Every local/frozen source whose exact bytes define the experiment."""

    return {
        "contract.json": sha256_file(CONTRACT_PATH),
        "paired_neural_core.py": sha256_file(CORE_PATH),
        "paired_neural_models.py": sha256_file(MODELS_PATH),
        "run_paired_neural.py": sha256_file(RUNNER_PATH),
        "parent_cycle_pilot.py": sha256_file(
            HERE.parent / "model" / "pairtoken_model.py"
        ),
        "parent_raw_4h_contract.json": sha256_file(PARENT_CONTRACT_PATH),
    }


def source_bundle_sha256(source_hashes: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json(dict(source_hashes))).hexdigest()


def configure_determinism(seed: int = FIXED_SEEDS[0]) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cycle.set_reproducible_seed(int(seed))


def runtime_environment(*, device: str) -> dict[str, Any]:
    requested = str(device)
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
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
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if requested == "cuda" else None,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def runtime_environment_sha256(environment: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(environment))).hexdigest()


def peak_process_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; Darwin reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def forecast_identity(
    *,
    contract_sha256: str,
    component_sha256: str,
    pair_axis_sha256_value: str,
    source_hashes: Mapping[str, str],
    source_bundle_sha256_value: str,
    runtime_environment_sha256_value: str,
    economic_artifact_sha256: str,
    economic_manifest_sha256: str,
    fold: str,
    representation: str,
    model: str,
    seed: int | str,
    role: str,
) -> dict[str, Any]:
    return {
        "schema": FORECAST_SCHEMA,
        "contract_sha256": str(contract_sha256),
        "component_sha256": str(component_sha256),
        "pair_axis_sha256": str(pair_axis_sha256_value),
        "core_source_sha256": str(source_hashes["paired_neural_core.py"]),
        "models_source_sha256": str(source_hashes["paired_neural_models.py"]),
        "runner_source_sha256": str(source_hashes["run_paired_neural.py"]),
        "source_bundle_sha256": str(source_bundle_sha256_value),
        "runtime_environment_sha256": str(runtime_environment_sha256_value),
        "economic_artifact_sha256": str(economic_artifact_sha256),
        "economic_manifest_sha256": str(economic_manifest_sha256),
        "fold": str(fold),
        "representation": str(representation),
        "model": str(model),
        "seed": str(seed),
        "role": str(role),
    }


def atomic_create(path: Path, payload: bytes) -> None:
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_create(path, canonical_json(value))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_create(path, frame.to_csv(index=False).encode("utf-8"))


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("contract_name") != "paired_native_1m_vs_causal_15m_neural_development":
        raise ValueError("paired contract name differs")
    if contract.get("contract_version") != 2:
        raise ValueError("paired contract version differs")
    if sha256_file(PARENT_CONTRACT_PATH) != EXPECTED_PARENT_CONTRACT_SHA256:
        raise ValueError("frozen parent raw-4h contract hash differs")
    if contract["frozen_parent"]["raw_4h_contract_sha256"] != EXPECTED_PARENT_CONTRACT_SHA256:
        raise ValueError("paired contract parent binding differs")
    parent_model = HERE.parent / "model" / "pairtoken_model.py"
    if contract["frozen_parent"]["cycle_model_sha256"] != sha256_file(parent_model):
        raise ValueError("paired contract parent model binding differs")
    if tuple(contract["models"]["common_training"]["fixed_seeds"]) != FIXED_SEEDS:
        raise ValueError("paired contract fixed seeds differ")
    if contract["data"]["hard_cutoff_exclusive_utc"] != "2025-07-01T00:00:00Z":
        raise ValueError("paired contract data cutoff differs")
    if contract["required_outputs"]["schemas"] != {
        "components": COMPONENT_SCHEMA,
        "forecast": FORECAST_SCHEMA,
        "forecast_run_manifest": "paired_neural_forecast_run_manifest_v2",
    }:
        raise ValueError("paired contract artifact schemas differ")
    if contract["models"]["lstm"] != {
        "path_input_size": 1,
        "path_hidden_size": 16,
        "path_layers": 1,
        "common_tail_projection": 32,
        "dropout": 0.0,
    }:
        raise ValueError("paired LSTM configuration differs")
    if contract["models"]["itransformer"] != {
        "implementation": "frozen parent PairTokenITransformer",
        "d_model": 48,
        "attention_heads": 4,
        "encoder_layers": 1,
        "feedforward_dimension": 96,
        "dropout": 0.1,
    }:
        raise ValueError("paired iTransformer configuration differs")
    common = contract["models"]["common_training"]
    expected_training = {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "maximum_epochs": 18,
        "early_stopping_patience": 4,
        "minimum_improvement": 1e-6,
    }
    if any(common.get(field) != value for field, value in expected_training.items()):
        raise ValueError("paired common training configuration differs")
    return contract


def static_universe() -> tuple[str, ...]:
    parent = json.loads(PARENT_CONTRACT_PATH.read_text(encoding="utf-8"))
    symbols = tuple(parent["universe"]["static_symbols_in_rank_order"])
    if len(symbols) != 24 or len(set(symbols)) != 24:
        raise ValueError("parent universe is not a static Top-24")
    return symbols


def validate_governed_gap_report(
    report: object, policy: object
) -> dict[str, Any]:
    if not isinstance(report, dict) or not isinstance(policy, dict):
        raise ValueError("transient-open-gap report or contract policy is missing")
    expected_report_keys = {
        "policy",
        "no_fill",
        "no_shortening",
        "feature_and_decision_availability_unchanged",
        "coin_gap_cells",
        "pair_label_cells_masked",
        "oos_coin_gap_cells",
        "oos_pair_label_cells",
        "events",
    }
    if set(report) != expected_report_keys:
        raise ValueError("transient-open-gap report schema differs")
    observed_events = [
        {
            "decision_timestamp": event["decision_timestamp"],
            "missing_next_open_timestamp": event["missing_next_open_timestamp"],
            "symbols": event["symbols"],
        }
        for event in report["events"]
    ]
    checks = {
        "policy": policy.get("rule"),
        "coin_gap_cells": policy.get("expected_coin_gap_cells"),
        "pair_label_cells_masked": policy.get("expected_pair_label_cells_masked"),
        "oos_coin_gap_cells": policy.get("maximum_oos_coin_gap_cells"),
        "oos_pair_label_cells": policy.get("maximum_oos_pair_label_cells"),
    }
    if any(report[field] != expected for field, expected in checks.items()) or any(
        report[field] is not True
        for field in (
            "no_fill",
            "no_shortening",
            "feature_and_decision_availability_unchanged",
        )
    ):
        raise ValueError("observed transient open gaps differ from the paired contract")
    if observed_events != policy.get("expected_events"):
        raise ValueError("transient-open-gap event identities differ from the paired contract")
    if any(
        set(event) != {
            "decision_timestamp",
            "missing_next_open_timestamp",
            "symbols",
            "coin_gap_cells",
            "pair_label_cells_masked",
        }
        for event in report["events"]
    ):
        raise ValueError("transient-open-gap event schema differs")
    return dict(report)


def prepare_components(args: argparse.Namespace) -> int:
    contract = load_contract()
    contract_hash = sha256_file(CONTRACT_PATH)
    economic, economic_manifest = load_hourly_economic_inputs(
        args.economic_inputs, args.economic_manifest
    )
    if economic_manifest.get("contract_sha256") != contract_hash:
        raise ValueError("economic input is not bound to the current paired contract")
    transient_report = validate_governed_gap_report(
        economic_manifest.get("transient_open_gap_report"),
        contract["data"].get("transient_open_gap_policy"),
    )
    manifest = load_acquisition_manifest(args.acquisition_manifest, require_full_contract=True)
    manifest_bindings = {
        "acquisition_manifest_sha256": sha256_file(
            Path(args.acquisition_manifest) / "acquisition_manifest.json"
        ),
        "archive_availability_sha256": sha256_file(
            Path(args.acquisition_manifest) / "archive_availability.csv"
        ),
        "artifact_hashes_sha256": sha256_file(
            Path(args.acquisition_manifest) / "artifact_hashes.json"
        ),
    }
    for field, observed in manifest_bindings.items():
        if contract["data"].get(field) != observed:
            raise ValueError(f"production acquisition binding differs: {field}")
    if Path(args.acquisition_manifest).resolve().name != contract["data"][
        "required_acquisition_manifest_directory_name"
    ]:
        raise ValueError("production acquisition directory name differs")
    symbols = static_universe()
    if manifest.universe != symbols:
        raise ValueError("acquisition and parent universe axes differ")
    if economic.symbols != symbols or len(economic.pair_ids) != 276:
        raise ValueError("economic input axes differ from the fixed universe")
    requested_start = pd.Timestamp(args.start)
    requested_stop = pd.Timestamp(args.end_exclusive)
    if requested_start != economic.clock[0] or requested_stop != (
        economic.clock[-1] + pd.Timedelta(hours=1)
    ):
        raise ValueError("raw component window must exactly equal the economic input clock")
    closes = load_development_closes(
        args.cache_root,
        manifest_directory=args.acquisition_manifest,
        symbols=symbols,
        start=args.start,
        end_exclusive=args.end_exclusive,
        require_full_contract_manifest=True,
    )
    components = build_components_from_closes(
        closes,
        economic_inputs=economic,
    )
    if components.symbols != symbols or len(components.pair_ids) != 276:
        raise ValueError("prepared component axes differ from the fixed universe")
    audit_rows = np.flatnonzero(components.feature_available.any(axis=1))
    if len(audit_rows) == 0:
        raise ValueError("prepared panel contains no usable feature rows")
    audit = paired_feature_audit(components, audit_rows[: min(24, len(audit_rows))])
    sources = current_source_hashes()
    environment = runtime_environment(device="cpu")
    provenance: dict[str, Any] = {
        "paired_contract_sha256": contract_hash,
        "parent_contract_sha256": sha256_file(PARENT_CONTRACT_PATH),
        **manifest_bindings,
        "economic_artifact_sha256": sha256_file(args.economic_inputs),
        "economic_manifest_sha256": sha256_file(args.economic_manifest),
        "economic_manifest_schema": economic_manifest["schema"],
        "economic_metadata_schema": economic.metadata["schema"],
        "economic_audited_source_hashes": economic_manifest["audited_source_hashes"],
        "transient_open_gap_report": transient_report,
        "source_hashes": sources,
        "source_bundle_sha256": source_bundle_sha256(sources),
        "preparation_environment": environment,
        "preparation_environment_sha256": runtime_environment_sha256(environment),
    }
    if not np.array_equal(components.target_cashflows, economic.pair_target_cashflows):
        raise ValueError("component targets differ from the audited economic artifact")
    if not np.array_equal(components.target_available, economic.pair_target_available):
        raise ValueError("component target mask differs from the audited economic artifact")
    save_components(args.output, components, provenance=provenance)
    receipt = {
        "schema": "paired_1m_factorized_components_receipt_v2",
        "status": "factorized_top24_components_v2_complete",
        "component_cache": str(Path(args.output).resolve()),
        "component_cache_sha256": sha256_file(args.output),
        "clock_start": components.clock[0].isoformat(),
        "clock_end": components.clock[-1].isoformat(),
        "symbols": len(components.symbols),
        "pairs": len(components.pair_ids),
        "feature_available_fraction": float(components.feature_available.mean()),
        "target_available_fraction": float(components.target_available.mean()),
        "paired_feature_audit": audit,
        "target_adapter": {
            "economic_artifact_sha256": provenance["economic_artifact_sha256"],
            "economic_manifest_sha256": provenance["economic_manifest_sha256"],
            "target_values_exactly_equal": True,
            "target_mask_exactly_equal": True,
            "one_period_ledger_identity": economic_manifest["one_period_ledger_identity"],
            "multi_period_ledger_identity": economic_manifest["multi_period_ledger_identity"],
            "transient_open_gap_report": transient_report,
        },
        "provenance": provenance,
        "confirmation_parsed": False,
    }
    atomic_json(Path(args.output).with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _small_smoke_fold(components: Any) -> Any:
    usable = np.flatnonzero(
        components.feature_available.all(axis=1) & components.target_available.all(axis=1)
    )
    if len(usable) < 180:
        raise ValueError("BTC/ETH smoke panel needs at least 180 complete decisions")
    selected = usable[-180:]
    return cycle.PilotFold("btceth_smoke", selected[:80], selected[80:130], selected[130:])


def smoke_existing(args: argparse.Namespace) -> int:
    load_contract()
    closes = load_development_closes(
        args.cache_root,
        manifest_directory=args.acquisition_manifest,
        symbols=("BTCUSDT", "ETHUSDT"),
        start=args.start,
        end_exclusive=args.end_exclusive,
        require_full_contract_manifest=False,
    )
    components = build_components_from_closes(
        closes,
        pairs=(("ETHUSDT", "BTCUSDT"),),
        allow_synthetic_target=True,
    )
    fold = _small_smoke_fold(components)
    audit = paired_feature_audit(components, fold.test_indices[:24])
    reports: list[dict[str, Any]] = []
    ridge_prediction: np.ndarray | None = None
    for representation in REPRESENTATIONS:
        panel, local_fold, _ = materialize_fold(
            components, fold, representation=representation
        )
        ridge = fit_endpoint_ridge(panel, local_fold)
        test = local_fold.test_indices
        prediction = ridge.predict(panel.raw_features[test], panel.feature_available[test])
        if ridge_prediction is None:
            ridge_prediction = prediction
        elif not np.allclose(ridge_prediction, prediction, equal_nan=True, atol=0.0, rtol=0.0):
            raise AssertionError("shared endpoint ridge changed across representations")
        reports.append(
            {
                "representation": representation,
                "model": "endpoint_ridge",
                "seed": None,
                "prediction_count": int(np.isfinite(prediction).sum()),
            }
        )
        lstm = fit_lstm(
            panel,
            local_fold,
            config=LSTMConfig(
                maximum_epochs=1,
                early_stopping_patience=1,
                batch_size=32,
                seed=FIXED_SEEDS[0],
                device="cpu",
            ),
        )
        lstm_prediction = lstm.predict(
            panel.raw_features[test], panel.feature_available[test]
        )
        reports.append(
            {
                "representation": representation,
                "model": "lstm",
                "seed": FIXED_SEEDS[0],
                "prediction_count": int(np.isfinite(lstm_prediction).sum()),
                "training_loss": float(lstm.training_losses[-1]),
                "validation_loss": float(lstm.validation_losses[-1]),
            }
        )
        transformer = fit_itransformer(
            panel,
            local_fold,
            config=transformer_config(
                seed=FIXED_SEEDS[0],
                device="cpu",
                maximum_epochs=1,
                early_stopping_patience=1,
                batch_size=32,
            ),
        )
        transformer_prediction = transformer.predict(
            panel.raw_features[test],
            panel.zero_auxiliary[test],
            panel.feature_available[test],
        )
        reports.append(
            {
                "representation": representation,
                "model": "itransformer",
                "seed": FIXED_SEEDS[0],
                "prediction_count": int(np.isfinite(transformer_prediction).sum()),
                "training_loss": float(transformer.training_losses[-1]),
                "validation_loss": float(transformer.validation_losses[-1]),
            }
        )
        del panel, lstm, transformer
    payload = {
        "status": "btceth_data_and_training_smoke_passed",
        "sample_role": "mechanical_smoke_only_no_cross_sectional_or_portfolio_claim",
        "clock_start": components.clock[0].isoformat(),
        "clock_end": components.clock[-1].isoformat(),
        "decision_semantics": "clock t predicts pair cashflow [t,t+1h)",
        "feature_semantics": "path t-1 and endpoint history strictly before t",
        "paired_feature_audit": audit,
        "runs": reports,
        "confirmation_parsed": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _forecast_path(output: Path, *, fold: str, representation: str, model: str, seed: int | str) -> Path:
    return Path(output) / "forecasts" / fold / f"{representation}__{model}__seed-{seed}.npz"


def _save_forecast(
    path: Path,
    *,
    clock: pd.DatetimeIndex,
    predictions: np.ndarray,
    diagnostics: Mapping[str, Any],
    pair_targets: np.ndarray | None = None,
) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "clock_ns": clock.asi8,
        "predictions": np.asarray(predictions, dtype=np.float32),
        "diagnostics_json": np.asarray(json.dumps(dict(diagnostics), sort_keys=True, separators=(",", ":"))),
    }
    if pair_targets is not None:
        arrays["pair_targets"] = np.asarray(pair_targets, dtype=np.float32)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_forecast_artifact(
    path: Path,
    *,
    expected_clock: pd.DatetimeIndex,
    expected_feature_mask: np.ndarray,
    expected_identity: Mapping[str, Any],
    allowed_diagnostic_extras: set[str],
    portfolio: Mapping[str, Any],
    require_pair_targets: bool,
    expected_predictions: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    """Fail closed on stale, swapped, or partially trusted resume artifacts."""

    source = Path(path)
    if set(expected_identity) != FORECAST_IDENTITY_KEYS:
        raise ValueError("caller supplied an incomplete forecast identity contract")
    expected_members = {"clock_ns", "predictions", "diagnostics_json"}
    if require_pair_targets:
        expected_members.add("pair_targets")
    try:
        with np.load(source, allow_pickle=False) as stored:
            if set(stored.files) != expected_members:
                raise ValueError(f"forecast member family differs: {source}")
            clock_ns = np.asarray(stored["clock_ns"], dtype=np.int64)
            prediction = np.asarray(stored["predictions"], dtype=np.float32)
            diagnostics = json.loads(str(stored["diagnostics_json"].item()))
            pair_targets = (
                np.asarray(stored["pair_targets"], dtype=np.float32)
                if require_pair_targets
                else None
            )
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid forecast artifact: {source}") from error
    if not np.array_equal(clock_ns, expected_clock.asi8):
        raise ValueError(f"forecast clock differs: {source}")
    mask = np.asarray(expected_feature_mask, dtype=bool)
    if prediction.shape != mask.shape:
        raise ValueError(f"forecast axes differ: {source}")
    if np.isinf(prediction).any() or not np.array_equal(np.isfinite(prediction), mask):
        raise ValueError(f"forecast finite/missing mask differs: {source}")
    if expected_predictions is not None and not np.array_equal(
        prediction, np.asarray(expected_predictions, dtype=np.float32), equal_nan=True
    ):
        raise ValueError(f"forecast prediction content differs from governed rebuild: {source}")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"forecast diagnostics are not a mapping: {source}")
    expected_keys = set(expected_identity) | set(allowed_diagnostic_extras)
    if set(diagnostics) != expected_keys:
        raise ValueError(f"forecast diagnostic schema differs: {source}")
    for field, expected in expected_identity.items():
        if diagnostics.get(field) != expected:
            raise ValueError(f"forecast diagnostic identity differs for {field}: {source}")
    if require_pair_targets:
        assert pair_targets is not None
        if pair_targets.shape != prediction.shape or not np.isfinite(pair_targets).all():
            raise ValueError(f"forecast pair-target axes or values differ: {source}")
        rebuilt = cycle.rank_pair_targets(
            prediction,
            mask,
            long_pairs=int(portfolio["long_pairs"]),
            short_pairs=int(portfolio["short_pairs"]),
            absolute_pair_target=float(portfolio["absolute_pair_target"]),
        ).astype(np.float32)
        if not np.array_equal(pair_targets, rebuilt):
            raise ValueError(f"forecast pair targets differ from governed rebuild: {source}")
    return prediction, diagnostics, pair_targets


BENCHMARK_RECEIPT_KEYS = {
    "schema",
    "status",
    "contract_sha256",
    "component_sha256",
    "pair_axis_sha256",
    "source_hashes",
    "source_bundle_sha256",
    "runtime_environment",
    "runtime_environment_sha256",
    "economic_artifact_sha256",
    "economic_manifest_sha256",
    "fold",
    "representation",
    "model",
    "seed",
    "train_rows",
    "validation_rows",
    "test_rows_materialized_but_never_scored",
    "elapsed_seconds",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "peak_process_rss_bytes",
    "fit_discarded",
    "contains_predictions_targets_ic_or_performance",
    "confirmation_parsed",
}

RUN_MANIFEST_KEYS = {
    "schema",
    "status",
    "contract_sha256",
    "component_cache_sha256",
    "pair_axis_sha256",
    "source_hashes",
    "source_bundle_sha256",
    "runtime_environment",
    "runtime_environment_sha256",
    "economic_artifact_sha256",
    "economic_manifest_sha256",
    "discarded_benchmark_receipt_sha256",
    "fold_count",
    "unique_strategy_series",
    "forecast_artifact_count",
    "fold_metric_rows",
    "seed_metric_rows",
    "fixed_seeds",
    "seed_selection",
    "portfolio_backend_required",
    "portfolio_settings",
    "summary_sha256",
    "confirmation_parsed",
}


def validate_completed_forecast_run(
    output: Path,
    *,
    components: Any,
    contract: Mapping[str, Any],
    contract_hash: str,
    component_hash: str,
    pair_hash: str,
    sources: Mapping[str, str],
    bundle_hash: str,
    environment: Mapping[str, Any],
    environment_hash: str,
    economic_artifact_hash: str,
    economic_manifest_hash: str,
    benchmark_receipt_hash: str,
) -> dict[str, Any]:
    """Validate a completed run deeply enough to make reruns idempotent."""

    root = Path(output).resolve()
    manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(run_manifest) != RUN_MANIFEST_KEYS:
        raise ValueError("completed forecast run-manifest schema differs")
    expected = {
        "schema": "paired_neural_forecast_run_manifest_v2",
        "status": "forecast_stage_v2_complete_portfolio_ledger_pending",
        "contract_sha256": contract_hash,
        "component_cache_sha256": component_hash,
        "pair_axis_sha256": pair_hash,
        "source_hashes": dict(sources),
        "source_bundle_sha256": bundle_hash,
        "runtime_environment": dict(environment),
        "runtime_environment_sha256": environment_hash,
        "economic_artifact_sha256": economic_artifact_hash,
        "economic_manifest_sha256": economic_manifest_hash,
        "discarded_benchmark_receipt_sha256": benchmark_receipt_hash,
        "fold_count": len(contract["folds"]),
        "unique_strategy_series": 5,
        "forecast_artifact_count": 6 * 17,
        "fold_metric_rows": 6 * 5,
        "seed_metric_rows": 6 * 2 * 2 * 3,
        "fixed_seeds": list(FIXED_SEEDS),
        "seed_selection": False,
        "portfolio_backend_required": "the existing audited multicoin pair ledger",
        "portfolio_settings": contract["portfolio"],
        "confirmation_parsed": False,
    }
    for field, value in expected.items():
        if run_manifest.get(field) != value:
            raise ValueError(f"completed forecast run identity differs: {field}")
    expected_summaries = {
        "seed_metrics.csv",
        "fold_metrics.csv",
        "ic_by_decision.csv",
        "forecast_manifest.csv",
        "attempt_ledger.csv",
    }
    summary_hashes = run_manifest["summary_sha256"]
    if not isinstance(summary_hashes, dict) or set(summary_hashes) != expected_summaries:
        raise ValueError("completed forecast summary family differs")
    for name, digest in summary_hashes.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"completed forecast summary differs: {name}")

    forecast_manifest = pd.read_csv(
        root / "forecast_manifest.csv", dtype=str, keep_default_na=False
    )
    keys = ["fold", "representation", "model", "seed"]
    if set(forecast_manifest.columns) != set(keys + ["path", "sha256"]):
        raise ValueError("completed forecast artifact manifest schema differs")
    if len(forecast_manifest) != 6 * 17 or forecast_manifest.duplicated(keys).any():
        raise ValueError("completed forecast artifact manifest count differs")
    indexed = forecast_manifest.set_index(keys)
    consumed: set[tuple[str, str, str, str]] = set()

    def artifact_path(key: tuple[str, str, str, str]) -> Path:
        if key not in indexed.index:
            raise ValueError(f"completed forecast manifest lacks {key}")
        consumed.add(key)
        row = indexed.loc[key]
        candidate = (root / str(row["path"])).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("completed forecast path escapes the output root") from error
        expected_relative = _forecast_path(
            root,
            fold=key[0],
            representation=key[1],
            model=key[2],
            seed=key[3],
        )
        if candidate != expected_relative or not candidate.is_file():
            raise ValueError(f"completed forecast path differs: {key}")
        if sha256_file(candidate) != str(row["sha256"]):
            raise ValueError(f"completed forecast hash differs: {key}")
        return candidate

    for fold_spec in contract["folds"]:
        fold_name = str(fold_spec["name"])
        expected_clock = pd.date_range(
            pd.Timestamp(fold_spec["test_start"]),
            pd.Timestamp(fold_spec["test_end_exclusive"]) - pd.Timedelta(hours=1),
            freq="1h",
        )
        locations = components.clock.get_indexer(expected_clock)
        if (locations < 0).any():
            raise ValueError(f"completed forecast clock lies outside components: {fold_name}")
        mask = components.feature_available[locations]
        ridge_identity = forecast_identity(
            contract_sha256=contract_hash,
            component_sha256=component_hash,
            pair_axis_sha256_value=pair_hash,
            source_hashes=sources,
            source_bundle_sha256_value=bundle_hash,
            runtime_environment_sha256_value=environment_hash,
            economic_artifact_sha256=economic_artifact_hash,
            economic_manifest_sha256=economic_manifest_hash,
            fold=fold_name,
            representation="shared_endpoint",
            model="endpoint_ridge",
            seed="none",
            role="shared_endpoint_only_baseline",
        )
        validate_forecast_artifact(
            artifact_path((fold_name, "shared_endpoint", "endpoint_ridge", "none")),
            expected_clock=expected_clock,
            expected_feature_mask=mask,
            expected_identity=ridge_identity,
            allowed_diagnostic_extras=set(),
            portfolio=contract["portfolio"],
            require_pair_targets=True,
        )
        for representation in REPRESENTATIONS:
            for model in NEURAL_MODELS:
                seeds: dict[int, np.ndarray] = {}
                for seed in FIXED_SEEDS:
                    identity = forecast_identity(
                        contract_sha256=contract_hash,
                        component_sha256=component_hash,
                        pair_axis_sha256_value=pair_hash,
                        source_hashes=sources,
                        source_bundle_sha256_value=bundle_hash,
                        runtime_environment_sha256_value=environment_hash,
                        economic_artifact_sha256=economic_artifact_hash,
                        economic_manifest_sha256=economic_manifest_hash,
                        fold=fold_name,
                        representation=representation,
                        model=model,
                        seed=seed,
                        role="individual_fixed_seed_forecast",
                    )
                    prediction, diagnostics, _ = validate_forecast_artifact(
                        artifact_path((fold_name, representation, model, str(seed))),
                        expected_clock=expected_clock,
                        expected_feature_mask=mask,
                        expected_identity=identity,
                        allowed_diagnostic_extras={
                            "epochs",
                            "training_loss",
                            "validation_loss",
                        },
                        portfolio=contract["portfolio"],
                        require_pair_targets=False,
                    )
                    if int(diagnostics["epochs"]) < 1 or not np.isfinite(
                        (float(diagnostics["training_loss"]), float(diagnostics["validation_loss"]))
                    ).all():
                        raise ValueError("completed seed training diagnostics differ")
                    seeds[seed] = prediction
                ensemble = mean_seed_predictions(seeds)
                ensemble_model = f"{model}_seed_mean"
                ensemble_identity = forecast_identity(
                    contract_sha256=contract_hash,
                    component_sha256=component_hash,
                    pair_axis_sha256_value=pair_hash,
                    source_hashes=sources,
                    source_bundle_sha256_value=bundle_hash,
                    runtime_environment_sha256_value=environment_hash,
                    economic_artifact_sha256=economic_artifact_hash,
                    economic_manifest_sha256=economic_manifest_hash,
                    fold=fold_name,
                    representation=representation,
                    model=ensemble_model,
                    seed="all3",
                    role="fixed_three_seed_arithmetic_mean",
                )
                _, diagnostics, _ = validate_forecast_artifact(
                    artifact_path((fold_name, representation, ensemble_model, "all3")),
                    expected_clock=expected_clock,
                    expected_feature_mask=mask,
                    expected_identity=ensemble_identity,
                    allowed_diagnostic_extras={"seeds", "aggregation"},
                    portfolio=contract["portfolio"],
                    require_pair_targets=True,
                    expected_predictions=ensemble,
                )
                if diagnostics["seeds"] != list(FIXED_SEEDS) or diagnostics[
                    "aggregation"
                ] != "arithmetic_mean_never_best_seed":
                    raise ValueError("completed ensemble diagnostics differ")
    if len(consumed) != len(forecast_manifest):
        raise ValueError("completed forecast manifest contains undeclared artifacts")
    return run_manifest


def validate_benchmark_receipt(
    path: Path,
    *,
    contract_hash: str,
    component_hash: str,
    pair_hash: str,
    sources: Mapping[str, str],
    bundle_hash: str,
    environment: Mapping[str, Any],
    environment_hash: str,
    economic_artifact_hash: str,
    economic_manifest_hash: str,
) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(receipt) != BENCHMARK_RECEIPT_KEYS:
        raise ValueError("discarded benchmark receipt schema differs")
    expected = {
        "schema": "paired_neural_discarded_benchmark_v2",
        "status": "full_fold_engineering_fit_completed_and_discarded",
        "contract_sha256": contract_hash,
        "component_sha256": component_hash,
        "pair_axis_sha256": pair_hash,
        "source_hashes": dict(sources),
        "source_bundle_sha256": bundle_hash,
        "runtime_environment": dict(environment),
        "runtime_environment_sha256": environment_hash,
        "economic_artifact_sha256": economic_artifact_hash,
        "economic_manifest_sha256": economic_manifest_hash,
        "fold": "2023Q2",
        "representation": "native_1m",
        "model": "itransformer",
        "seed": 1701,
        "fit_discarded": True,
        "contains_predictions_targets_ic_or_performance": False,
        "confirmation_parsed": False,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"discarded benchmark identity differs: {field}")
    for field in ("train_rows", "validation_rows", "test_rows_materialized_but_never_scored"):
        if not isinstance(receipt[field], int) or receipt[field] <= 0:
            raise ValueError(f"discarded benchmark row count differs: {field}")
    if not isinstance(receipt["elapsed_seconds"], (int, float)) or receipt[
        "elapsed_seconds"
    ] <= 0.0:
        raise ValueError("discarded benchmark elapsed time is invalid")
    for field in (
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "peak_process_rss_bytes",
    ):
        if not isinstance(receipt[field], int) or receipt[field] < 0:
            raise ValueError(f"discarded benchmark memory field is invalid: {field}")
    return receipt


def benchmark_discarded_fold(args: argparse.Namespace) -> int:
    """Fit the predeclared full fold once, record resources, discard all outcomes."""

    contract = load_contract()
    contract_hash = sha256_file(CONTRACT_PATH)
    component_hash = sha256_file(args.components)
    components, metadata = load_components(args.components)
    provenance = metadata.get("provenance", {})
    sources = current_source_hashes()
    bundle_hash = source_bundle_sha256(sources)
    if (
        not isinstance(provenance, dict)
        or provenance.get("paired_contract_sha256") != contract_hash
        or provenance.get("source_hashes") != sources
        or provenance.get("source_bundle_sha256") != bundle_hash
    ):
        raise ValueError("benchmark components differ from the current contract/source bundle")
    economic_artifact_hash = provenance.get("economic_artifact_sha256")
    economic_manifest_hash = provenance.get("economic_manifest_sha256")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in (economic_artifact_hash, economic_manifest_hash)
    ):
        raise ValueError("benchmark components lack economic target bindings")
    spec = contract["engineering_benchmark"]
    if spec != {
        "required_before_scientific_training": True,
        "fold": "2023Q2",
        "representation": "native_1m",
        "model": "itransformer",
        "seed": 1701,
        "artifact_role": "runtime_and_peak-memory_receipt_only",
        "must_not_store": [
            "predictions",
            "pair_targets",
            "IC",
            "loss",
            "portfolio_returns",
            "performance_metrics",
        ],
        "discard_after_fit": True,
    }:
        raise ValueError("engineering benchmark contract differs")
    device = str(args.device)
    configure_determinism(seed=int(spec["seed"]))
    environment = runtime_environment(device=device)
    environment_hash = runtime_environment_sha256(environment)
    fold_spec = next(row for row in contract["folds"] if row["name"] == spec["fold"])
    fold = make_walk_forward_fold(
        components,
        name=fold_spec["name"],
        test_start=fold_spec["test_start"],
        test_end_exclusive=fold_spec["test_end_exclusive"],
        training_days=int(contract["walk_forward"]["training_days"]),
        validation_days=int(contract["walk_forward"]["validation_days"]),
        purge_hours=int(contract["walk_forward"]["target_purge_hours"]),
    )
    panel, local_fold, _ = materialize_fold(
        components, fold, representation=str(spec["representation"])
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fitted = fit_itransformer(
        panel,
        local_fold,
        config=transformer_config(seed=int(spec["seed"]), device=device),
    )
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0
    peak_rss = peak_process_rss_bytes()
    # Deliberately do not call predict and do not inspect/store fit losses.
    del fitted, panel
    if device == "cuda":
        torch.cuda.empty_cache()
    pair_hash = pair_axis_sha256(components.pair_ids)
    receipt = {
        "schema": "paired_neural_discarded_benchmark_v2",
        "status": "full_fold_engineering_fit_completed_and_discarded",
        "contract_sha256": contract_hash,
        "component_sha256": component_hash,
        "pair_axis_sha256": pair_hash,
        "source_hashes": sources,
        "source_bundle_sha256": bundle_hash,
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_hash,
        "economic_artifact_sha256": economic_artifact_hash,
        "economic_manifest_sha256": economic_manifest_hash,
        "fold": str(spec["fold"]),
        "representation": str(spec["representation"]),
        "model": str(spec["model"]),
        "seed": int(spec["seed"]),
        "train_rows": int(len(local_fold.train_indices)),
        "validation_rows": int(len(local_fold.validation_indices)),
        "test_rows_materialized_but_never_scored": int(len(local_fold.test_indices)),
        "elapsed_seconds": float(elapsed),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_process_rss_bytes": peak_rss,
        "fit_discarded": True,
        "contains_predictions_targets_ic_or_performance": False,
        "confirmation_parsed": False,
    }
    atomic_json(args.output, receipt)
    validate_benchmark_receipt(
        args.output,
        contract_hash=contract_hash,
        component_hash=component_hash,
        pair_hash=pair_hash,
        sources=sources,
        bundle_hash=bundle_hash,
        environment=environment,
        environment_hash=environment_hash,
        economic_artifact_hash=economic_artifact_hash,
        economic_manifest_hash=economic_manifest_hash,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def train_full(args: argparse.Namespace) -> int:
    contract = load_contract()
    contract_hash = sha256_file(CONTRACT_PATH)
    component_hash = sha256_file(args.components)
    components, metadata = load_components(args.components)
    provenance = metadata.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("paired_contract_sha256") != contract_hash:
        raise ValueError("component cache is not bound to the current paired contract")
    sources = current_source_hashes()
    bundle_hash = source_bundle_sha256(sources)
    if provenance.get("source_hashes") != sources or provenance.get(
        "source_bundle_sha256"
    ) != bundle_hash:
        raise ValueError("component cache is not bound to the current source bundle")
    economic_artifact_hash = provenance.get("economic_artifact_sha256")
    economic_manifest_hash = provenance.get("economic_manifest_sha256")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in (economic_artifact_hash, economic_manifest_hash)
    ):
        raise ValueError("component cache lacks exact economic artifact bindings")
    if components.clock[-1] >= DEVELOPMENT_END_EXCLUSIVE:
        raise ValueError("component clock reaches confirmation")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair_hash = pair_axis_sha256(components.pair_ids)
    if metadata.get("pair_axis_sha256") != pair_hash:
        raise ValueError("component pair-axis hash differs from the training axis")
    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    device = str(args.device)
    configure_determinism()
    environment = runtime_environment(device=device)
    environment_hash = runtime_environment_sha256(environment)
    validate_benchmark_receipt(
        args.benchmark_receipt,
        contract_hash=contract_hash,
        component_hash=component_hash,
        pair_hash=pair_hash,
        sources=sources,
        bundle_hash=bundle_hash,
        environment=environment,
        environment_hash=environment_hash,
        economic_artifact_hash=economic_artifact_hash,
        economic_manifest_hash=economic_manifest_hash,
    )
    benchmark_receipt_hash = sha256_file(args.benchmark_receipt)
    if (output / "run_manifest.json").exists():
        completed = validate_completed_forecast_run(
            output,
            components=components,
            contract=contract,
            contract_hash=contract_hash,
            component_hash=component_hash,
            pair_hash=pair_hash,
            sources=sources,
            bundle_hash=bundle_hash,
            environment=environment,
            environment_hash=environment_hash,
            economic_artifact_hash=economic_artifact_hash,
            economic_manifest_hash=economic_manifest_hash,
            benchmark_receipt_hash=benchmark_receipt_hash,
        )
        print(json.dumps(completed, sort_keys=True))
        return 0
    fold_specs = contract["folds"]
    for fold_spec in fold_specs:
        fold = make_walk_forward_fold(
            components,
            name=fold_spec["name"],
            test_start=fold_spec["test_start"],
            test_end_exclusive=fold_spec["test_end_exclusive"],
            training_days=int(contract["walk_forward"]["training_days"]),
            validation_days=int(contract["walk_forward"]["validation_days"]),
            purge_hours=int(contract["walk_forward"]["target_purge_hours"]),
        )
        expected_test_clock = components.clock.take(fold.test_indices)
        if expected_test_clock[0] != pd.Timestamp(fold_spec["test_start"]):
            raise ValueError(f"fold start differs: {fold.name}")
        if expected_test_clock[-1] + pd.Timedelta(hours=1) != pd.Timestamp(
            fold_spec["test_end_exclusive"]
        ):
            raise ValueError(f"fold end differs: {fold.name}")

        # Shared endpoint benchmark is fitted once; control invariance is tested
        # by construction because it receives no within-hour coordinates.
        panel, local_fold, local = materialize_fold(
            components, fold, representation="native_1m"
        )
        test = local_fold.test_indices
        endpoint_path = _forecast_path(
            output,
            fold=fold.name,
            representation="shared_endpoint",
            model="endpoint_ridge",
            seed="none",
        )
        endpoint_identity = forecast_identity(
            contract_sha256=contract_hash,
            component_sha256=component_hash,
            pair_axis_sha256_value=pair_hash,
            source_hashes=sources,
            source_bundle_sha256_value=bundle_hash,
            runtime_environment_sha256_value=environment_hash,
            economic_artifact_sha256=economic_artifact_hash,
            economic_manifest_sha256=economic_manifest_hash,
            fold=fold.name,
            representation="shared_endpoint",
            model="endpoint_ridge",
            seed="none",
            role="shared_endpoint_only_baseline",
        )
        if not endpoint_path.exists():
            endpoint = fit_endpoint_ridge(
                panel,
                local_fold,
                ridge_penalty=float(contract["models"]["endpoint_ridge"]["ridge_penalty"]),
            )
            endpoint_prediction = endpoint.predict(
                panel.raw_features[test], panel.feature_available[test]
            )
            pair_targets = cycle.rank_pair_targets(
                endpoint_prediction,
                panel.feature_available[test],
                long_pairs=int(contract["portfolio"]["long_pairs"]),
                short_pairs=int(contract["portfolio"]["short_pairs"]),
                absolute_pair_target=float(contract["portfolio"]["absolute_pair_target"]),
            )
            _save_forecast(
                endpoint_path,
                clock=expected_test_clock,
                predictions=endpoint_prediction,
                pair_targets=pair_targets,
                diagnostics=endpoint_identity,
            )
        endpoint_prediction, _, _ = validate_forecast_artifact(
            endpoint_path,
            expected_clock=expected_test_clock,
            expected_feature_mask=panel.feature_available[test],
            expected_identity=endpoint_identity,
            allowed_diagnostic_extras=set(),
            portfolio=contract["portfolio"],
            require_pair_targets=True,
        )
        endpoint_ic = decision_ic(
            endpoint_prediction,
            panel.target_cashflows[test],
            panel.label_mask[test],
        )
        metric_rows.append(
            {
                "fold": fold.name,
                "representation": "shared_endpoint",
                "model": "endpoint_ridge",
                "mean_ic": float(np.nanmean(endpoint_ic)),
                "finite_ic_decisions": int(np.isfinite(endpoint_ic).sum()),
            }
        )
        for timestamp, value in zip(expected_test_clock, endpoint_ic, strict=True):
            ic_rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "fold": fold.name,
                    "representation": "shared_endpoint",
                    "model": "endpoint_ridge",
                    "ic": float(value),
                }
            )
        manifest_rows.append(
            {
                "fold": fold.name,
                "representation": "shared_endpoint",
                "model": "endpoint_ridge",
                "seed": "none",
                "path": os.path.relpath(endpoint_path, output),
                "sha256": sha256_file(endpoint_path),
            }
        )
        del panel

        for representation in REPRESENTATIONS:
            panel, local_fold, local = materialize_fold(
                components, fold, representation=representation
            )
            test = local_fold.test_indices
            targets = panel.target_cashflows[test]
            mask = panel.label_mask[test]
            for model_name in NEURAL_MODELS:
                predictions_by_seed: dict[int, np.ndarray] = {}
                for seed in FIXED_SEEDS:
                    path = _forecast_path(
                        output,
                        fold=fold.name,
                        representation=representation,
                        model=model_name,
                        seed=seed,
                    )
                    seed_identity = forecast_identity(
                        contract_sha256=contract_hash,
                        component_sha256=component_hash,
                        pair_axis_sha256_value=pair_hash,
                        source_hashes=sources,
                        source_bundle_sha256_value=bundle_hash,
                        runtime_environment_sha256_value=environment_hash,
                        economic_artifact_sha256=economic_artifact_hash,
                        economic_manifest_sha256=economic_manifest_hash,
                        fold=fold.name,
                        representation=representation,
                        model=model_name,
                        seed=seed,
                        role="individual_fixed_seed_forecast",
                    )
                    if path.exists():
                        prediction, stored_diagnostics, _ = validate_forecast_artifact(
                            path,
                            expected_clock=expected_test_clock,
                            expected_feature_mask=panel.feature_available[test],
                            expected_identity=seed_identity,
                            allowed_diagnostic_extras={
                                "epochs",
                                "training_loss",
                                "validation_loss",
                            },
                            portfolio=contract["portfolio"],
                            require_pair_targets=False,
                        )
                        training_loss = float(stored_diagnostics["training_loss"])
                        validation_loss = float(stored_diagnostics["validation_loss"])
                        epochs = int(stored_diagnostics["epochs"])
                    elif model_name == "lstm":
                        fitted = fit_lstm(
                            panel,
                            local_fold,
                            config=LSTMConfig(seed=seed, device=device),
                        )
                        prediction = fitted.predict(
                            panel.raw_features[test], panel.feature_available[test]
                        )
                        training_loss = float(fitted.training_losses[-1])
                        validation_loss = float(fitted.validation_losses[-1])
                        epochs = int(fitted.epochs_trained)
                        _save_forecast(
                            path,
                            clock=expected_test_clock,
                            predictions=prediction,
                            diagnostics={
                                **seed_identity,
                                "epochs": epochs,
                                "training_loss": training_loss,
                                "validation_loss": validation_loss,
                            },
                        )
                        del fitted
                    else:
                        fitted = fit_itransformer(
                            panel,
                            local_fold,
                            config=transformer_config(seed=seed, device=device),
                        )
                        prediction = fitted.predict(
                            panel.raw_features[test],
                            panel.zero_auxiliary[test],
                            panel.feature_available[test],
                        )
                        training_loss = float(fitted.training_losses[-1])
                        validation_loss = float(fitted.validation_losses[-1])
                        epochs = int(fitted.epochs_trained)
                        _save_forecast(
                            path,
                            clock=expected_test_clock,
                            predictions=prediction,
                            diagnostics={
                                **seed_identity,
                                "epochs": epochs,
                                "training_loss": training_loss,
                                "validation_loss": validation_loss,
                            },
                        )
                        del fitted
                    prediction, stored_diagnostics, _ = validate_forecast_artifact(
                        path,
                        expected_clock=expected_test_clock,
                        expected_feature_mask=panel.feature_available[test],
                        expected_identity=seed_identity,
                        allowed_diagnostic_extras={
                            "epochs",
                            "training_loss",
                            "validation_loss",
                        },
                        portfolio=contract["portfolio"],
                        require_pair_targets=False,
                    )
                    epochs = int(stored_diagnostics["epochs"])
                    training_loss = float(stored_diagnostics["training_loss"])
                    validation_loss = float(stored_diagnostics["validation_loss"])
                    if epochs < 1 or not np.isfinite((training_loss, validation_loss)).all():
                        raise ValueError(f"stored training diagnostics are invalid: {path}")
                    predictions_by_seed[seed] = prediction
                    ic = decision_ic(prediction, targets, mask)
                    seed_rows.append(
                        {
                            "fold": fold.name,
                            "representation": representation,
                            "model": model_name,
                            "seed": seed,
                            "mean_ic": float(np.nanmean(ic)),
                            "finite_ic_decisions": int(np.isfinite(ic).sum()),
                            "epochs": epochs,
                            "training_loss": training_loss,
                            "validation_loss": validation_loss,
                        }
                    )
                    manifest_rows.append(
                        {
                            "fold": fold.name,
                            "representation": representation,
                            "model": model_name,
                            "seed": seed,
                            "path": os.path.relpath(path, output),
                            "sha256": sha256_file(path),
                        }
                    )
                ensemble = mean_seed_predictions(predictions_by_seed)
                pair_targets = cycle.rank_pair_targets(
                    ensemble,
                    panel.feature_available[test],
                    long_pairs=int(contract["portfolio"]["long_pairs"]),
                    short_pairs=int(contract["portfolio"]["short_pairs"]),
                    absolute_pair_target=float(contract["portfolio"]["absolute_pair_target"]),
                )
                ensemble_path = _forecast_path(
                    output,
                    fold=fold.name,
                    representation=representation,
                    model=f"{model_name}_seed_mean",
                    seed="all3",
                )
                ensemble_identity = forecast_identity(
                    contract_sha256=contract_hash,
                    component_sha256=component_hash,
                    pair_axis_sha256_value=pair_hash,
                    source_hashes=sources,
                    source_bundle_sha256_value=bundle_hash,
                    runtime_environment_sha256_value=environment_hash,
                    economic_artifact_sha256=economic_artifact_hash,
                    economic_manifest_sha256=economic_manifest_hash,
                    fold=fold.name,
                    representation=representation,
                    model=f"{model_name}_seed_mean",
                    seed="all3",
                    role="fixed_three_seed_arithmetic_mean",
                )
                _save_forecast(
                    ensemble_path,
                    clock=expected_test_clock,
                    predictions=ensemble,
                    pair_targets=pair_targets,
                    diagnostics={
                        **ensemble_identity,
                        "seeds": list(FIXED_SEEDS),
                        "aggregation": "arithmetic_mean_never_best_seed",
                    },
                )
                ensemble, ensemble_diagnostics, pair_targets = validate_forecast_artifact(
                    ensemble_path,
                    expected_clock=expected_test_clock,
                    expected_feature_mask=panel.feature_available[test],
                    expected_identity=ensemble_identity,
                    allowed_diagnostic_extras={"seeds", "aggregation"},
                    portfolio=contract["portfolio"],
                    require_pair_targets=True,
                    expected_predictions=ensemble,
                )
                if ensemble_diagnostics["seeds"] != list(FIXED_SEEDS) or ensemble_diagnostics[
                    "aggregation"
                ] != "arithmetic_mean_never_best_seed":
                    raise ValueError(f"ensemble seed aggregation differs: {ensemble_path}")
                ensemble_ic = decision_ic(ensemble, targets, mask)
                metric_rows.append(
                    {
                        "fold": fold.name,
                        "representation": representation,
                        "model": model_name,
                        "mean_ic": float(np.nanmean(ensemble_ic)),
                        "finite_ic_decisions": int(np.isfinite(ensemble_ic).sum()),
                    }
                )
                for timestamp, value in zip(expected_test_clock, ensemble_ic, strict=True):
                    ic_rows.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "fold": fold.name,
                            "representation": representation,
                            "model": model_name,
                            "ic": float(value),
                        }
                    )
                manifest_rows.append(
                    {
                        "fold": fold.name,
                        "representation": representation,
                        "model": f"{model_name}_seed_mean",
                        "seed": "all3",
                        "path": os.path.relpath(ensemble_path, output),
                        "sha256": sha256_file(ensemble_path),
                    }
                )
            del panel

    if len(seed_rows) != 6 * 2 * 2 * 3:
        raise AssertionError("seed metric count differs from the frozen family")
    if len(metric_rows) != 6 * 5:
        raise AssertionError("fold metric count differs from the frozen five-series family")
    if len(manifest_rows) != 6 * 17:
        raise AssertionError("forecast manifest count differs from the frozen artifact family")
    summary_frames = {
        "seed_metrics.csv": pd.DataFrame(seed_rows),
        "fold_metrics.csv": pd.DataFrame(metric_rows),
        "ic_by_decision.csv": pd.DataFrame(ic_rows),
        "forecast_manifest.csv": pd.DataFrame(manifest_rows),
    }
    for name, frame in summary_frames.items():
        atomic_csv(output / name, frame)
    attempts = pd.DataFrame(
        [
            {"candidate_id": "B00", "series": "endpoint_ridge_shared", "status": "forecast_complete"},
            {"candidate_id": "N01", "series": "native_1m_lstm_seed_mean", "status": "forecast_complete"},
            {"candidate_id": "C01", "series": "causal_15m_lstm_seed_mean", "status": "forecast_complete"},
            {"candidate_id": "N02", "series": "native_1m_itransformer_seed_mean", "status": "forecast_complete"},
            {"candidate_id": "C02", "series": "causal_15m_itransformer_seed_mean", "status": "forecast_complete"},
        ]
    )
    atomic_csv(output / "attempt_ledger.csv", attempts)
    summary_hashes = {
        name: sha256_file(output / name)
        for name in (*summary_frames, "attempt_ledger.csv")
    }
    run_manifest = {
        "schema": "paired_neural_forecast_run_manifest_v2",
        "status": "forecast_stage_v2_complete_portfolio_ledger_pending",
        "contract_sha256": contract_hash,
        "component_cache_sha256": component_hash,
        "pair_axis_sha256": pair_hash,
        "source_hashes": sources,
        "source_bundle_sha256": bundle_hash,
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_hash,
        "economic_artifact_sha256": economic_artifact_hash,
        "economic_manifest_sha256": economic_manifest_hash,
        "discarded_benchmark_receipt_sha256": benchmark_receipt_hash,
        "fold_count": len(fold_specs),
        "unique_strategy_series": 5,
        "forecast_artifact_count": len(manifest_rows),
        "fold_metric_rows": len(metric_rows),
        "seed_metric_rows": len(seed_rows),
        "fixed_seeds": list(FIXED_SEEDS),
        "seed_selection": False,
        "portfolio_backend_required": "the existing audited multicoin pair ledger",
        "portfolio_settings": contract["portfolio"],
        "summary_sha256": summary_hashes,
        "confirmation_parsed": False,
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, sort_keys=True))
    return 0


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-contract")
    validate.set_defaults(function=lambda _args: (load_contract(), print("contract_valid"), 0)[2])

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--acquisition-manifest", type=Path, required=True)
    prepare.add_argument("--economic-inputs", type=Path, required=True)
    prepare.add_argument("--economic-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--start", default="2021-01-01T00:00:00Z")
    prepare.add_argument("--end-exclusive", default="2025-07-01T00:00:00Z")
    prepare.set_defaults(function=prepare_components)

    smoke = subparsers.add_parser("smoke-existing")
    smoke.add_argument("--cache-root", type=Path, required=True)
    smoke.add_argument("--acquisition-manifest", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--start", default="2023-09-01T00:00:00Z")
    smoke.add_argument("--end-exclusive", default="2023-10-01T00:00:00Z")
    smoke.set_defaults(function=smoke_existing)

    benchmark = subparsers.add_parser("benchmark-discarded-fold")
    benchmark.add_argument("--components", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    benchmark.set_defaults(function=benchmark_discarded_fold)

    train = subparsers.add_parser("train")
    train.add_argument("--components", type=Path, required=True)
    train.add_argument("--benchmark-receipt", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.set_defaults(function=train_full)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
