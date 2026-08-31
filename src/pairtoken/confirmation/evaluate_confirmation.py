#!/usr/bin/env python3
"""Authorize and execute the one-shot source-audited confirmation evaluation.

The frozen development evaluator cannot be called because it rejects all
post-June-2025 clocks and assumes a different five-series family.  This module
therefore provides both keys: ``authorize`` binds the still-unparsed economic
payload to a complete all-fold forecast seal; ``run`` consumes that
authorization exactly once, parses the payload, and invokes the hash-pinned
audited risk/projection/fixed-quantity ledger primitives.  Ordering is security
relevant: every forecast, seal, manifest, source and authorization check occurs
before the first outcome array is parsed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from confirmation_governance import (
    AUTHORIZATION_SCHEMA,
    CONTRACT_PATH,
    ConfirmationGateError,
    atomic_json,
    axis_sha256,
    create_source_freeze,
    expected_artifact_path,
    fold_contract,
    load_contract,
    load_json,
    pair_ids,
    ForecastSpec,
    rank_pair_targets,
    read_forecast_artifact,
    register_forecast,
    seal_all_forecasts,
    sha256_file,
    verify_all_forecast_seal,
    verify_source_freeze,
    write_forecast_artifact,
)
from prepare_confirmation import (
    DEFAULT_AUDITED_ROOT,
    _array_family_sha256,
    _write_npz,
    audited_dependencies,
)
from train_confirmation import (
    INPUT_MEMBERS,
    INPUT_SCHEMA,
    _bind_runtime_manifest,
    _diagnostics,
    _expected_partition_clocks,
    load_blinded_fold_input,
)


EVALUATION_MANIFEST_SCHEMA = "pairtoken_1m_confirmation_evaluation_payload_manifest_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAYLOAD_MEMBERS = {
    "metadata_json", "decision_clock_ns", "ledger_clock_ns", "symbols", "pair_ids",
    "pair_leg_indices", "pair_weights", "coin_open_prices",
    "coin_open_to_open_returns", "funding_at_boundary", "coin_availability",
    "pair_decision_available", "pair_target_cashflows", "pair_target_available",
    "forecast_feature_available",
    "risk_beta", "risk_covariance", "terminal_row_mask",
}


def authorize_evaluation(
    *,
    forecast_root: Path,
    source_freeze_path: Path,
    forecast_seal_path: Path,
    evaluation_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    # Do not move any evaluation-manifest or payload operation above this line.
    seal = verify_all_forecast_seal(
        forecast_root,
        source_freeze_path=source_freeze_path,
        seal_path=forecast_seal_path,
    )
    contract = load_contract()
    freeze = verify_source_freeze(source_freeze_path)

    manifest = load_json(evaluation_manifest_path, label="evaluation-payload manifest")
    required = {
        "schema", "status", "contract_sha256", "source_freeze_sha256",
        "payload_path", "payload_sha256", "payload_role", "clock_start",
        "clock_end_exclusive", "terminal_rule", "symbols", "pairs",
        "funding_included", "cost_grid_bps", "risk_engine", "audited_source_hashes",
        "blinded_input_artifacts",
        "forecast_payload_absent", "portfolio_metrics_computed",
    }
    if set(manifest) != required or manifest.get("schema") != EVALUATION_MANIFEST_SCHEMA:
        raise ConfirmationGateError("evaluation-payload manifest schema differs")
    expected = {
        "status": "sealed_economic_and_execution_inputs_never_opened_by_forecast_stage",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "payload_role": "confirmation_outcomes_and_execution_only",
        "clock_start": "2025-08-01T00:00:00Z",
        "clock_end_exclusive": "2026-06-30T23:00:00Z",
        "terminal_rule": "zero_pair_target_liquidation_at_observed_2026_06_30_23_open",
        "symbols": contract["data"]["symbols_in_order"],
        "pairs": 276,
        "funding_included": True,
        "cost_grid_bps": [0.0, 0.5, 1.5, 5.0],
        "risk_engine": {
            "covariance_lookback_days": 60,
            "minimum_complete_hourly_observations": 720,
            "maximum_total_gross": 1.0,
            "maximum_absolute_coin_weight": 0.15,
            "maximum_absolute_dollar_net": 0.02,
            "maximum_absolute_btc_beta": 0.05,
            "target_annualized_volatility": 0.1,
        },
        "forecast_payload_absent": True,
        "portfolio_metrics_computed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ConfirmationGateError(f"evaluation manifest differs for {key}")
    sources = manifest.get("audited_source_hashes")
    if not isinstance(sources, dict) or not sources or any(
        SHA256_RE.fullmatch(str(value)) is None for value in sources.values()
    ):
        raise ConfirmationGateError("evaluation adapter source binding is absent")
    expected_blinded = [
        (row["name"], representation, f"{row['name']}/{representation}.blinded.npz")
        for row in contract["refits"]
        for representation in ("native_1m", "causal_15m_control")
    ]
    blinded = manifest.get("blinded_input_artifacts")
    if not isinstance(blinded, list) or len(blinded) != len(expected_blinded):
        raise ConfirmationGateError("evaluation manifest blinded-input family differs")
    for binding, (fold, representation, relative_path) in zip(
        blinded, expected_blinded, strict=True
    ):
        if not isinstance(binding, dict) or set(binding) != {
            "fold", "representation", "relative_path", "sha256"
        }:
            raise ConfirmationGateError("evaluation manifest blinded-input binding differs")
        if (
            binding["fold"] != fold
            or binding["representation"] != representation
            or binding["relative_path"] != relative_path
            or SHA256_RE.fullmatch(str(binding["sha256"])) is None
        ):
            raise ConfirmationGateError("evaluation manifest blinded-input identity differs")
    payload_path = (Path(evaluation_manifest_path).parent / str(manifest["payload_path"])).resolve()
    try:
        payload_path.relative_to(Path(evaluation_manifest_path).parent.resolve())
    except ValueError as exc:
        raise ConfirmationGateError("evaluation payload escapes its manifest directory") from exc
    if SHA256_RE.fullmatch(str(manifest.get("payload_sha256"))) is None:
        raise ConfirmationGateError("evaluation payload hash is invalid")
    # The payload is first touched only after the complete forecast seal and
    # manifest contract have passed.  sha256_file streams bytes; it never parses
    # outcomes. Parsing belongs only to the single-use ``run`` backend below.
    if not payload_path.is_file() or sha256_file(payload_path) != manifest["payload_sha256"]:
        raise ConfirmationGateError("evaluation payload differs from its sealed manifest")
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "authorized_for_exactly_one_frozen_portfolio_evaluation",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "forecast_seal_sha256": sha256_file(forecast_seal_path),
        "forecast_manifest_sha256": seal["forecast_manifest_sha256"],
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "evaluation_payload_sha256": manifest["payload_sha256"],
        "evaluation_adapter_source_hashes": sources,
        "required_models": ["N02", "C02"],
        "required_cost_grid_bps": [0.0, 0.5, 1.5, 5.0],
        "single_use_backend_receipt_required": True,
        "evaluation_payload_parsed_by_this_gate": False,
        "portfolio_metrics_computed_by_this_gate": False,
    }
    atomic_json(output, authorization)
    return authorization


def _verify_blinded_input_bindings(
    *,
    blinded_input_root: Path,
    evaluation_manifest_path: Path,
    forecast_seal_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Bind sealed forecasts, causal feature masks, and evaluation components."""

    evaluation_manifest = load_json(
        evaluation_manifest_path, label="evaluation-payload manifest"
    )
    bindings = evaluation_manifest.get("blinded_input_artifacts")
    seal = load_json(forecast_seal_path, label="all-fold forecast seal")
    forecast_manifest_path = Path(forecast_seal_path).parent / str(
        seal.get("forecast_manifest_path", "")
    )
    forecast_manifest = load_json(forecast_manifest_path, label="sealed forecast manifest")
    registered: dict[tuple[str, str], set[str]] = {}
    for entry in forecast_manifest.get("entries", []):
        if not isinstance(entry, dict):
            raise ConfirmationGateError("sealed forecast entry differs")
        key = (str(entry.get("fold")), str(entry.get("representation")))
        registered.setdefault(key, set()).add(str(entry.get("input_artifact_sha256")))

    expected_pairs = pair_ids(contract["data"]["symbols_in_order"])
    expected_legs = np.asarray([
        (second, first)
        for first in range(24) for second in range(first + 1, 24)
    ], dtype=np.int64)
    weight_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    root = Path(blinded_input_root).resolve()
    expected_bindings = [
        (row, representation)
        for row in contract["refits"]
        for representation in ("native_1m", "causal_15m_control")
    ]
    if not isinstance(bindings, list) or len(bindings) != 8:
        raise ConfirmationGateError("blinded-input binding family differs")
    paired_fold: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for binding, (fold, representation) in zip(bindings, expected_bindings, strict=True):
        key = (fold["name"], representation)
        relative = f"{key[0]}/{representation}.blinded.npz"
        if (
            not isinstance(binding, dict)
            or binding.get("fold") != key[0]
            or binding.get("representation") != representation
            or binding.get("relative_path") != relative
        ):
            raise ConfirmationGateError("blinded-input binding order or identity differs")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ConfirmationGateError("blinded input escapes its governed root") from exc
        digest = str(binding.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != digest:
            raise ConfirmationGateError("blinded input differs from evaluation binding")
        if registered.get(key) != {digest}:
            raise ConfirmationGateError("sealed forecasts name another blinded input")
        try:
            with np.load(path, allow_pickle=False) as stored:
                if set(stored.files) != INPUT_MEMBERS:
                    raise ConfirmationGateError("bound blinded-input member family differs")
                metadata = json.loads(str(stored["metadata_json"].item()))
                source_clock = np.asarray(stored["clock_source_ns"], dtype=np.int64)
                test_rows = np.asarray(stored["test_rows"], dtype=np.int64)
                stored_pairs = tuple(stored["pair_ids"].astype(str))
                legs = np.asarray(stored["pair_leg_indices"], dtype=np.int64)
                weights = np.asarray(stored["pair_weights"], dtype=np.float64)[test_rows]
                features = np.asarray(stored["feature_available"], dtype=bool)[test_rows]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ConfirmationGateError("bound blinded input is invalid") from exc
        if (
            metadata.get("schema") != INPUT_SCHEMA
            or metadata.get("fold") != key[0]
            or metadata.get("representation") != representation
            or stored_pairs != expected_pairs
            or not np.array_equal(legs, expected_legs)
            or not np.array_equal(
                source_clock[test_rows],
                np.arange(
                    pd.Timestamp(fold["test_start"]).value,
                    pd.Timestamp(fold["test_end_exclusive"]).value,
                    3_600_000_000_000,
                    dtype=np.int64,
                ),
            )
        ):
            raise ConfirmationGateError("bound blinded-input axes differ")
        content_hash = str(metadata.get("factorized_content_sha256", ""))
        if representation == "native_1m":
            paired_fold[key[0]] = (weights.copy(), features.copy(), content_hash)
            weight_parts.append(weights.copy())
            feature_parts.append(features.copy())
        else:
            native = paired_fold.get(key[0])
            if native is None or content_hash != native[2]:
                raise ConfirmationGateError("paired blinded-input content hashes differ")
            if not np.array_equal(weights, native[0], equal_nan=True) or not np.array_equal(
                features, native[1]
            ):
                raise ConfirmationGateError("paired blinded-input execution tensors differ")
    return {
        "pair_leg_indices": expected_legs,
        "pair_weights": np.concatenate(weight_parts, axis=0),
        "feature_available": np.concatenate(feature_parts, axis=0),
    }


def _verify_authorization(
    path: Path,
    *,
    forecast_seal_path: Path,
    source_freeze_path: Path,
    evaluation_manifest_path: Path,
) -> dict[str, Any]:
    authorization = load_json(path, label="evaluation authorization")
    required = {
        "schema", "status", "contract_sha256", "source_freeze_sha256",
        "source_bundle_sha256", "forecast_seal_sha256", "forecast_manifest_sha256",
        "evaluation_manifest_sha256", "evaluation_payload_sha256",
        "evaluation_adapter_source_hashes", "required_models", "required_cost_grid_bps",
        "single_use_backend_receipt_required", "evaluation_payload_parsed_by_this_gate",
        "portfolio_metrics_computed_by_this_gate",
    }
    if set(authorization) != required or authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise ConfirmationGateError("evaluation authorization schema differs")
    freeze = verify_source_freeze(source_freeze_path)
    expected = {
        "status": "authorized_for_exactly_one_frozen_portfolio_evaluation",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "forecast_seal_sha256": sha256_file(forecast_seal_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "required_models": ["N02", "C02"],
        "required_cost_grid_bps": [0.0, 0.5, 1.5, 5.0],
        "single_use_backend_receipt_required": True,
        "evaluation_payload_parsed_by_this_gate": False,
        "portfolio_metrics_computed_by_this_gate": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ConfirmationGateError(f"evaluation authorization differs for {key}")
    return authorization


def _load_payload_after_authorization(
    manifest_path: Path, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = load_json(manifest_path, label="evaluation payload manifest")
    payload_path = (Path(manifest_path).parent / str(manifest["payload_path"])).resolve()
    if manifest.get("payload_sha256") != authorization.get("evaluation_payload_sha256"):
        raise ConfirmationGateError("authorization and payload hash differ")
    if not payload_path.is_file() or sha256_file(payload_path) != manifest["payload_sha256"]:
        raise ConfirmationGateError("authorized evaluation payload differs")
    try:
        with np.load(payload_path, allow_pickle=False) as stored:
            if set(stored.files) != PAYLOAD_MEMBERS:
                raise ConfirmationGateError("evaluation payload member family differs")
            result = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, KeyError) as exc:
        raise ConfirmationGateError("invalid authorized evaluation payload") from exc
    metadata = json.loads(str(result.pop("metadata_json").item()))
    required_metadata = {
        "schema", "contract_sha256", "source_freeze_sha256",
        "forecast_payload_absent", "portfolio_metrics_computed", "decision_rows",
        "ledger_rows", "terminal_rule", "acquisition_manifest_sha256",
        "archive_availability_sha256", "artifact_hashes_sha256",
    }
    expected_metadata = {
        "schema": "pairtoken_1m_confirmation_evaluation_payload_v1",
        "contract_sha256": authorization["contract_sha256"],
        "source_freeze_sha256": authorization["source_freeze_sha256"],
        "forecast_payload_absent": True,
        "portfolio_metrics_computed": False,
        "decision_rows": 8015,
        "ledger_rows": 8016,
        "terminal_rule": "zero_pair_target_liquidation_at_observed_2026_06_30_23_open",
    }
    if not isinstance(metadata, dict) or set(metadata) != required_metadata:
        raise ConfirmationGateError("evaluation payload metadata family differs")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ConfirmationGateError(f"evaluation payload metadata differs for {key}")
    for key in (
        "acquisition_manifest_sha256", "archive_availability_sha256",
        "artifact_hashes_sha256",
    ):
        if SHA256_RE.fullmatch(str(metadata.get(key))) is None:
            raise ConfirmationGateError(f"evaluation payload metadata hash differs for {key}")
    result["metadata"] = metadata
    return result


def _decision_ic(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    response = np.asarray(targets, dtype=float)
    available = np.asarray(mask, dtype=bool)
    result = np.full(len(values), np.nan)
    for row in range(len(values)):
        chosen = available[row] & np.isfinite(values[row]) & np.isfinite(response[row])
        if int(chosen.sum()) < 3:
            continue
        x = pd.Series(values[row, chosen]).rank(method="average").to_numpy()
        y = pd.Series(response[row, chosen]).rank(method="average").to_numpy()
        if np.std(x) > 0 and np.std(y) > 0:
            result[row] = float(np.corrcoef(x, y)[0, 1])
    return result


def _metrics(bars: pd.DataFrame) -> dict[str, float]:
    dates = bars.index.normalize()
    daily = (1.0 + bars["net_return"]).groupby(dates).prod() - 1.0
    mean = float(daily.mean())
    volatility = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    compound = float((1.0 + daily).prod())
    years = len(daily) / 365.0
    nav = (1.0 + daily).cumprod()
    peak = np.maximum.accumulate(
        np.concatenate(([1.0], nav.to_numpy(dtype=float)))
    )[1:]
    drawdown = nav.to_numpy(dtype=float) / peak - 1.0
    gross_daily = (1.0 + bars["gross_return"]).groupby(dates).prod() - 1.0
    return {
        "daily_observations": float(len(daily)),
        "annual_return": compound ** (1.0 / years) - 1.0 if years > 0 and compound > 0 else float("nan"),
        "annual_volatility": volatility * math.sqrt(365.0),
        "sharpe": math.sqrt(365.0) * mean / volatility if volatility > 0 else float("nan"),
        "maximum_drawdown": float(np.min(drawdown)),
        "total_return": compound - 1.0,
        "gross_total_return": float((1.0 + gross_daily).prod() - 1.0),
        "total_turnover": float(bars["turnover"].sum()),
        "turnover_per_day": float(bars["turnover"].sum() / len(daily)),
        "total_transaction_cost": float(bars["transaction_cost"].sum()),
        "total_funding_return": float(bars["funding_return"].sum()),
        "total_price_return": float(bars["price_return"].sum()),
        "mean_gross_exposure": float(bars["gross_exposure"].mean()),
    }


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def run_evaluation(
    *,
    forecast_root: Path,
    source_freeze_path: Path,
    forecast_seal_path: Path,
    evaluation_manifest_path: Path,
    authorization_path: Path,
    blinded_input_root: Path,
    audited_root: Path,
    output: Path,
) -> dict[str, Any]:
    # First key: every forecast and its all-fold seal.
    seal = verify_all_forecast_seal(
        forecast_root, source_freeze_path=source_freeze_path, seal_path=forecast_seal_path
    )
    # Second key: the pre-issued content binding to the still-unparsed payload.
    authorization = _verify_authorization(
        authorization_path,
        forecast_seal_path=forecast_seal_path,
        source_freeze_path=source_freeze_path,
        evaluation_manifest_path=evaluation_manifest_path,
    )
    if authorization["forecast_manifest_sha256"] != seal["forecast_manifest_sha256"]:
        raise ConfirmationGateError("authorization names another forecast manifest")
    contract = load_contract()
    blinded = _verify_blinded_input_bindings(
        blinded_input_root=blinded_input_root,
        evaluation_manifest_path=evaluation_manifest_path,
        forecast_seal_path=forecast_seal_path,
        contract=contract,
    )
    dependencies = audited_dependencies(audited_root)
    if dict(dependencies.hashes) != authorization["evaluation_adapter_source_hashes"]:
        raise ConfirmationGateError("authorized audited-source family differs")
    consumed_path = Path(authorization_path).with_suffix(".consumed.json")
    if consumed_path.exists():
        raise ConfirmationGateError("evaluation authorization was already consumed")
    if Path(output).exists():
        raise FileExistsError(output)
    # Mark the authorization spent before parsing outcomes. A failed run is
    # intentionally fail-closed and requires an explicit governance decision.
    atomic_json(consumed_path, {
        "schema": "pairtoken_1m_confirmation_authorization_consumed_v1",
        "authorization_sha256": sha256_file(authorization_path),
        "forecast_seal_sha256": sha256_file(forecast_seal_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "status": "consumed_before_first_payload_parse",
    })
    payload = _load_payload_after_authorization(evaluation_manifest_path, authorization)
    decision_clock = pd.to_datetime(payload["decision_clock_ns"], utc=True)
    ledger_clock = pd.to_datetime(payload["ledger_clock_ns"], utc=True)
    expected_decision = pd.date_range(
        "2025-08-01T00:00:00Z", "2026-06-30T22:00:00Z", freq="1h"
    )
    expected_ledger = expected_decision.append(
        pd.DatetimeIndex([pd.Timestamp("2026-06-30T23:00:00Z")])
    )
    if not decision_clock.equals(expected_decision) or not ledger_clock.equals(expected_ledger):
        raise ConfirmationGateError("evaluation payload clocks differ")
    symbols = tuple(payload["symbols"].astype(str))
    pairs = tuple(payload["pair_ids"].astype(str))
    expected_pairs = pair_ids(contract["data"]["symbols_in_order"])
    if symbols != tuple(contract["data"]["symbols_in_order"]) or pairs != expected_pairs:
        raise ConfirmationGateError("evaluation payload axes differ")
    legs = np.asarray(payload["pair_leg_indices"], dtype=np.int64)
    pair_weights = np.asarray(payload["pair_weights"], dtype=float)
    open_prices = np.asarray(payload["coin_open_prices"], dtype=float)
    returns = np.asarray(payload["coin_open_to_open_returns"], dtype=float)
    funding = np.asarray(payload["funding_at_boundary"], dtype=float)
    availability = np.asarray(payload["coin_availability"], dtype=bool)
    beta = np.asarray(payload["risk_beta"], dtype=float)
    covariance = np.asarray(payload["risk_covariance"], dtype=float)
    pair_available = np.asarray(payload["pair_decision_available"], dtype=bool)
    actual = np.asarray(payload["pair_target_cashflows"], dtype=float)
    actual_available = np.asarray(payload["pair_target_available"], dtype=bool)
    forecast_feature = np.asarray(payload["forecast_feature_available"], dtype=bool)
    terminal_mask = np.asarray(payload["terminal_row_mask"], dtype=bool)
    expected_shapes = {
        "legs": (276, 2), "pair_weights": (8016, 276, 2),
        "open_prices": (8016, 24),
        "returns": (8016, 24), "funding": (8016, 24),
        "availability": (8016, 24), "beta": (8016, 24),
        "covariance": (8016, 24, 24), "pair_available": (8015, 276),
        "actual": (8015, 276), "actual_available": (8015, 276),
        "forecast_feature": (8015, 276), "terminal_mask": (8016,),
    }
    for name, shape in expected_shapes.items():
        if locals()[name].shape != shape:
            raise ConfirmationGateError(f"evaluation payload axes differ: {name}")
    expected_legs = np.asarray([
        (second, first) for first in range(24) for second in range(first + 1, 24)
    ], dtype=np.int64)
    if not np.array_equal(legs, expected_legs) or not np.array_equal(
        legs, blinded["pair_leg_indices"]
    ):
        raise ConfirmationGateError("evaluation pair-leg axis differs")
    if not np.array_equal(terminal_mask, np.arange(8016) == 8015):
        raise ConfirmationGateError("terminal-row mask differs")
    if not availability[-1, 0] or np.any(returns[-1] != 0.0):
        raise ConfirmationGateError("terminal row is not an observed zero-return liquidation boundary")
    if np.isinf(open_prices).any() or np.any(
        availability & (~np.isfinite(open_prices) | (open_prices <= 0.0))
    ):
        raise ConfirmationGateError("observed execution opens are invalid")
    if not np.array_equal(forecast_feature, blinded["feature_available"]):
        raise ConfirmationGateError("evaluation and blinded feature masks differ")
    if not np.array_equal(pair_weights[:-1], blinded["pair_weights"], equal_nan=True):
        raise ConfirmationGateError("evaluation and blinded hedge weights differ")
    if np.any(forecast_feature & ~pair_available):
        raise ConfirmationGateError("forecast feature mask exceeds executable pair availability")
    selected_weights = pair_weights[:-1][pair_available]
    if not np.isfinite(selected_weights).all() or not np.allclose(
        np.abs(selected_weights).sum(axis=1), 1.0, atol=1e-12, rtol=1e-12
    ):
        raise ConfirmationGateError("active pair weights are nonfinite or not gross-normalized")

    stitched: dict[str, np.ndarray] = {}
    ic_rows: list[dict[str, Any]] = []
    for representation, model_id in (("native_1m", "N02"), ("causal_15m_control", "C02")):
        prediction_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        clock_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        for fold in contract["refits"]:
            spec = ForecastSpec(fold["name"], representation, "all3", "seed_mean")
            artifact = read_forecast_artifact(
                expected_artifact_path(forecast_root, spec), spec, contract
            )
            prediction_parts.append(artifact["predictions"])
            target_parts.append(artifact["pair_targets"])
            clock_parts.append(artifact["clock_ns"])
            feature_parts.append(artifact["feature_available"])
        if not np.array_equal(np.concatenate(clock_parts), decision_clock.asi8):
            raise ConfirmationGateError(f"stitched forecast clock differs: {model_id}")
        prediction = np.concatenate(prediction_parts)
        targets = np.concatenate(target_parts)
        if not np.array_equal(np.concatenate(feature_parts), forecast_feature):
            raise ConfirmationGateError(f"stitched forecast feature mask differs: {model_id}")
        if np.any((targets != 0.0) & ~pair_available):
            raise ConfirmationGateError(f"{model_id} trades an unavailable pair")
        stitched[model_id] = targets
        ic = _decision_ic(prediction, actual, actual_available & pair_available)
        for timestamp, value in zip(decision_clock, ic, strict=True):
            ic_rows.append({
                "model_id": model_id, "timestamp": timestamp.isoformat(),
                "ic": float(value) if np.isfinite(value) else np.nan,
            })

    hedge = np.zeros((8016, 24, 276), dtype=float)
    for pair, (signal, hedge_coin) in enumerate(legs):
        hedge[:, signal, pair] = np.nan_to_num(pair_weights[:, pair, 0])
        hedge[:, hedge_coin, pair] = np.nan_to_num(pair_weights[:, pair, 1])
    returns_frame = pd.DataFrame(returns, index=ledger_clock, columns=symbols)
    funding_frame = pd.DataFrame(funding, index=ledger_clock, columns=symbols)
    availability_frame = pd.DataFrame(availability, index=ledger_clock, columns=symbols)
    constraints = dependencies.execution.PortfolioConstraints(
        maximum_total_gross=1.0,
        maximum_absolute_coin_weight=0.15,
        maximum_absolute_dollar_net=0.02,
        maximum_absolute_btc_beta=0.05,
        target_annualized_volatility=0.1,
    )
    metric_rows: list[dict[str, Any]] = []
    hourly_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    coin_rows: list[pd.DataFrame] = []
    attribution_rows: list[pd.DataFrame] = []
    risk_rows: list[dict[str, Any]] = []
    risk_hourly_rows: list[pd.DataFrame] = []
    for model_id in ("N02", "C02"):
        pair_target_values = np.vstack((stitched[model_id], np.zeros((1, 276))))
        pair_target_frame = pd.DataFrame(pair_target_values, index=ledger_clock, columns=pairs)
        for cost in (0.0, 0.5, 1.5, 5.0):
            ledger = dependencies.execution.backtest_multicoin_pair_ledger(
                pair_target_frame,
                hedge,
                returns_frame,
                funding_frame,
                transaction_cost_bps=cost,
                coin_symbols=symbols,
                availability=availability_frame,
                constraints=constraints,
                btc_beta_loadings=beta,
                covariance_matrices=covariance,
                liquidate_at_end=False,
            )
            terminal_close = ledger.coin_ledger.xs(ledger_clock[-1], level="timestamp")["close_notional"]
            if not np.allclose(terminal_close, 0.0, atol=1e-12, rtol=0.0):
                raise ConfirmationGateError("terminal liquidation left a position")
            bars = ledger.bars.copy()
            bars.insert(0, "timestamp", bars.index)
            bars.insert(0, "cost_bps", cost)
            bars.insert(0, "model_id", model_id)
            hourly_rows.append(bars.reset_index(drop=True))
            daily = ledger.daily_returns.rename("net_return").to_frame().reset_index()
            daily.insert(0, "cost_bps", cost)
            daily.insert(0, "model_id", model_id)
            daily_rows.append(daily)
            attribution = ledger.daily_attribution.reset_index()
            if not np.allclose(
                attribution["net_return"].to_numpy(dtype=float),
                ledger.daily_returns.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-14,
            ):
                raise ConfirmationGateError("audited daily attribution does not reconcile")
            attribution.insert(0, "cost_bps", cost)
            attribution.insert(0, "model_id", model_id)
            attribution_rows.append(attribution)
            coin = ledger.daily_coin_attribution.reset_index()
            coin.insert(0, "cost_bps", cost)
            coin.insert(0, "model_id", model_id)
            coin_rows.append(coin)
            gross_sum = float(ledger.bars["gross_return"].sum())
            turnover = float(ledger.bars["turnover"].sum())
            break_even = 10_000.0 * gross_sum / turnover if turnover > 0 else float("nan")
            metric_rows.append({
                "scope": "aggregate", "model_id": model_id, "refit": "ALL",
                "cost_bps": cost, **_metrics(ledger.bars),
                "linearized_break_even_cost_bps": break_even,
            })
            for fold in contract["refits"]:
                start = pd.Timestamp(fold["test_start"])
                end = pd.Timestamp(fold["test_end_exclusive"])
                chosen = ledger.bars.loc[(ledger.bars.index >= start) & (ledger.bars.index < end)]
                if fold["name"] == contract["refits"][-1]["name"]:
                    chosen = ledger.bars.loc[(ledger.bars.index >= start) & (ledger.bars.index <= end)]
                chosen_turnover = float(chosen["turnover"].sum())
                chosen_break_even = (
                    10_000.0 * float(chosen["gross_return"].sum()) / chosen_turnover
                    if chosen_turnover > 0 else float("nan")
                )
                metric_rows.append({
                    "scope": "refit", "model_id": model_id, "refit": fold["name"],
                    "cost_bps": cost, **_metrics(chosen),
                    "linearized_break_even_cost_bps": chosen_break_even,
                })
            projected = ledger.projection.targets.to_numpy(dtype=float)
            long_gross = np.clip(projected, 0.0, None).sum(axis=1)
            short_gross = -np.clip(projected, None, 0.0).sum(axis=1)
            gross = long_gross + short_gross
            net = projected.sum(axis=1)
            max_coin = np.abs(projected).max(axis=1)
            active = projected != 0.0
            if np.any(active & ~np.isfinite(beta)):
                raise ConfirmationGateError("an active coin lacks a finite BTC-beta loading")
            active_covariance = active[:, :, None] & active[:, None, :]
            if np.any(active_covariance & ~np.isfinite(covariance)):
                raise ConfirmationGateError("active coin coordinates lack finite covariance")
            safe_beta = np.where(active, beta, 0.0)
            safe_covariance = np.where(active_covariance, covariance, 0.0)
            btc_exposure = np.einsum("tc,tc->t", projected, safe_beta)
            variance = np.einsum("ti,tij,tj->t", projected, safe_covariance, projected)
            if not np.isfinite(variance).all() or np.any(variance < -1e-12):
                raise ConfirmationGateError("risk covariance implies an invalid variance")
            volatility = np.sqrt(np.maximum(variance, 0.0) * 365.0 * 24.0)
            violations = (
                (gross > 1.0 + 1e-10) | (max_coin > 0.15 + 1e-10)
                | (np.abs(net) > 0.02 + 1e-10) | (np.abs(btc_exposure) > 0.05 + 1e-10)
                | (volatility > 0.1 + 1e-10)
            )
            squared = np.square(np.abs(projected)).sum(axis=1)
            effective_count = np.divide(
                np.square(gross), squared, out=np.zeros_like(gross), where=squared > 0.0
            )
            concentration = np.divide(
                max_coin, gross, out=np.zeros_like(gross), where=gross > 0.0
            )
            hourly_risk = pd.DataFrame({
                "model_id": model_id,
                "cost_bps": cost,
                "timestamp": ledger.projection.targets.index,
                "long_gross": long_gross,
                "short_gross": short_gross,
                "total_gross": gross,
                "maximum_absolute_coin_weight": max_coin,
                "dollar_net": net,
                "btc_beta": btc_exposure,
                "ex_ante_annualized_volatility": volatility,
                "effective_coin_count": effective_count,
                "weight_concentration": concentration,
                "realized_gross_exposure": ledger.bars["gross_exposure"].to_numpy(float),
                "realized_net_exposure": ledger.bars["net_exposure"].to_numpy(float),
                "turnover": ledger.bars["turnover"].to_numpy(float),
                "transaction_cost": ledger.bars["transaction_cost"].to_numpy(float),
                "any_constraint_violation": violations,
            })
            risk_hourly_rows.append(hourly_risk)
            risk_rows.append({
                "model_id": model_id, "cost_bps": cost,
                "constraint_violations": int(violations.sum()),
                "maximum_total_gross": float(gross.max()),
                "maximum_absolute_coin_weight": float(max_coin.max()),
                "maximum_absolute_dollar_net": float(np.abs(net).max()),
                "maximum_absolute_btc_beta": float(np.abs(btc_exposure).max()),
                "maximum_annualized_volatility": float(volatility.max()),
                "mean_long_gross": float(long_gross.mean()),
                "mean_short_gross": float(short_gross.mean()),
                "mean_effective_coin_count": float(effective_count.mean()),
                "median_effective_coin_count": float(np.median(effective_count)),
                "mean_weight_concentration": float(concentration.mean()),
                "maximum_weight_concentration": float(concentration.max()),
            })
    if len(hourly_rows) != 8 or any(row["constraint_violations"] for row in risk_rows):
        raise ConfirmationGateError("ledger count or risk constraints differ")

    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        daily_frame = pd.concat(daily_rows, ignore_index=True)
        paired = daily_frame.loc[daily_frame["model_id"].eq("N02")].merge(
            daily_frame.loc[daily_frame["model_id"].eq("C02")],
            on=["date", "cost_bps"], suffixes=("_N02", "_C02"), validate="one_to_one",
        )
        paired["N02_minus_C02"] = paired["net_return_N02"] - paired["net_return_C02"]
        bootstrap_rows: list[dict[str, Any]] = []
        for cost in (0.0, 0.5, 1.5, 5.0):
            rng = np.random.default_rng(0x13527D8)
            values = paired.loc[np.isclose(paired["cost_bps"], cost), "N02_minus_C02"].to_numpy(dtype=float)
            if len(values) < 14 or not np.isfinite(values).all():
                raise ConfirmationGateError("paired daily series is invalid")
            block = 7
            blocks_needed = int(math.ceil(len(values) / block))
            means = np.empty(10000, dtype=float)
            for replication in range(10000):
                starts = rng.integers(0, len(values) - block + 1, size=blocks_needed)
                sample = np.concatenate([values[start : start + block] for start in starts])[:len(values)]
                means[replication] = sample.mean()
            bootstrap_rows.append({
                "cost_bps": cost,
                "daily_observations": len(values),
                "mean_N02_minus_C02": float(values.mean()),
                "block_length_days": 7,
                "replications": 10000,
                "seed": 0x13527D8,
                "one_sided_90_lower": float(np.quantile(means, 0.10)),
                "two_sided_95_lower": float(np.quantile(means, 0.025)),
                "two_sided_95_upper": float(np.quantile(means, 0.975)),
            })
        metric_frame = pd.DataFrame(metric_rows)
        refit = metric_frame.loc[metric_frame["scope"].eq("refit")]
        refit_stability = refit.loc[refit["model_id"].eq("N02")].merge(
            refit.loc[refit["model_id"].eq("C02")],
            on=["refit", "cost_bps", "scope"], suffixes=("_N02", "_C02"),
            validate="one_to_one",
        )
        refit_stability["sharpe_difference_N02_minus_C02"] = (
            refit_stability["sharpe_N02"] - refit_stability["sharpe_C02"]
        )
        refit_stability["N02_positive_sharpe"] = refit_stability["sharpe_N02"] > 0.0
        refit_stability["N02_beats_C02"] = (
            refit_stability["sharpe_difference_N02_minus_C02"] > 0.0
        )
        tables = {
            "hourly_ledger.csv": pd.concat(hourly_rows, ignore_index=True),
            "daily_returns.csv": daily_frame,
            "daily_attribution.csv": pd.concat(attribution_rows, ignore_index=True),
            "portfolio_metrics.csv": metric_frame,
            "daily_coin_attribution.csv": pd.concat(coin_rows, ignore_index=True),
            "risk_audit_summary.csv": pd.DataFrame(risk_rows),
            "hourly_constraint_audit.csv": pd.concat(risk_hourly_rows, ignore_index=True),
            "forecast_ic.csv": pd.DataFrame(ic_rows),
            "paired_daily_differences.csv": paired,
            "paired_block_bootstrap.csv": pd.DataFrame(bootstrap_rows),
            "refit_stability.csv": refit_stability,
        }
        for name, frame in tables.items():
            _write_table(staging / name, frame)
        hashes = {name: sha256_file(staging / name) for name in sorted(tables)}
        atomic_json(staging / "artifact_hashes.json", {
            "schema": "pairtoken_1m_confirmation_evaluation_hashes_v1", "files": hashes
        })
        primary = pd.DataFrame(metric_rows)
        primary = primary.loc[
            primary["scope"].eq("aggregate") & np.isclose(primary["cost_bps"], 0.5)
        ].set_index("model_id")
        primary_bootstrap = pd.DataFrame(bootstrap_rows).set_index("cost_bps").loc[0.5]
        receipt = {
            "schema": "pairtoken_1m_confirmation_evaluation_completion_v1",
            "status": "complete_single_use_two_model_four_cost_continuous_evaluation",
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "source_freeze_sha256": sha256_file(source_freeze_path),
            "forecast_seal_sha256": sha256_file(forecast_seal_path),
            "authorization_sha256": sha256_file(authorization_path),
            "authorization_consumed_sha256": sha256_file(consumed_path),
            "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
            "artifact_hashes_sha256": sha256_file(staging / "artifact_hashes.json"),
            "ledger_count": 8,
            "decision_hours": 8015,
            "terminal_rows": 1,
            "constraint_violations": 0,
            "primary_cost_bps": 0.5,
            "N02_primary_sharpe": float(primary.loc["N02", "sharpe"]),
            "C02_primary_sharpe": float(primary.loc["C02", "sharpe"]),
            "primary_mean_paired_daily_difference": float(
                primary_bootstrap["mean_N02_minus_C02"]
            ),
            "primary_one_sided_90_block_bootstrap_lower": float(
                primary_bootstrap["one_sided_90_lower"]
            ),
            "confirmation_outcomes_opened": True,
        }
        atomic_json(staging / "completion_receipt.json", receipt)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def run_synthetic_self_test(
    *, audited_root: Path, output: Path, work_root: Path | None = None
) -> dict[str, Any]:
    """Exercise seal, authorization consumption, eight audited ledgers and receipt."""

    if Path(output).exists():
        raise FileExistsError(output)
    temporary = work_root is None
    root = (
        Path(tempfile.mkdtemp(prefix="pairtoken-confirmation-self-test-"))
        if temporary
        else Path(work_root).resolve()
    )
    if not temporary:
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
    try:
        contract = load_contract()
        freeze_path = root / "source_freeze.json"
        create_source_freeze(freeze_path)
        dependencies = audited_dependencies(audited_root)
        blinded_root = root / "blinded"
        forecast_root = root / "forecasts"
        symbols = tuple(contract["data"]["symbols_in_order"])
        pairs = pair_ids(symbols)
        legs = np.asarray([
            (second, first)
            for first in range(24) for second in range(first + 1, 24)
        ], dtype=np.int64)
        runtime = {
            "self_test": True,
            "purpose": "synthetic_protocol_smoke_without_confirmation_data",
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        _bind_runtime_manifest(forecast_root, runtime, freeze_path)
        blinded_bindings: list[dict[str, str]] = []
        for fold in contract["refits"]:
            train_ns, validation_ns, test_ns = _expected_partition_clocks(fold)
            first = int(train_ns[0]) - 168 * 3_600_000_000_000
            source_clock = np.arange(
                first,
                int(test_ns[-1]) + 3_600_000_000_000,
                3_600_000_000_000,
                dtype=np.int64,
            )
            location = {int(value): row for row, value in enumerate(source_clock)}
            train_rows = np.asarray([location[int(value)] for value in train_ns], dtype=np.int64)
            validation_rows = np.asarray(
                [location[int(value)] for value in validation_ns], dtype=np.int64
            )
            test_rows = np.asarray([location[int(value)] for value in test_ns], dtype=np.int64)
            rows = len(source_clock)
            factorized = {
                "clock_source_ns": source_clock,
                "pair_ids": np.asarray(pairs),
                "pair_leg_indices": legs,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "test_rows": test_rows,
                "coin_minute_returns": np.zeros((rows, 24, 60), dtype=np.float32),
                "pair_weights": np.broadcast_to(
                    np.asarray([0.5, -0.5], dtype=np.float64), (rows, 276, 2)
                ).copy(),
                "pair_path_endpoints": np.zeros((rows, 276), dtype=np.float32),
                "feature_available": np.ones((rows, 276), dtype=bool),
                "target_cashflows_train": np.zeros((len(train_rows), 276), dtype=np.float64),
                "target_cashflows_validation": np.zeros(
                    (len(validation_rows), 276), dtype=np.float64
                ),
                "target_available_train": np.ones((len(train_rows), 276), dtype=bool),
                "target_available_validation": np.ones(
                    (len(validation_rows), 276), dtype=bool
                ),
            }
            content_hash = _array_family_sha256(factorized)
            for representation, model_id in (
                ("native_1m", "N02"), ("causal_15m_control", "C02")
            ):
                input_path = blinded_root / fold["name"] / f"{representation}.blinded.npz"
                metadata = {
                    "schema": INPUT_SCHEMA,
                    "status": "complete_factorized_train_validation_labels_and_test_features_only",
                    "contract_sha256": sha256_file(CONTRACT_PATH),
                    "source_freeze_sha256": sha256_file(freeze_path),
                    "source_bundle_sha256": verify_source_freeze(freeze_path)["source_bundle_sha256"],
                    "fold": fold["name"],
                    "representation": representation,
                    "pair_count": 276,
                    "raw_dimension": 232,
                    "auxiliary_dimension": 84,
                    "auxiliary_values": "all_zeros_created_by_trainer",
                    "pair_axis_sha256": axis_sha256(pairs),
                    "test_target_labels_included": False,
                    "causal_test_history_features_included": True,
                    "confirmation_metrics_computed": False,
                    "factorized_storage": True,
                    "factorized_content_sha256": content_hash,
                }
                _write_npz(input_path, {
                    "metadata_json": np.asarray(json.dumps(
                        metadata, sort_keys=True, separators=(",", ":")
                    )),
                    **factorized,
                })
                input_hash = sha256_file(input_path)
                blinded_bindings.append({
                    "fold": fold["name"],
                    "representation": representation,
                    "relative_path": f"{fold['name']}/{representation}.blinded.npz",
                    "sha256": input_hash,
                })
                row_phase = np.sin(np.arange(len(test_ns), dtype=float)[:, None] / 37.0)
                pair_axis = np.linspace(-1.0, 1.0, 276, dtype=float)[None, :]
                base = pair_axis + 0.001 * row_phase
                if representation == "causal_15m_control":
                    base = -base
                seed_predictions: list[np.ndarray] = []
                for seed_index, seed in enumerate((1701, 1702, 1703)):
                    prediction = (base + (seed_index - 1) * 1e-4).astype(np.float32)
                    spec = ForecastSpec(
                        fold["name"], representation, str(seed), "individual_seed"
                    )
                    artifact_path = expected_artifact_path(forecast_root, spec)
                    diagnostics = _diagnostics(
                        spec=spec,
                        input_hash=input_hash,
                        source_freeze_path=freeze_path,
                        runtime=runtime,
                        epochs=1,
                        training_loss=0.0,
                        validation_loss=0.0,
                    )
                    write_forecast_artifact(
                        artifact_path,
                        clock_ns=test_ns,
                        pair_axis=pairs,
                        feature_available=np.ones(prediction.shape, dtype=bool),
                        predictions=prediction,
                        diagnostics=diagnostics,
                        pair_targets=None,
                    )
                    register_forecast(
                        forecast_root,
                        spec=spec,
                        artifact_path=artifact_path,
                        input_sha256=input_hash,
                        source_freeze_path=freeze_path,
                    )
                    seed_predictions.append(prediction)
                ensemble = (
                    np.stack(seed_predictions).astype(np.float64).sum(axis=0) / 3.0
                ).astype(np.float32)
                mask = np.ones(ensemble.shape, dtype=bool)
                ensemble_spec = ForecastSpec(
                    fold["name"], representation, "all3", "seed_mean"
                )
                ensemble_path = expected_artifact_path(forecast_root, ensemble_spec)
                write_forecast_artifact(
                    ensemble_path,
                    clock_ns=test_ns,
                    pair_axis=pairs,
                    feature_available=mask,
                    predictions=ensemble,
                    diagnostics=_diagnostics(
                        spec=ensemble_spec,
                        input_hash=input_hash,
                        source_freeze_path=freeze_path,
                        runtime=runtime,
                    ),
                    pair_targets=rank_pair_targets(ensemble, mask),
                )
                register_forecast(
                    forecast_root,
                    spec=ensemble_spec,
                    artifact_path=ensemble_path,
                    input_sha256=input_hash,
                    source_freeze_path=freeze_path,
                )
            del factorized

        seal_path = root / "sealed" / "all_fold_forecast_seal.json"
        seal_all_forecasts(
            forecast_root, source_freeze_path=freeze_path, output=seal_path
        )
        decision_clock = pd.date_range(
            "2025-08-01T00:00:00Z", "2026-06-30T22:00:00Z", freq="1h"
        )
        ledger_clock = decision_clock.append(
            pd.DatetimeIndex([pd.Timestamp("2026-06-30T23:00:00Z")])
        )
        hours = np.arange(len(ledger_clock), dtype=float)[:, None]
        coin_axis = np.arange(24, dtype=float)[None, :]
        returns = 2e-5 * np.sin(hours / 17.0 + coin_axis / 5.0)
        returns += (coin_axis - 11.5) * 2e-7 * np.cos(hours / 29.0)
        returns[-1] = 0.0
        funding = np.zeros_like(returns)
        funding[8::8] = 2e-7 * np.where((np.arange(len(funding[8::8]))[:, None] % 2) == 0, 1.0, -1.0)
        opens = np.empty_like(returns)
        opens[0] = 100.0 + np.arange(24)
        opens[1:] = opens[0] * np.cumprod(1.0 + returns[:-1], axis=0)
        pair_weights = np.broadcast_to(
            np.asarray([0.5, -0.5]), (len(ledger_clock), 276, 2)
        ).copy()
        actual = np.empty((len(decision_clock), 276), dtype=np.float64)
        for pair, (signal, hedge) in enumerate(legs):
            price = 0.5 * returns[:-1, signal] - 0.5 * returns[:-1, hedge]
            next_funding = -(
                0.5 * (1.0 + returns[:-1, signal]) * funding[1:, signal]
                - 0.5 * (1.0 + returns[:-1, hedge]) * funding[1:, hedge]
            )
            actual[:, pair] = price + next_funding
        covariance = np.zeros((len(ledger_clock), 24, 24), dtype=np.float64)
        diagonal = np.arange(24)
        covariance[:, diagonal, diagonal] = 1e-6
        terminal_mask = np.zeros(len(ledger_clock), dtype=bool)
        terminal_mask[-1] = True
        evaluation_directory = root / "evaluation"
        payload_path = evaluation_directory / "evaluation_payload.npz"
        _write_npz(payload_path, {
            "metadata_json": np.asarray(json.dumps({
                "schema": "pairtoken_1m_confirmation_evaluation_payload_v1",
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "source_freeze_sha256": sha256_file(freeze_path),
                "forecast_payload_absent": True,
                "portfolio_metrics_computed": False,
                "decision_rows": 8015,
                "ledger_rows": 8016,
                "terminal_rule": "zero_pair_target_liquidation_at_observed_2026_06_30_23_open",
                "acquisition_manifest_sha256": "1" * 64,
                "archive_availability_sha256": "2" * 64,
                "artifact_hashes_sha256": "3" * 64,
            }, sort_keys=True, separators=(",", ":"))),
            "decision_clock_ns": decision_clock.asi8,
            "ledger_clock_ns": ledger_clock.asi8,
            "symbols": np.asarray(symbols),
            "pair_ids": np.asarray(pairs),
            "pair_leg_indices": legs,
            "pair_weights": pair_weights,
            "coin_open_prices": opens,
            "coin_open_to_open_returns": returns,
            "funding_at_boundary": funding,
            "coin_availability": np.ones((len(ledger_clock), 24), dtype=bool),
            "pair_decision_available": np.ones((len(decision_clock), 276), dtype=bool),
            "pair_target_cashflows": actual,
            "pair_target_available": np.ones(actual.shape, dtype=bool),
            "forecast_feature_available": np.ones(actual.shape, dtype=bool),
            "risk_beta": np.zeros((len(ledger_clock), 24), dtype=np.float64),
            "risk_covariance": covariance,
            "terminal_row_mask": terminal_mask,
        })
        evaluation_manifest = evaluation_directory / "evaluation_payload.manifest.json"
        atomic_json(evaluation_manifest, {
            "schema": EVALUATION_MANIFEST_SCHEMA,
            "status": "sealed_economic_and_execution_inputs_never_opened_by_forecast_stage",
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "source_freeze_sha256": sha256_file(freeze_path),
            "payload_path": payload_path.name,
            "payload_sha256": sha256_file(payload_path),
            "payload_role": "confirmation_outcomes_and_execution_only",
            "clock_start": "2025-08-01T00:00:00Z",
            "clock_end_exclusive": "2026-06-30T23:00:00Z",
            "terminal_rule": "zero_pair_target_liquidation_at_observed_2026_06_30_23_open",
            "symbols": list(symbols),
            "pairs": 276,
            "funding_included": True,
            "cost_grid_bps": [0.0, 0.5, 1.5, 5.0],
            "risk_engine": {
                "covariance_lookback_days": 60,
                "minimum_complete_hourly_observations": 720,
                "maximum_total_gross": 1.0,
                "maximum_absolute_coin_weight": 0.15,
                "maximum_absolute_dollar_net": 0.02,
                "maximum_absolute_btc_beta": 0.05,
                "target_annualized_volatility": 0.1,
            },
            "audited_source_hashes": dict(dependencies.hashes),
            "blinded_input_artifacts": blinded_bindings,
            "forecast_payload_absent": True,
            "portfolio_metrics_computed": False,
        })
        authorization_path = root / "sealed" / "evaluation_authorization.json"
        authorize_evaluation(
            forecast_root=forecast_root,
            source_freeze_path=freeze_path,
            forecast_seal_path=seal_path,
            evaluation_manifest_path=evaluation_manifest,
            output=authorization_path,
        )
        results = root / "results"
        completion = run_evaluation(
            forecast_root=forecast_root,
            source_freeze_path=freeze_path,
            forecast_seal_path=seal_path,
            evaluation_manifest_path=evaluation_manifest,
            authorization_path=authorization_path,
            blinded_input_root=blinded_root,
            audited_root=audited_root,
            output=results,
        )
        reuse_rejected = False
        try:
            run_evaluation(
                forecast_root=forecast_root,
                source_freeze_path=freeze_path,
                forecast_seal_path=seal_path,
                evaluation_manifest_path=evaluation_manifest,
                authorization_path=authorization_path,
                blinded_input_root=blinded_root,
                audited_root=audited_root,
                output=root / "forbidden_second_results",
            )
        except ConfirmationGateError as exc:
            reuse_rejected = "already consumed" in str(exc)
        if not reuse_rejected:
            raise ConfirmationGateError("synthetic self-test did not enforce single use")
        receipt = {
            "schema": "pairtoken_1m_confirmation_synthetic_self_test_v1",
            "status": "passed_seal_authorize_consume_eight_ledgers_and_completion",
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "confirmation_data_used": False,
            "synthetic_decision_hours": 8015,
            "forecast_artifacts": 32,
            "ledger_count": completion["ledger_count"],
            "single_use_reuse_rejected": True,
            "completion_receipt_sha256": sha256_file(results / "completion_receipt.json"),
            "work_root_preserved": not temporary,
            "work_root": str(root) if not temporary else None,
        }
        atomic_json(output, receipt)
        return receipt
    finally:
        if temporary and root.exists():
            shutil.rmtree(root)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    run = subparsers.add_parser("run")
    for child in (authorize, run):
        child.add_argument("--forecast-root", type=Path, required=True)
        child.add_argument("--source-freeze", type=Path, required=True)
        child.add_argument("--forecast-seal", type=Path, required=True)
        child.add_argument("--evaluation-manifest", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--blinded-input-root", type=Path, required=True)
    run.add_argument("--audited-root", type=Path, default=DEFAULT_AUDITED_ROOT)
    self_test = subparsers.add_parser("self-test")
    self_test.add_argument("--audited-root", type=Path, default=DEFAULT_AUDITED_ROOT)
    self_test.add_argument("--output", type=Path, required=True)
    self_test.add_argument("--work-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    if args.command == "authorize":
        result = authorize_evaluation(
            forecast_root=args.forecast_root,
            source_freeze_path=args.source_freeze,
            forecast_seal_path=args.forecast_seal,
            evaluation_manifest_path=args.evaluation_manifest,
            output=args.output,
        )
    elif args.command == "run":
        result = run_evaluation(
            forecast_root=args.forecast_root,
            source_freeze_path=args.source_freeze,
            forecast_seal_path=args.forecast_seal,
            evaluation_manifest_path=args.evaluation_manifest,
            authorization_path=args.authorization,
            blinded_input_root=args.blinded_input_root,
            audited_root=args.audited_root,
            output=args.output,
        )
    else:
        result = run_synthetic_self_test(
            audited_root=args.audited_root,
            output=args.output,
            work_root=args.work_root,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
