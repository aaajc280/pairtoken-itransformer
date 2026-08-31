#!/usr/bin/env python3
"""Fail-closed governance primitives for the one-shot confirmation.

This module never imports a market-data reader and never computes an economic
metric.  Its only responsibilities are contract validation, source binding,
deterministic forecast serialization, exact registry enumeration, and the
all-fold seal checked before the evaluation key may be opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contract.json"
CONTRACT_NAME = "pairtoken_native_1m_itransformer_one_shot_confirmation"
CONTRACT_VERSION = 1
FORECAST_SCHEMA = "pairtoken_1m_confirmation_forecast_v1"
REGISTRY_SCHEMA = "pairtoken_1m_confirmation_registry_entry_v1"
SEAL_SCHEMA = "pairtoken_1m_confirmation_all_fold_seal_v1"
SOURCE_FREEZE_SCHEMA = "pairtoken_1m_confirmation_source_freeze_v1"
AUTHORIZATION_SCHEMA = "pairtoken_1m_confirmation_evaluation_authorization_v1"
REPRESENTATIONS = ("native_1m", "causal_15m_control")
MODEL_IDS = {
    "native_1m": "N02",
    "causal_15m_control": "C02",
}
SEEDS = (1701, 1702, 1703)
PAIR_COUNT = 276
RAW_DIMENSION = 232
AUXILIARY_DIMENSION = 84
PATH_DIMENSION = 60
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PACKAGE_SOURCE_NAMES = (
    "contract.json",
    "acquisition_plan.json",
    "confirmation_governance.py",
    "source_freeze.py",
    "acquisition_plan.py",
    "acquire_confirmation.py",
    "prepare_confirmation.py",
    "train_confirmation.py",
    "seal_forecasts.py",
    "evaluate_confirmation.py",
)


class ConfirmationGateError(RuntimeError):
    """A frozen specification, artifact, or stage-boundary invariant failed."""


@dataclass(frozen=True, order=True)
class ForecastSpec:
    fold: str
    representation: str
    seed: str
    role: str

    @property
    def model_id(self) -> str:
        return MODEL_IDS[self.representation]

    @property
    def filename(self) -> str:
        return (
            f"{self.representation}__itransformer__"
            f"seed-{self.seed}__{self.role}.npz"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite JSON numbers are forbidden")
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def atomic_create(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"create-only artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_create(path, canonical_json(value))


def _npy_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ConfirmationGateError("object arrays are forbidden")
    output = BytesIO()
    np.lib.format.write_array(output, array, allow_pickle=False)
    return output.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    names = tuple(sorted(map(str, arrays)))
    if not names or len(names) != len(arrays) or any(not name or "/" in name for name in names):
        raise ConfirmationGateError("invalid NPZ member names")
    output = BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in names:
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationGateError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ConfirmationGateError(f"{label} must be one JSON object")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ConfirmationGateError(f"timezone is missing: {value}")
    return parsed.astimezone(timezone.utc)


def hourly_clock_ns(start: str, end_exclusive: str) -> np.ndarray:
    first = int(_utc(start).timestamp()) * 1_000_000_000
    stop = int(_utc(end_exclusive).timestamp()) * 1_000_000_000
    step = 3_600_000_000_000
    if first % step or stop % step or stop <= first:
        raise ConfirmationGateError("hourly interval is invalid")
    return np.arange(first, stop, step, dtype=np.int64)


def pair_ids(symbols: Sequence[str]) -> tuple[str, ...]:
    names = tuple(map(str, symbols))
    return tuple(
        f"{names[second]}__{names[first]}"
        for first in range(len(names))
        for second in range(first + 1, len(names))
    )


def axis_sha256(values: Sequence[str]) -> str:
    return sha256((json.dumps(list(map(str, values)), separators=(",", ":")) + "\n").encode()).hexdigest()


def load_contract() -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH, label="confirmation contract")
    if contract.get("contract_name") != CONTRACT_NAME or contract.get("contract_version") != CONTRACT_VERSION:
        raise ConfirmationGateError("confirmation contract identity differs")
    interval = contract.get("confirmation_interval", {})
    expected_clock = hourly_clock_ns(
        interval.get("start_inclusive_utc", ""),
        interval.get("end_exclusive_utc", ""),
    )
    if len(expected_clock) != 8015 or interval.get("expected_decision_hours") != 8015:
        raise ConfirmationGateError("confirmation interval is not exactly 8,015 hours")
    refits = contract.get("refits")
    if not isinstance(refits, list) or len(refits) != 4:
        raise ConfirmationGateError("exactly four refits are required")
    concatenated: list[np.ndarray] = []
    names: set[str] = set()
    for row in refits:
        if set(row) != {
            "name", "refit_at_utc", "test_start", "test_end_exclusive", "expected_test_hours"
        }:
            raise ConfirmationGateError("refit schema differs")
        if row["name"] in names or row["refit_at_utc"] != row["test_start"]:
            raise ConfirmationGateError("refit identity or timestamp differs")
        names.add(row["name"])
        clock = hourly_clock_ns(row["test_start"], row["test_end_exclusive"])
        if len(clock) != int(row["expected_test_hours"]):
            raise ConfirmationGateError(f"refit-hour count differs: {row['name']}")
        concatenated.append(clock)
    if not np.array_equal(np.concatenate(concatenated), expected_clock):
        raise ConfirmationGateError("refits do not partition the confirmation interval")
    symbols = contract.get("data", {}).get("symbols_in_order", [])
    pairs = pair_ids(symbols)
    if len(symbols) != 24 or len(set(symbols)) != 24 or len(pairs) != PAIR_COUNT:
        raise ConfirmationGateError("static Top-24/276 axes differ")
    if contract["model"].get("fixed_seeds") != list(SEEDS):
        raise ConfirmationGateError("fixed seed family differs")
    separation = contract.get("forecast_separation", {})
    if separation.get("required_total_forecast_artifacts") != 32:
        raise ConfirmationGateError("forecast-artifact budget differs")
    if contract["portfolio"] != {
        "mapping": "frozen parent rank_pair_targets after the three-seed arithmetic mean",
        "long_pairs": 8,
        "short_pairs": 8,
        "absolute_pair_target": 0.05,
        "stable_tie_break": "2021 rank-combination pair order",
        "flat_if_fewer_than_16_eligible_pairs": True,
        "pair_targets_netted_to_coins_before_constraints_and_costs": True,
        "maximum_total_gross": 1.0,
        "maximum_absolute_coin_weight": 0.15,
        "maximum_absolute_dollar_net": 0.02,
        "maximum_absolute_btc_beta": 0.05,
        "target_annualized_volatility": 0.1,
        "risk_covariance_lookback_days": 60,
        "risk_minimum_complete_hourly_observations": 720,
        "annualization_days": 365,
        "cost_bps": [0.0, 0.5, 1.5, 5.0],
        "primary_cost_bps": 0.5,
        "stress_cost_bps": 1.5,
        "funding": "realized Binance funding cashflows at governed boundaries are included",
        "transaction_cost_formula": "one-way cost equals cost_bps times traded gross divided by 10000 at every hourly action, including entry, rebalancing, and terminal liquidation",
    }:
        raise ConfirmationGateError("portfolio/risk/cost contract differs")
    return contract


def expected_specs(contract: Mapping[str, Any] | None = None) -> tuple[ForecastSpec, ...]:
    frozen = load_contract() if contract is None else contract
    result: list[ForecastSpec] = []
    for fold in frozen["refits"]:
        for representation in REPRESENTATIONS:
            result.extend(
                ForecastSpec(fold["name"], representation, str(seed), "individual_seed")
                for seed in SEEDS
            )
            result.append(ForecastSpec(fold["name"], representation, "all3", "seed_mean"))
    if len(result) != 32 or len(set(result)) != 32:
        raise ConfirmationGateError("expected forecast family is not exactly 32")
    return tuple(result)


def expected_artifact_path(root: Path, spec: ForecastSpec) -> Path:
    return Path(root) / "forecasts" / spec.fold / spec.filename


def expected_registry_path(root: Path, spec: ForecastSpec) -> Path:
    return Path(root) / "registry" / spec.fold / f"{spec.filename}.json"


def fold_contract(contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = [row for row in contract["refits"] if row["name"] == name]
    if len(rows) != 1:
        raise ConfirmationGateError(f"unknown or duplicate refit: {name}")
    return rows[0]


def parent_source_paths(contract: Mapping[str, Any]) -> dict[str, tuple[Path, str]]:
    parent = contract["frozen_parent"]
    mapping = {
        "paired_contract": (parent["paired_contract_relative_path"], parent["paired_contract_sha256"]),
        "paired_core": (parent["paired_core_relative_path"], parent["paired_core_sha256"]),
        "paired_models": (parent["paired_models_relative_path"], parent["paired_models_sha256"]),
        "development_runner": (parent["development_runner_relative_path"], parent["development_runner_sha256"]),
        "development_evaluator": (parent["development_evaluator_relative_path"], parent["development_evaluator_sha256"]),
        "cycle_model": (parent["cycle_model_relative_path"], parent["cycle_model_sha256"]),
        "raw_4h_contract": (parent["raw_4h_contract_relative_path"], parent["raw_4h_contract_sha256"]),
    }
    return {
        name: ((HERE / relative).resolve(), str(digest))
        for name, (relative, digest) in mapping.items()
    }


def current_source_hashes(contract: Mapping[str, Any] | None = None) -> dict[str, str]:
    frozen = load_contract() if contract is None else contract
    hashes: dict[str, str] = {}
    for name in PACKAGE_SOURCE_NAMES:
        path = HERE / name
        if not path.is_file():
            raise ConfirmationGateError(f"package source is missing: {path}")
        hashes[f"package/{name}"] = sha256_file(path)
    for name, (path, expected) in parent_source_paths(frozen).items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ConfirmationGateError(f"frozen parent source differs: {name}")
        hashes[f"parent/{name}"] = expected
    return hashes


def source_bundle_sha256(hashes: Mapping[str, str]) -> str:
    if not hashes or any(SHA256_RE.fullmatch(str(value)) is None for value in hashes.values()):
        raise ConfirmationGateError("source hash family is invalid")
    return sha256(canonical_json(dict(hashes))).hexdigest()


def create_source_freeze(path: Path) -> dict[str, Any]:
    contract = load_contract()
    hashes = current_source_hashes(contract)
    payload = {
        "schema": SOURCE_FREEZE_SCHEMA,
        "status": "complete_before_confirmation_acquisition_or_outcome_open",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_hashes": hashes,
        "source_bundle_sha256": source_bundle_sha256(hashes),
        "confirmation_data_parsed": False,
        "confirmation_metrics_computed": False,
        "development_runner_called": False,
        "development_evaluator_called": False,
    }
    atomic_json(path, payload)
    return payload


def verify_source_freeze(path: Path) -> dict[str, Any]:
    receipt = load_json(path, label="source-freeze receipt")
    required = {
        "schema", "status", "contract_sha256", "source_hashes", "source_bundle_sha256",
        "confirmation_data_parsed", "confirmation_metrics_computed",
        "development_runner_called", "development_evaluator_called",
    }
    if set(receipt) != required or receipt.get("schema") != SOURCE_FREEZE_SCHEMA:
        raise ConfirmationGateError("source-freeze schema differs")
    if receipt.get("status") != "complete_before_confirmation_acquisition_or_outcome_open":
        raise ConfirmationGateError("source freeze is incomplete")
    if any(receipt.get(name) is not False for name in (
        "confirmation_data_parsed", "confirmation_metrics_computed",
        "development_runner_called", "development_evaluator_called",
    )):
        raise ConfirmationGateError("source freeze was not result blind")
    if receipt.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ConfirmationGateError("source freeze binds a different contract")
    current = current_source_hashes(load_contract())
    if receipt.get("source_hashes") != current:
        raise ConfirmationGateError("source bytes changed after the freeze")
    if receipt.get("source_bundle_sha256") != source_bundle_sha256(current):
        raise ConfirmationGateError("source-bundle hash differs")
    return receipt


def write_forecast_artifact(
    path: Path,
    *,
    clock_ns: np.ndarray,
    pair_axis: Sequence[str],
    feature_available: np.ndarray,
    predictions: np.ndarray,
    diagnostics: Mapping[str, Any],
    pair_targets: np.ndarray | None,
) -> str:
    mask = np.asarray(feature_available, dtype=bool)
    values = np.asarray(predictions, dtype=np.float32)
    pairs = np.asarray(tuple(map(str, pair_axis)))
    if values.shape != mask.shape or values.shape != (len(clock_ns), PAIR_COUNT):
        raise ConfirmationGateError("forecast axes differ")
    if len(pairs) != PAIR_COUNT or len(set(pairs.tolist())) != PAIR_COUNT:
        raise ConfirmationGateError("forecast pair axis differs")
    if np.isinf(values).any() or not np.array_equal(np.isfinite(values), mask):
        raise ConfirmationGateError("forecast finite mask differs")
    arrays: dict[str, np.ndarray] = {
        "clock_ns": np.asarray(clock_ns, dtype=np.int64),
        "pair_ids": pairs,
        "feature_available": mask,
        "predictions": values,
        "diagnostics_json": np.asarray(json.dumps(_jsonable(diagnostics), sort_keys=True, separators=(",", ":"))),
    }
    if pair_targets is not None:
        targets = np.asarray(pair_targets, dtype=np.float32)
        if targets.shape != mask.shape or not np.isfinite(targets).all():
            raise ConfirmationGateError("pair-target axes or values differ")
        if np.any((targets != 0.0) & ~mask):
            raise ConfirmationGateError("pair targets trade an unavailable pair")
        arrays["pair_targets"] = targets
    payload = deterministic_npz_bytes(arrays)
    atomic_create(path, payload)
    return sha256_bytes(payload)


def rank_pair_targets(
    predictions: np.ndarray, available: np.ndarray
) -> np.ndarray:
    """Exact frozen stable Top-8/Bottom-8 map, locally reproduced for sealing."""

    values = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ConfirmationGateError("rank-map axes differ")
    result = np.zeros_like(values)
    for row in range(len(values)):
        eligible = np.flatnonzero(mask[row] & np.isfinite(values[row]))
        if len(eligible) < 16:
            continue
        row_values = values[row, eligible]
        if float(np.ptp(row_values)) <= 1e-12:
            continue
        order = eligible[np.argsort(row_values, kind="mergesort")]
        result[row, order[:8]] = -0.05
        result[row, order[-8:]] = 0.05
    return result.astype(np.float32)


def read_forecast_artifact(path: Path, spec: ForecastSpec, contract: Mapping[str, Any]) -> dict[str, Any]:
    expected_members = {
        "clock_ns", "pair_ids", "feature_available", "predictions", "diagnostics_json"
    }
    if spec.role == "seed_mean":
        expected_members.add("pair_targets")
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != expected_members:
                raise ConfirmationGateError(f"forecast member family differs: {path}")
            clock_ns = np.asarray(stored["clock_ns"], dtype=np.int64)
            pairs = tuple(stored["pair_ids"].astype(str))
            mask = np.asarray(stored["feature_available"], dtype=bool)
            predictions = np.asarray(stored["predictions"], dtype=np.float32)
            diagnostics = json.loads(str(stored["diagnostics_json"].item()))
            pair_targets = (
                np.asarray(stored["pair_targets"], dtype=np.float32)
                if spec.role == "seed_mean" else None
            )
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ConfirmationGateError(f"invalid forecast artifact: {path}") from exc
    fold = fold_contract(contract, spec.fold)
    expected_clock = hourly_clock_ns(fold["test_start"], fold["test_end_exclusive"])
    expected_pairs = pair_ids(contract["data"]["symbols_in_order"])
    if not np.array_equal(clock_ns, expected_clock) or pairs != expected_pairs:
        raise ConfirmationGateError(f"forecast clock or pair axis differs: {path}")
    shape = (len(expected_clock), PAIR_COUNT)
    if mask.shape != shape or predictions.shape != shape:
        raise ConfirmationGateError(f"forecast shape differs: {path}")
    if np.isinf(predictions).any() or not np.array_equal(np.isfinite(predictions), mask):
        raise ConfirmationGateError(f"forecast finite mask differs: {path}")
    if not isinstance(diagnostics, dict):
        raise ConfirmationGateError(f"forecast diagnostics are invalid: {path}")
    forbidden_fragments = ("target", "return", "sharpe", "drawdown", "turnover", "cost")
    forbidden_exact = {"ic", "mean_ic", "test_ic", "rank_ic", "spearman_ic"}
    allowed_target_keys = {"target_horizon_hours"}
    for key in diagnostics:
        lowered = str(key).lower()
        if key not in allowed_target_keys and (
            lowered in forbidden_exact
            or any(fragment in lowered for fragment in forbidden_fragments)
        ):
            raise ConfirmationGateError(f"outcome diagnostic leaked into forecast stage: {key}")
    identity = {
        "schema": FORECAST_SCHEMA,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "fold": spec.fold,
        "representation": spec.representation,
        "model_id": spec.model_id,
        "seed": spec.seed,
        "role": spec.role,
        "target_horizon_hours": 1,
        "confirmation_metrics_computed": False,
        "test_outcomes_loaded": False,
    }
    for key, value in identity.items():
        if diagnostics.get(key) != value:
            raise ConfirmationGateError(f"forecast identity differs for {key}: {path}")
    common_keys = {
        "schema", "contract_sha256", "source_freeze_sha256", "source_bundle_sha256",
        "input_artifact_sha256", "runtime_sha256", "fold", "representation", "model_id",
        "seed", "role", "target_horizon_hours", "confirmation_metrics_computed",
        "test_outcomes_loaded",
    }
    role_keys = (
        {"seeds", "aggregation"}
        if spec.role == "seed_mean"
        else {"epochs", "training_loss", "validation_loss"}
    )
    if set(diagnostics) != common_keys | role_keys:
        raise ConfirmationGateError(f"forecast diagnostic key family differs: {path}")
    for key in (
        "source_freeze_sha256", "source_bundle_sha256", "input_artifact_sha256", "runtime_sha256"
    ):
        if SHA256_RE.fullmatch(str(diagnostics.get(key))) is None:
            raise ConfirmationGateError(f"forecast diagnostic hash differs: {key}")
    if spec.role == "seed_mean":
        if diagnostics.get("seeds") != list(SEEDS) or diagnostics.get("aggregation") != (
            "arithmetic_mean_never_best_seed"
        ):
            raise ConfirmationGateError("seed-mean diagnostics differ")
    elif (
        not isinstance(diagnostics.get("epochs"), int)
        or diagnostics["epochs"] < 1
        or not np.isfinite((diagnostics.get("training_loss"), diagnostics.get("validation_loss"))).all()
    ):
        raise ConfirmationGateError("individual-seed training diagnostics differ")
    if spec.role == "seed_mean":
        assert pair_targets is not None
        if pair_targets.shape != shape or not np.isfinite(pair_targets).all():
            raise ConfirmationGateError(f"pair targets differ: {path}")
        if np.any((pair_targets != 0.0) & ~mask):
            raise ConfirmationGateError(f"pair targets trade an unavailable pair: {path}")
        rebuilt_targets = rank_pair_targets(predictions, mask)
        if not np.array_equal(pair_targets, rebuilt_targets):
            raise ConfirmationGateError(f"pair targets do not rebuild: {path}")
    return {
        "clock_ns": clock_ns,
        "pair_ids": pairs,
        "feature_available": mask,
        "predictions": predictions,
        "pair_targets": pair_targets,
        "diagnostics": diagnostics,
    }


def register_forecast(
    root: Path,
    *,
    spec: ForecastSpec,
    artifact_path: Path,
    input_sha256: str,
    source_freeze_path: Path,
) -> dict[str, Any]:
    contract = load_contract()
    freeze = verify_source_freeze(source_freeze_path)
    artifact = read_forecast_artifact(artifact_path, spec, contract)
    if SHA256_RE.fullmatch(input_sha256) is None:
        raise ConfirmationGateError("forecast input hash is invalid")
    resolved_root = Path(root).resolve()
    resolved_artifact = Path(artifact_path).resolve()
    try:
        relative = resolved_artifact.relative_to(resolved_root)
    except ValueError as exc:
        raise ConfirmationGateError("forecast artifact escapes its root") from exc
    expected = expected_artifact_path(resolved_root, spec).resolve()
    if resolved_artifact != expected:
        raise ConfirmationGateError("forecast artifact path is not governed")
    payload = {
        "schema": REGISTRY_SCHEMA,
        "status": "registered_before_confirmation_evaluation",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "fold": spec.fold,
        "representation": spec.representation,
        "model_id": spec.model_id,
        "seed": spec.seed,
        "role": spec.role,
        "input_artifact_sha256": input_sha256,
        "forecast_path": str(relative),
        "forecast_sha256": sha256_file(resolved_artifact),
        "confirmation_metrics_computed": False,
        "test_outcomes_loaded": False,
    }
    diagnostics = artifact["diagnostics"]
    if (
        diagnostics["source_freeze_sha256"] != sha256_file(source_freeze_path)
        or diagnostics["source_bundle_sha256"] != freeze["source_bundle_sha256"]
        or diagnostics["input_artifact_sha256"] != input_sha256
    ):
        raise ConfirmationGateError("forecast artifact is not bound to registry inputs")
    atomic_json(expected_registry_path(resolved_root, spec), payload)
    return payload


def verify_registry_entry(
    root: Path,
    spec: ForecastSpec,
    *,
    source_freeze_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    path = expected_registry_path(root, spec)
    entry = load_json(path, label="forecast registry entry")
    required = {
        "schema", "status", "contract_sha256", "source_freeze_sha256",
        "source_bundle_sha256", "fold", "representation", "model_id", "seed",
        "role", "input_artifact_sha256", "forecast_path", "forecast_sha256",
        "confirmation_metrics_computed", "test_outcomes_loaded",
    }
    if set(entry) != required or entry.get("schema") != REGISTRY_SCHEMA:
        raise ConfirmationGateError(f"registry schema differs: {path}")
    freeze = verify_source_freeze(source_freeze_path)
    expected_identity = {
        "status": "registered_before_confirmation_evaluation",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "fold": spec.fold,
        "representation": spec.representation,
        "model_id": spec.model_id,
        "seed": spec.seed,
        "role": spec.role,
        "confirmation_metrics_computed": False,
        "test_outcomes_loaded": False,
    }
    for key, value in expected_identity.items():
        if entry.get(key) != value:
            raise ConfirmationGateError(f"registry identity differs for {key}: {path}")
    if SHA256_RE.fullmatch(str(entry.get("input_artifact_sha256"))) is None:
        raise ConfirmationGateError("registry input hash is invalid")
    artifact = expected_artifact_path(root, spec).resolve()
    expected_relative = str(artifact.relative_to(Path(root).resolve()))
    if entry.get("forecast_path") != expected_relative:
        raise ConfirmationGateError("registry forecast path differs")
    if entry.get("forecast_sha256") != sha256_file(artifact):
        raise ConfirmationGateError("registered forecast hash differs")
    read_forecast_artifact(artifact, spec, contract)
    return entry


def seal_all_forecasts(
    root: Path,
    *,
    source_freeze_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract()
    freeze = verify_source_freeze(source_freeze_path)
    rows: list[dict[str, Any]] = []
    runtime_hashes: set[str] = set()
    ensembles: dict[tuple[str, str], dict[str, Any]] = {}
    individuals: dict[tuple[str, str, int], dict[str, Any]] = {}
    for spec in expected_specs(contract):
        entry = verify_registry_entry(
            root, spec, source_freeze_path=source_freeze_path, contract=contract
        )
        artifact = read_forecast_artifact(expected_artifact_path(root, spec), spec, contract)
        runtime_hashes.add(str(artifact["diagnostics"]["runtime_sha256"]))
        if spec.role == "seed_mean":
            ensembles[(spec.fold, spec.representation)] = artifact
        else:
            individuals[(spec.fold, spec.representation, int(spec.seed))] = artifact
        rows.append(dict(entry))
    for fold in (row["name"] for row in contract["refits"]):
        for representation in REPRESENTATIONS:
            ensemble = ensembles[(fold, representation)]
            stacked = np.stack([
                individuals[(fold, representation, seed)]["predictions"] for seed in SEEDS
            ]).astype(np.float64)
            all_nan = np.isnan(stacked).all(axis=0)
            if np.any(np.isnan(stacked) != all_nan[None, ...]):
                raise ConfirmationGateError("seed forecast missingness differs")
            rebuilt = np.where(
                all_nan,
                np.nan,
                np.where(np.isnan(stacked), 0.0, stacked).sum(axis=0) / len(SEEDS),
            ).astype(np.float32)
            if not np.array_equal(rebuilt, ensemble["predictions"], equal_nan=True):
                raise ConfirmationGateError("seed-mean forecast does not rebuild exactly")
    runtime_path = Path(root) / "runtime_manifest.json"
    runtime_manifest = load_json(runtime_path, label="runtime manifest")
    required_runtime = {
        "schema", "contract_sha256", "source_freeze_sha256", "runtime_sha256", "runtime"
    }
    if (
        set(runtime_manifest) != required_runtime
        or runtime_manifest.get("schema") != "pairtoken_1m_confirmation_runtime_v1"
        or runtime_manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or runtime_manifest.get("source_freeze_sha256") != sha256_file(source_freeze_path)
        or not isinstance(runtime_manifest.get("runtime"), dict)
    ):
        raise ConfirmationGateError("forecast runtime manifest differs")
    rebuilt_runtime = sha256_bytes(
        json.dumps(
            runtime_manifest["runtime"], sort_keys=True, separators=(",", ":")
        ).encode()
    )
    if runtime_hashes != {rebuilt_runtime} or runtime_manifest["runtime_sha256"] != rebuilt_runtime:
        raise ConfirmationGateError("forecast artifacts and runtime manifest differ")
    manifest_payload = {
        "schema": "pairtoken_1m_confirmation_forecast_manifest_v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "runtime_manifest_sha256": sha256_file(runtime_path),
        "entries": rows,
    }
    manifest_bytes = canonical_json(manifest_payload)
    manifest_path = Path(output).with_name("forecast_manifest.json")
    atomic_create(manifest_path, manifest_bytes)
    receipt = {
        "schema": SEAL_SCHEMA,
        "status": "all_four_refits_both_arms_all_seeds_and_seed_means_sealed",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "forecast_manifest_path": manifest_path.name,
        "forecast_manifest_sha256": sha256_bytes(manifest_bytes),
        "runtime_manifest_path": str(runtime_path.resolve().relative_to(Path(root).resolve())),
        "runtime_manifest_sha256": sha256_file(runtime_path),
        "forecast_artifact_count": len(rows),
        "individual_forecast_count": 24,
        "seed_mean_forecast_count": 8,
        "folds": [row["name"] for row in contract["refits"]],
        "representations": list(REPRESENTATIONS),
        "fixed_seeds": list(SEEDS),
        "confirmation_interval_start": contract["confirmation_interval"]["start_inclusive_utc"],
        "confirmation_interval_end_exclusive": contract["confirmation_interval"]["end_exclusive_utc"],
        "confirmation_metrics_computed": False,
        "evaluation_payload_opened": False,
    }
    atomic_json(output, receipt)
    return receipt


def verify_all_forecast_seal(
    root: Path,
    *,
    source_freeze_path: Path,
    seal_path: Path,
) -> dict[str, Any]:
    contract = load_contract()
    freeze = verify_source_freeze(source_freeze_path)
    seal = load_json(seal_path, label="all-fold forecast seal")
    required = {
        "schema", "status", "contract_sha256", "source_freeze_sha256",
        "source_bundle_sha256", "forecast_manifest_path", "forecast_manifest_sha256",
        "runtime_manifest_path", "runtime_manifest_sha256",
        "forecast_artifact_count", "individual_forecast_count", "seed_mean_forecast_count",
        "folds", "representations", "fixed_seeds", "confirmation_interval_start",
        "confirmation_interval_end_exclusive", "confirmation_metrics_computed",
        "evaluation_payload_opened",
    }
    if set(seal) != required or seal.get("schema") != SEAL_SCHEMA:
        raise ConfirmationGateError("all-fold seal schema differs")
    expected = {
        "status": "all_four_refits_both_arms_all_seeds_and_seed_means_sealed",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "runtime_manifest_path": "runtime_manifest.json",
        "forecast_artifact_count": 32,
        "individual_forecast_count": 24,
        "seed_mean_forecast_count": 8,
        "folds": [row["name"] for row in contract["refits"]],
        "representations": list(REPRESENTATIONS),
        "fixed_seeds": list(SEEDS),
        "confirmation_interval_start": "2025-08-01T00:00:00Z",
        "confirmation_interval_end_exclusive": "2026-06-30T23:00:00Z",
        "confirmation_metrics_computed": False,
        "evaluation_payload_opened": False,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise ConfirmationGateError(f"all-fold seal differs for {key}")
    manifest_path = Path(seal_path).parent / str(seal["forecast_manifest_path"])
    if not manifest_path.is_file() or sha256_file(manifest_path) != seal["forecast_manifest_sha256"]:
        raise ConfirmationGateError("sealed forecast manifest differs")
    manifest = load_json(manifest_path, label="sealed forecast manifest")
    if manifest.get("schema") != "pairtoken_1m_confirmation_forecast_manifest_v1":
        raise ConfirmationGateError("sealed forecast-manifest schema differs")
    runtime_path = (Path(root).resolve() / str(seal["runtime_manifest_path"])).resolve()
    try:
        runtime_path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ConfirmationGateError("sealed runtime manifest escapes forecast root") from exc
    if (
        not runtime_path.is_file()
        or sha256_file(runtime_path) != seal["runtime_manifest_sha256"]
        or manifest.get("runtime_manifest_sha256") != seal["runtime_manifest_sha256"]
    ):
        raise ConfirmationGateError("sealed runtime manifest differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 32:
        raise ConfirmationGateError("sealed forecast manifest is incomplete")
    for spec, recorded in zip(expected_specs(contract), entries, strict=True):
        current = verify_registry_entry(
            root, spec, source_freeze_path=source_freeze_path, contract=contract
        )
        if current != recorded:
            raise ConfirmationGateError("registry changed after all-fold seal")
    return seal


__all__ = [
    "AUXILIARY_DIMENSION", "AUTHORIZATION_SCHEMA", "ConfirmationGateError",
    "CONTRACT_PATH", "FORECAST_SCHEMA", "ForecastSpec", "MODEL_IDS", "PAIR_COUNT",
    "PATH_DIMENSION", "RAW_DIMENSION", "REPRESENTATIONS", "SEEDS", "SOURCE_FREEZE_SCHEMA",
    "atomic_create", "atomic_json", "axis_sha256", "canonical_json",
    "create_source_freeze", "deterministic_npz_bytes", "expected_artifact_path",
    "expected_registry_path", "expected_specs", "fold_contract", "hourly_clock_ns",
    "load_contract", "load_json", "pair_ids", "read_forecast_artifact",
    "rank_pair_targets", "register_forecast", "seal_all_forecasts", "sha256_bytes", "sha256_file",
    "verify_all_forecast_seal", "verify_source_freeze", "write_forecast_artifact",
]
