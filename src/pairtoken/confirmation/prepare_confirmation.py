#!/usr/bin/env python3
"""Build blinded N02/C02 fold inputs and a separately keyed outcome payload.

This adapter extends the frozen one-minute economic/component semantics to the
confirmation clock without importing either development CLI.  It reads only
the exact hash-verified March-2024--June-2026 acquisition allowlist.  Forecast
inputs are factorized (roughly one order of magnitude smaller than dense raw
tokens) and structurally omit confirmation targets.  The evaluation payload
contains outcomes but no forecasts and is written to a separately supplied
directory.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from confirmation_governance import (
    CONTRACT_PATH,
    ConfirmationGateError,
    atomic_json,
    axis_sha256,
    fold_contract,
    hourly_clock_ns,
    load_contract,
    pair_ids,
    sha256_file,
    verify_source_freeze,
)
from train_confirmation import _expected_partition_clocks


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_AUDITED_ROOT = (
    WORKSPACE / "dependencies" / "pairs_research"
)
AUDITED_HASHES = {
    "binance_vision.py": "fcb5b20886e9dbcf7fd9a7880ce1e7f3aca4e7653587a3329d53a09f0c40d3de",
    "binance_um_data.py": "d45467fe331712b34ea6b0a34cc5de371b3bb4740ddabef24a1858169ce33717",
    "crypto_multicoin_execution.py": "4a24afe9f5e24c8186325dbdb30b02f67214590ad396c0cc07a134b61f9f5655",
    "crypto_multicoin_pipeline.py": "db05fadaa330a2a19d172600b4dc6e95dbf045b7b5588d8d6aee915f46d391e5",
    "crypto_multicoin_data.py": "8d4904c324425853bdf574e2cd50916b100d5851e0d7d223809d55984455a2b6",
}
BLINDED_SCHEMA = "pairtoken_1m_confirmation_blinded_fold_v1"
EVALUATION_SCHEMA = "pairtoken_1m_confirmation_evaluation_payload_v1"
EVALUATION_MANIFEST_SCHEMA = "pairtoken_1m_confirmation_evaluation_payload_manifest_v1"
MONTHS = tuple(str(period) for period in pd.period_range("2024-03", "2026-06", freq="M"))
HOUR_NS = 3_600_000_000_000
ENDPOINT_HISTORY = 168
MINUTES_PER_HOUR = 60
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Dependencies:
    vision: Any
    um: Any
    pipeline: Any
    execution: Any
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class Market:
    minute_closes: pd.DataFrame
    hourly_opens: pd.DataFrame
    hourly_last_closes: pd.DataFrame
    previous_minute_available: pd.DataFrame
    funding_at_boundary: pd.DataFrame
    funding_verified_archives: int
    funding_missing_archives: int


@dataclass(frozen=True)
class Components:
    clock: pd.DatetimeIndex
    symbols: tuple[str, ...]
    pair_ids: tuple[str, ...]
    legs: np.ndarray
    coin_minute_returns: np.ndarray
    pair_weights: np.ndarray
    pair_endpoints: np.ndarray
    feature_available: np.ndarray
    target_cashflows: np.ndarray
    target_available: np.ndarray
    coin_open_prices: np.ndarray
    coin_returns: np.ndarray
    funding_boundary: np.ndarray
    coin_execution_available: np.ndarray
    pair_decision_available: np.ndarray
    terminal_exit: np.ndarray


def audited_dependencies(root: Path) -> Dependencies:
    source = Path(root).resolve() / "src/pairs_research"
    for name, digest in AUDITED_HASHES.items():
        path = source / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ConfirmationGateError(f"audited dependency differs: {name}")
    if str(source.parent) not in sys.path:
        sys.path.insert(0, str(source.parent))
    vision = importlib.import_module("pairs_research.binance_vision")
    um = importlib.import_module("pairs_research.binance_um_data")
    pipeline = importlib.import_module("pairs_research.crypto_multicoin_pipeline")
    execution = importlib.import_module("pairs_research.crypto_multicoin_execution")
    for name, module in {
        "binance_vision.py": vision,
        "binance_um_data.py": um,
        "crypto_multicoin_pipeline.py": pipeline,
        "crypto_multicoin_execution.py": execution,
    }.items():
        if Path(module.__file__).resolve() != (source / name).resolve():
            raise ConfirmationGateError(f"audited import escaped snapshot: {name}")
    return Dependencies(vision, um, pipeline, execution, dict(AUDITED_HASHES))


def verify_acquisition(directory: Path, source_freeze_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    root = Path(directory).resolve()
    required = (
        "acquisition_manifest.json", "requested_inventory.csv", "archive_availability.csv",
        "archive_manifest.csv", "development_overlap_receipt.json",
        "artifact_hashes.json", "artifact_hashes.sha256",
    )
    if any(not (root / name).is_file() for name in required):
        raise ConfirmationGateError("acquisition receipt is incomplete")
    receipt = (root / "artifact_hashes.sha256").read_text(encoding="ascii").split()
    if len(receipt) != 2 or receipt[1] != "artifact_hashes.json":
        raise ConfirmationGateError("acquisition hash receipt is invalid")
    if sha256_file(root / "artifact_hashes.json") != receipt[0]:
        raise ConfirmationGateError("acquisition hash receipt differs")
    hashes = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    if hashes.get("schema") != "pairtoken_1m_confirmation_acquisition_v1":
        raise ConfirmationGateError("acquisition hash schema differs")
    files = hashes.get("files")
    payload_names = set(required[:5])
    if not isinstance(files, dict) or set(files) != payload_names:
        raise ConfirmationGateError("acquisition hashed-file family differs")
    for name, digest in files.items():
        if SHA_RE.fullmatch(str(digest)) is None or sha256_file(root / name) != digest:
            raise ConfirmationGateError(f"acquisition file differs: {name}")
    manifest = json.loads((root / "acquisition_manifest.json").read_text(encoding="utf-8"))
    freeze = verify_source_freeze(source_freeze_path)
    expected = {
        "schema": "pairtoken_1m_confirmation_acquisition_v1",
        "status": "complete_exact_allowlist_archives_hash_verified_no_csv_parsed",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "source_bundle_sha256": freeze["source_bundle_sha256"],
        "first_month": "2024-03",
        "last_month": "2026-06",
        "request_count": 1344,
        "archive_csv_bodies_parsed": False,
        "july_2026_or_later_path_count": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ConfirmationGateError(f"acquisition manifest differs for {key}")
    overlap_path = root / "development_overlap_receipt.json"
    if manifest.get("development_overlap_receipt_sha256") != sha256_file(overlap_path):
        raise ConfirmationGateError("acquisition is not bound to development overlap parity")
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if (
        overlap.get("status") != "passed_exact_parent_parity_before_confirmation_firewall"
        or overlap.get("source_freeze_sha256") != sha256_file(source_freeze_path)
        or overlap.get("confirmation_rows_parsed") is not False
    ):
        raise ConfirmationGateError("development overlap receipt differs at preparation")
    rows = pd.read_csv(root / "archive_availability.csv", dtype=str, keep_default_na=False)
    expected_columns = {
        "ordinal", "liquidity_rank", "symbol", "month", "dataset", "interval",
        "availability", "cache_disposition", "archive_relative_path", "archive_sha256",
        "official_checksum_sha256", "checksum_relative_path", "checksum_sidecar_sha256",
        "source_url",
    }
    if set(rows.columns) != expected_columns or len(rows) != 1344:
        raise ConfirmationGateError("acquisition availability table differs")
    if rows.duplicated(["symbol", "month", "dataset"]).any():
        raise ConfirmationGateError("acquisition keys are duplicated")
    return rows, {
        "acquisition_manifest_sha256": sha256_file(root / "acquisition_manifest.json"),
        "archive_availability_sha256": sha256_file(root / "archive_availability.csv"),
        "artifact_hashes_sha256": sha256_file(root / "artifact_hashes.json"),
    }


def _verified_archive(row: Any, cache_root: Path, dependencies: Dependencies) -> Any:
    relative = dependencies.vision.archive_relative_path(
        str(row.symbol), str(row.month), dataset=str(row.dataset), interval="1m"
    )
    if relative.as_posix() != str(row.archive_relative_path):
        raise ConfirmationGateError("archive relative path differs")
    path = Path(cache_root).resolve() / relative
    sidecar = path.with_name(f"{path.name}.CHECKSUM")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    official = dependencies.vision.parse_checksum(
        sidecar.read_text(encoding="ascii"), expected_filename=path.name
    )
    if (
        observed != str(row.archive_sha256)
        or observed != str(row.official_checksum_sha256)
        or observed != official
        or sha256_file(sidecar) != str(row.checksum_sidecar_sha256)
    ):
        raise ConfirmationGateError("archive/checksum binding differs")
    return dependencies.vision.VerifiedArchive(
        dataset=str(row.dataset), symbol=str(row.symbol), month=str(row.month),
        path=path, sha256=observed, source_url=str(row.source_url),
    )


def load_market(
    *,
    cache_root: Path,
    acquisition_rows: pd.DataFrame,
    symbols: tuple[str, ...],
    dependencies: Dependencies,
) -> Market:
    minute_grid = pd.date_range(
        "2024-03-01T00:00:00Z", "2026-06-30T23:59:00Z", freq="1min"
    )
    hour_grid = pd.date_range(
        "2024-03-01T00:00:00Z", "2026-06-30T23:00:00Z", freq="1h"
    )
    close_columns: dict[str, pd.Series] = {}
    open_columns: dict[str, pd.Series] = {}
    last_close_columns: dict[str, pd.Series] = {}
    previous_columns: dict[str, pd.Series] = {}
    funding_blocks: dict[str, list[tuple[str, pd.Series | None]]] = {}
    funding_verified = 0
    funding_missing = 0
    lookup = {
        (str(row.symbol), str(row.month), str(row.dataset)): row
        for row in acquisition_rows.itertuples(index=False)
    }
    expected_keys = {
        (symbol, month, dataset)
        for symbol in symbols for month in MONTHS for dataset in ("klines", "fundingRate")
    }
    if set(lookup) != expected_keys:
        raise ConfirmationGateError("acquisition table does not match exact allowlist")
    for symbol in symbols:
        kline_frames: list[pd.DataFrame] = []
        funding_for_symbol: list[tuple[str, pd.Series | None]] = []
        for month in MONTHS:
            kline_row = lookup[(symbol, month, "klines")]
            if kline_row.availability == "available_verified":
                archive = _verified_archive(kline_row, cache_root, dependencies)
                normalized, _audit = dependencies.um.normalize_kline_frame(
                    dependencies.vision.read_single_csv(archive), symbol=symbol, interval="1m"
                )
                month_start = pd.Timestamp(f"{month}-01T00:00:00Z")
                month_stop = month_start + pd.offsets.MonthBegin(1)
                if normalized.index.min() < month_start or normalized.index.max() >= month_stop:
                    raise ConfirmationGateError("kline archive contains another month")
                kline_frames.append(normalized)
            elif kline_row.availability != "missing_remote_404":
                raise ConfirmationGateError("unsupported kline availability")
            funding_row = lookup[(symbol, month, "fundingRate")]
            if funding_row.availability == "available_verified":
                archive = _verified_archive(funding_row, cache_root, dependencies)
                values = dependencies.um.normalize_funding_frame(
                    dependencies.vision.read_single_csv(archive), symbol=symbol
                )
                month_start = pd.Timestamp(f"{month}-01T00:00:00Z")
                month_stop = month_start + pd.offsets.MonthBegin(1)
                if len(values) and (values.index.min() < month_start or values.index.max() >= month_stop):
                    raise ConfirmationGateError("funding archive contains another month")
                funding_for_symbol.append((month, values))
                funding_verified += 1
            elif funding_row.availability == "missing_remote_404":
                funding_for_symbol.append((month, None))
                funding_missing += 1
            else:
                raise ConfirmationGateError("unsupported funding availability")
        if kline_frames:
            full = pd.concat(kline_frames).sort_index()
            if full.index.duplicated().any():
                raise ConfirmationGateError("kline months overlap")
            close_columns[symbol] = full["close"].astype(float).reindex(minute_grid)
            open_columns[symbol] = full.loc[full.index.minute == 0, "open"].astype(float).reindex(hour_grid)
            last_close_columns[symbol] = full["close"].astype(float).resample(
                "1h", label="left", closed="left"
            ).last().reindex(hour_grid)
            observed = full.index
            previous_columns[symbol] = pd.Series(
                (hour_grid - pd.Timedelta(minutes=1)).isin(observed), index=hour_grid, dtype=bool
            )
        else:
            close_columns[symbol] = pd.Series(np.nan, index=minute_grid)
            open_columns[symbol] = pd.Series(np.nan, index=hour_grid)
            last_close_columns[symbol] = pd.Series(np.nan, index=hour_grid)
            previous_columns[symbol] = pd.Series(False, index=hour_grid, dtype=bool)
        funding_blocks[symbol] = funding_for_symbol
    all_funding_times = sorted({
        pd.Timestamp(timestamp)
        for blocks in funding_blocks.values() for _month, series in blocks
        if series is not None for timestamp in series.index
        if pd.Timestamp(timestamp) <= hour_grid[-1]
    })
    event_clock = pd.DatetimeIndex(all_funding_times, tz="UTC")
    funding = pd.DataFrame(0.0, index=hour_grid, columns=symbols)
    for symbol, blocks in funding_blocks.items():
        for month, series in blocks:
            start = pd.Timestamp(f"{month}-01T00:00:00Z")
            stop = start + pd.offsets.MonthBegin(1)
            events = event_clock[(event_clock >= start) & (event_clock < stop)]
            if series is None:
                funding.loc[events, symbol] = np.nan
            else:
                funding.loc[events, symbol] = series.reindex(events).fillna(0.0).to_numpy()
    return Market(
        minute_closes=pd.DataFrame(close_columns, index=minute_grid),
        hourly_opens=pd.DataFrame(open_columns, index=hour_grid),
        hourly_last_closes=pd.DataFrame(last_close_columns, index=hour_grid),
        previous_minute_available=pd.DataFrame(previous_columns, index=hour_grid).astype(bool),
        funding_at_boundary=funding,
        funding_verified_archives=funding_verified,
        funding_missing_archives=funding_missing,
    )


def rolling_pair_weights(
    returns: pd.DataFrame, legs: np.ndarray, *, lookback: int = 720, minimum: int = 168
) -> np.ndarray:
    log_returns = np.log1p(pd.DataFrame(returns, dtype=float).where(returns > -1.0)).shift(1)
    result = np.full((len(returns), len(legs), 2), np.nan, dtype=np.float64)
    for pair, (signal, hedge) in enumerate(legs):
        y = log_returns.iloc[:, int(signal)]
        x = log_returns.iloc[:, int(hedge)]
        covariance = pd.concat({"x": x, "y": y}, axis=1).rolling(
            lookback, min_periods=minimum
        ).cov().unstack().loc[:, ("x", "y")]
        variance = x.rolling(lookback, min_periods=minimum).var(ddof=1)
        beta = covariance.div(variance.where(variance > 1e-12)).clip(-3.0, 3.0)
        scale = 1.0 + beta.abs()
        result[:, pair, 0] = (1.0 / scale).to_numpy()
        result[:, pair, 1] = (-beta / scale).to_numpy()
    return result


def build_components(market: Market, contract: Mapping[str, Any]) -> Components:
    symbols = tuple(contract["data"]["symbols_in_order"])
    pairs = pair_ids(symbols)
    legs = np.asarray([
        (second, first)
        for first in range(len(symbols)) for second in range(first + 1, len(symbols))
    ], dtype=np.int64)
    clock = market.hourly_opens.index
    opens = market.hourly_opens.loc[:, list(symbols)].astype(float)
    next_opens = opens.shift(-1)
    terminal_marks = market.hourly_last_closes.loc[:, list(symbols)]
    decision = (
        market.previous_minute_available.loc[:, list(symbols)]
        & opens.notna() & opens.gt(0.0)
    )
    terminal_exit = pd.DataFrame(False, index=clock, columns=symbols)
    for symbol in symbols:
        valid = opens[symbol].dropna()
        if valid.empty:
            continue
        final = valid.index[-1]
        if final < clock[-1]:
            terminal_exit.loc[final, symbol] = bool(
                pd.isna(next_opens.loc[final, symbol])
                and pd.notna(terminal_marks.loc[final, symbol])
            )
    exits = next_opens.where(next_opens.notna(), terminal_marks.where(terminal_exit))
    simple = exits.div(opens).sub(1.0).where(decision)
    funding = market.funding_at_boundary.loc[:, list(symbols)].copy()
    prior_terminal = terminal_exit.shift(1, fill_value=False)
    funding = funding.mask(prior_terminal, 0.0)
    funding_next = funding.shift(-1).mask(terminal_exit, 0.0)
    growth = 1.0 + simple
    coin_target = decision & simple.notna() & funding_next.notna() & growth.gt(0.0)
    # The final 23:00 UTC row is a liquidation boundary, never an outcome row.
    coin_target.iloc[-1] = False
    simple.iloc[-1] = np.nan
    weights = rolling_pair_weights(simple, legs)
    decision_array = decision.to_numpy(dtype=bool)
    target_array = coin_target.to_numpy(dtype=bool)
    return_array = simple.to_numpy(dtype=float)
    growth_array = growth.to_numpy(dtype=float)
    funding_next_array = funding_next.to_numpy(dtype=float)
    pair_decision = np.zeros((len(clock), len(pairs)), dtype=bool)
    pair_target = np.zeros_like(pair_decision)
    cashflows = np.zeros((len(clock), len(pairs)), dtype=np.float64)
    for pair, (signal, hedge) in enumerate(legs):
        finite_weight = np.isfinite(weights[:, pair]).all(axis=1)
        pair_decision[:, pair] = finite_weight & decision_array[:, signal] & decision_array[:, hedge]
        pair_target[:, pair] = pair_decision[:, pair] & target_array[:, signal] & target_array[:, hedge]
        price = (
            weights[:, pair, 0] * return_array[:, signal]
            + weights[:, pair, 1] * return_array[:, hedge]
        )
        next_funding = -(
            weights[:, pair, 0] * growth_array[:, signal] * funding_next_array[:, signal]
            + weights[:, pair, 1] * growth_array[:, hedge] * funding_next_array[:, hedge]
        )
        value = price + next_funding
        cashflows[:, pair] = np.where(pair_target[:, pair], value, 0.0)
    if not np.isfinite(cashflows).all():
        raise ConfirmationGateError("masked target cashflows are nonfinite")
    # Confirmation outcomes must be complete wherever a model can trade.
    test = (clock >= pd.Timestamp("2025-08-01T00:00:00Z")) & (
        clock < pd.Timestamp("2026-06-30T23:00:00Z")
    )
    incomplete = decision.to_numpy()[test] & ~coin_target.to_numpy()[test]
    if incomplete.any():
        timestamp, coin = np.argwhere(incomplete)[0]
        raise ConfirmationGateError(
            f"unmeasurable confirmation outcome at {clock[test][timestamp]} {symbols[coin]}"
        )

    minute_grid = market.minute_closes.index
    differencing = minute_grid.insert(0, minute_grid[0] - pd.Timedelta(minutes=1))
    dense = market.minute_closes.reindex(differencing)
    minute_returns = np.log(dense.where(dense > 0.0)).diff().reindex(minute_grid)
    cube = minute_returns.to_numpy(dtype=float).reshape(-1, 60, len(symbols))
    complete = np.isfinite(cube).all(axis=1)
    hourly_coin = np.where(complete, np.nansum(cube, axis=1), np.nan)
    clean_cube = np.where(np.isfinite(cube), cube, 0.0).transpose(0, 2, 1).astype(np.float32)
    endpoints = np.zeros((len(clock), len(pairs)), dtype=np.float32)
    path_available = np.zeros_like(pair_decision)
    for pair, (signal, hedge) in enumerate(legs):
        available = (
            np.isfinite(hourly_coin[:, signal])
            & np.isfinite(hourly_coin[:, hedge])
            & np.isfinite(weights[:, pair]).all(axis=1)
            & pair_decision[:, pair]
        )
        value = weights[:, pair, 0] * hourly_coin[:, signal] + weights[:, pair, 1] * hourly_coin[:, hedge]
        endpoints[available, pair] = value[available].astype(np.float32)
        path_available[:, pair] = available
    prefix = np.vstack((
        np.zeros((1, len(pairs)), dtype=np.int64),
        np.cumsum(~path_available, axis=0, dtype=np.int64),
    ))
    feature = np.zeros_like(pair_decision)
    rows = np.arange(ENDPOINT_HISTORY, len(clock))
    feature[rows] = (
        (prefix[rows] - prefix[rows - ENDPOINT_HISTORY] == 0)
        & pair_decision[rows]
    )
    return Components(
        clock, symbols, pairs, legs, clean_cube, weights.astype(np.float64), endpoints,
        feature, cashflows.astype(np.float64), pair_target,
        opens.to_numpy(dtype=np.float64), return_array,
        funding.to_numpy(dtype=np.float64), decision_array, pair_decision,
        terminal_exit.to_numpy(dtype=bool),
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _array_family_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def write_blinded_inputs(
    components: Components,
    *,
    output_root: Path,
    source_freeze_path: Path,
    acquisition_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    contract = load_contract()
    freeze = verify_source_freeze(source_freeze_path)
    lookup = pd.Series(np.arange(len(components.clock)), index=components.clock)
    receipts: list[dict[str, Any]] = []
    for row in contract["refits"]:
        train_ns, validation_ns, test_ns = _expected_partition_clocks(row)
        train_clock = pd.to_datetime(train_ns, utc=True)
        validation_clock = pd.to_datetime(validation_ns, utc=True)
        test_clock = pd.to_datetime(test_ns, utc=True)
        locations = (
            lookup.reindex(train_clock),
            lookup.reindex(validation_clock),
            lookup.reindex(test_clock),
        )
        if any(value.isna().any() for value in locations):
            raise ConfirmationGateError(f"component clock does not cover {row['name']}")
        selected = np.concatenate(tuple(
            value.to_numpy(dtype=np.int64) for value in locations
        ))
        source_first = int(selected.min()) - ENDPOINT_HISTORY
        source_last = int(selected.max())
        if source_first < 1:
            raise ConfirmationGateError("factorized fold lacks feature warm-up")
        source_rows = np.arange(source_first, source_last + 1, dtype=np.int64)
        relative_lookup = {int(global_row): local for local, global_row in enumerate(source_rows)}
        train_rows = np.asarray([relative_lookup[int(value)] for value in selected[:len(train_ns)]], dtype=np.int64)
        validation_rows = np.asarray([
            relative_lookup[int(value)] for value in selected[len(train_ns):len(train_ns)+len(validation_ns)]
        ], dtype=np.int64)
        test_rows = np.asarray([relative_lookup[int(value)] for value in selected[-len(test_ns):]], dtype=np.int64)
        proof = {
            "source_start": components.clock[source_rows[0]].isoformat(),
            "source_end": components.clock[source_rows[-1]].isoformat(),
            "training_start": train_clock[0].isoformat(),
            "training_end_inclusive": train_clock[-1].isoformat(),
            "validation_start": validation_clock[0].isoformat(),
            "validation_end_inclusive": validation_clock[-1].isoformat(),
            "test_start": test_clock[0].isoformat(),
            "test_end_inclusive": test_clock[-1].isoformat(),
            "endpoint_history_hours": ENDPOINT_HISTORY,
            "hedge_history_hours": 720,
            "risk_history_days": 60,
            "all_windows_complete": True,
        }
        shared_arrays = {
            "clock_source_ns": components.clock[source_rows].asi8,
            "pair_ids": np.asarray(components.pair_ids),
            "pair_leg_indices": components.legs,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
            "coin_minute_returns": components.coin_minute_returns[source_rows],
            "pair_weights": components.pair_weights[source_rows],
            "pair_path_endpoints": components.pair_endpoints[source_rows],
            "feature_available": components.feature_available[source_rows],
            "target_cashflows_train": components.target_cashflows[selected[:len(train_ns)]],
            "target_cashflows_validation": components.target_cashflows[
                selected[len(train_ns):len(train_ns)+len(validation_ns)]
            ],
            "target_available_train": components.target_available[selected[:len(train_ns)]],
            "target_available_validation": components.target_available[
                selected[len(train_ns):len(train_ns)+len(validation_ns)]
            ],
        }
        factorized_hash = _array_family_sha256(shared_arrays)
        for representation in ("native_1m", "causal_15m_control"):
            metadata = {
                "schema": BLINDED_SCHEMA,
                "status": "complete_factorized_train_validation_labels_and_test_features_only",
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "source_freeze_sha256": sha256_file(source_freeze_path),
                "source_bundle_sha256": freeze["source_bundle_sha256"],
                "fold": row["name"],
                "representation": representation,
                "pair_count": 276,
                "raw_dimension": 232,
                "auxiliary_dimension": 84,
                "auxiliary_values": "all_zeros_created_by_trainer",
                "pair_axis_sha256": axis_sha256(components.pair_ids),
                "test_target_labels_included": False,
                "causal_test_history_features_included": True,
                "confirmation_metrics_computed": False,
                "factorized_storage": True,
                "factorized_content_sha256": factorized_hash,
                "warmup_proof": proof,
                **dict(acquisition_hashes),
            }
            path = Path(output_root) / row["name"] / f"{representation}.blinded.npz"
            _write_npz(path, {
                "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
                **shared_arrays,
            })
            receipts.append({
                "fold": row["name"], "representation": representation,
                "path": f"{row['name']}/{representation}.blinded.npz",
                "sha256": sha256_file(path),
            })
    if len(receipts) != 8:
        raise ConfirmationGateError("blinded input family is not exactly eight")
    atomic_json(Path(output_root) / "blinded_input_manifest.json", {
        "schema": "pairtoken_1m_confirmation_blinded_input_manifest_v1",
        "status": "eight_factorized_inputs_without_same_decision_forward_labels",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "entries": receipts,
    })
    return receipts


def _asof_risk(risk: Any, clock: pd.DatetimeIndex, symbols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    beta = risk.beta_frame(clock).reindex(columns=list(symbols)).to_numpy(dtype=float)
    effective = pd.DatetimeIndex(sorted(risk.covariance_matrices))
    locations = effective.searchsorted(clock, side="right") - 1
    if (locations < 0).any():
        raise ConfirmationGateError("risk inputs lack a causal first estimate")
    covariance = np.stack([
        risk.covariance_matrices[effective[position]].reindex(
            index=list(symbols), columns=list(symbols)
        ).to_numpy(dtype=float)
        for position in locations
    ])
    return beta, covariance


def _parent_overlap_modules(source_freeze_path: Path) -> tuple[Any, Any]:
    """Import the hash-pinned parent builders for a development-only shadow run."""

    verify_source_freeze(source_freeze_path)
    parent = HERE.parent / "development"
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    core = importlib.import_module("paired_neural_core")
    evaluator_name = "pairtoken_confirmation_overlap_parent_evaluator"
    if evaluator_name in sys.modules:
        evaluator = sys.modules[evaluator_name]
    else:
        spec = importlib.util.spec_from_file_location(
            evaluator_name, parent / "portfolio_evaluation.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(parent / "portfolio_evaluation.py")
        evaluator = importlib.util.module_from_spec(spec)
        sys.modules[evaluator_name] = evaluator
        spec.loader.exec_module(evaluator)
    frozen = load_contract()["frozen_parent"]
    if (
        Path(core.__file__).resolve() != (parent / "paired_neural_core.py").resolve()
        or sha256_file(Path(core.__file__)) != frozen["paired_core_sha256"]
        or sha256_file(Path(evaluator.__file__)) != frozen["development_evaluator_sha256"]
    ):
        raise ConfirmationGateError("development overlap import escaped the frozen parent")
    return core, evaluator


def _maximum_finite_difference(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ConfirmationGateError("overlap arrays have different shapes or missingness")
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(np.isinf(a)) or np.any(np.isinf(b)):
        raise ConfirmationGateError("overlap arrays contain infinity")
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0


def validate_development_overlap(
    *,
    cache_root: Path,
    development_manifest: Path,
    source_freeze_path: Path,
    audited_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Prove wrapper/parent parity without opening a reserved confirmation row."""

    if Path(output).exists():
        raise FileExistsError(output)
    core, parent = _parent_overlap_modules(source_freeze_path)
    parent_dependencies = parent.install_audited_dependencies(audited_root)
    dependencies = audited_dependencies(audited_root)
    if dict(parent_dependencies.source_hashes) != dict(dependencies.hashes):
        raise ConfirmationGateError("overlap dependency hashes differ")
    symbols, pairs, legs = parent.static_axes()
    start = pd.Timestamp("2024-03-01T00:00:00Z")
    end = pd.Timestamp("2025-06-30T23:00:00Z")
    if end >= pd.Timestamp("2025-07-01T00:00:00Z"):
        raise ConfirmationGateError("overlap validation reaches the confirmation firewall")
    parent_market = parent.load_market_inputs(
        cache_root=cache_root,
        manifest_directory=development_manifest,
        symbols=symbols,
        start=start,
        end_inclusive=end,
        dependencies=parent_dependencies,
    )
    closes = core.load_development_closes(
        cache_root,
        manifest_directory=development_manifest,
        symbols=symbols,
        start=start,
        end_exclusive=end + pd.Timedelta(hours=1),
    )
    parent_contract = copy.deepcopy(parent.load_contract())
    parent_contract["folds"] = []
    parent_contract["data"]["transient_open_gap_policy"] = {
        "rule": "no_fill_no_shortening_mask_outcome_only",
        "expected_coin_gap_cells": 0,
        "expected_pair_label_cells_masked": 0,
        "expected_events": [],
        "maximum_oos_coin_gap_cells": 0,
        "maximum_oos_pair_label_cells": 0,
    }
    economic = parent.build_economic_inputs(
        market=parent_market,
        symbols=symbols,
        pair_ids=pairs,
        pair_leg_indices=legs,
        contract=parent_contract,
        provenance={"purpose": "development_only_confirmation_adapter_overlap"},
    )
    clock = parent_market.open_prices.index
    funding = pd.DataFrame(0.0, index=clock, columns=symbols)
    event = clock.isin(parent_market.funding_rates.index)
    funding.loc[event] = parent_market.funding_rates.reindex(clock[event]).to_numpy()
    minute_grid = pd.date_range(start, end + pd.Timedelta(minutes=59), freq="1min")
    wrapper_market = Market(
        minute_closes=closes.reindex(minute_grid),
        hourly_opens=parent_market.open_prices,
        hourly_last_closes=parent_market.last_closes,
        previous_minute_available=parent_market.previous_minute_available,
        funding_at_boundary=funding,
        funding_verified_archives=parent_market.funding_archive_count,
        funding_missing_archives=parent_market.missing_funding_archive_count,
    )
    wrapper = build_components(wrapper_market, load_contract())
    parent_components = core.build_components_from_closes(
        closes.reindex(minute_grid), economic_inputs=economic
    )
    exact_arrays = {
        "clock": np.array_equal(wrapper.clock.asi8, economic.clock.asi8),
        "pair_legs": np.array_equal(wrapper.legs, economic.pair_leg_indices),
        "coin_decision_available": np.array_equal(
            wrapper.coin_execution_available, economic.coin_decision_available
        ),
        "pair_decision_available": np.array_equal(
            wrapper.pair_decision_available, economic.pair_decision_available
        ),
        "pair_target_available": np.array_equal(
            wrapper.target_available, economic.pair_target_available
        ),
        "component_feature_available": np.array_equal(
            wrapper.feature_available, parent_components.feature_available
        ),
        "component_target_available": np.array_equal(
            wrapper.target_available, parent_components.target_available
        ),
    }
    if not all(exact_arrays.values()):
        raise ConfirmationGateError("development overlap masks or axes differ")
    numeric_differences = {
        "pair_weights": _maximum_finite_difference(wrapper.pair_weights, economic.pair_weights),
        "target_cashflows": _maximum_finite_difference(
            wrapper.target_cashflows, economic.pair_target_cashflows
        ),
        "pair_endpoints": _maximum_finite_difference(
            wrapper.pair_endpoints, parent_components.pair_path_endpoints
        ),
    }
    if any(value != 0.0 for value in numeric_differences.values()):
        raise ConfirmationGateError("development overlap economic/component values differ")

    candidate = np.flatnonzero(wrapper.feature_available.any(axis=1) & (np.arange(len(clock)) >= 168))
    if len(candidate) < 16:
        raise ConfirmationGateError("development overlap lacks feature rows")
    selected = np.unique(np.linspace(candidate[0], candidate[-1], 16, dtype=np.int64))
    feature_differences: dict[str, float] = {}
    for representation in ("native_1m", "causal_15m_control"):
        expected_raw = parent_components.features(selected, representation=representation)
        observed_raw, observed_mask = __import__(
            "train_confirmation"
        ).materialize_factorized_rows(
            source_clock=wrapper.clock.asi8,
            selected=selected,
            legs=wrapper.legs,
            coin_minute=wrapper.coin_minute_returns,
            weights=wrapper.pair_weights,
            endpoints=wrapper.pair_endpoints,
            source_feature=wrapper.feature_available,
            representation=representation,
        )
        if not np.array_equal(observed_mask, parent_components.feature_available[selected]):
            raise ConfirmationGateError("development overlap feature mask differs")
        feature_differences[representation] = _maximum_finite_difference(
            observed_raw, expected_raw
        )
    if any(value != 0.0 for value in feature_differences.values()):
        raise ConfirmationGateError("development overlap N02/C02 raw features differ")

    ledger_clock = clock[-169:]
    risk = dependencies.pipeline.estimate_daily_risk_inputs(
        parent_market.open_prices,
        start=ledger_clock[0],
        end_exclusive=ledger_clock[-1] + pd.Timedelta(hours=1),
        lookback_days=60,
        minimum_complete_observations=720,
    )
    beta, covariance = _asof_risk(risk, ledger_clock, symbols)
    parent_beta = risk.beta_frame(ledger_clock).reindex(columns=list(symbols)).to_numpy(float)
    parent_covariance = parent._asof_covariance(
        risk.covariance_matrices, ledger_clock, symbols
    )
    risk_differences = {
        "beta": _maximum_finite_difference(beta, parent_beta),
        "covariance": _maximum_finite_difference(covariance, parent_covariance),
    }
    if any(value != 0.0 for value in risk_differences.values()):
        raise ConfirmationGateError("development overlap risk tensors differ")

    locations = pd.Series(np.arange(len(clock)), index=clock).reindex(ledger_clock).to_numpy(int)
    target = np.zeros((len(ledger_clock), len(pairs)), dtype=float)
    for row in range(len(ledger_clock) - 1):
        eligible = np.flatnonzero(wrapper.feature_available[locations[row]])
        if len(eligible) >= 16:
            target[row, eligible[:8]] = -0.05
            target[row, eligible[-8:]] = 0.05
    target_frame = pd.DataFrame(target, index=ledger_clock, columns=pairs)

    def hedge_tensor(weights: np.ndarray) -> np.ndarray:
        result = np.zeros((len(ledger_clock), len(symbols), len(pairs)), dtype=float)
        selected_weights = weights[locations]
        for pair, (signal, hedge) in enumerate(legs):
            result[:, signal, pair] = np.nan_to_num(selected_weights[:, pair, 0])
            result[:, hedge, pair] = np.nan_to_num(selected_weights[:, pair, 1])
        return result

    returns = wrapper.coin_returns[locations].copy()
    returns[-1] = 0.0
    returns_frame = pd.DataFrame(returns, index=ledger_clock, columns=symbols)
    funding_frame = pd.DataFrame(
        wrapper.funding_boundary[locations], index=ledger_clock, columns=symbols
    )
    availability_frame = pd.DataFrame(
        wrapper.coin_execution_available[locations], index=ledger_clock, columns=symbols
    )
    constraints = dependencies.execution.PortfolioConstraints(
        maximum_total_gross=1.0,
        maximum_absolute_coin_weight=0.15,
        maximum_absolute_dollar_net=0.02,
        maximum_absolute_btc_beta=0.05,
        target_annualized_volatility=0.1,
    )
    ledgers = []
    for weights in (wrapper.pair_weights, economic.pair_weights):
        ledgers.append(dependencies.execution.backtest_multicoin_pair_ledger(
            target_frame,
            hedge_tensor(weights),
            returns_frame,
            funding_frame,
            transaction_cost_bps=0.5,
            coin_symbols=symbols,
            availability=availability_frame,
            constraints=constraints,
            btc_beta_loadings=beta,
            covariance_matrices=covariance,
            liquidate_at_end=False,
        ))
    ledger_columns = list(ledgers[0].bars.columns)
    ledger_difference = _maximum_finite_difference(
        ledgers[0].bars[ledger_columns].to_numpy(float),
        ledgers[1].bars[ledger_columns].to_numpy(float),
    )
    if ledger_difference != 0.0:
        raise ConfirmationGateError("development overlap ledger differs")
    receipt = {
        "schema": "pairtoken_1m_confirmation_development_overlap_v1",
        "status": "passed_exact_parent_parity_before_confirmation_firewall",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "development_start": start.isoformat(),
        "development_end_inclusive": end.isoformat(),
        "confirmation_rows_parsed": False,
        "exact_masks": exact_arrays,
        "maximum_absolute_differences": {
            **numeric_differences, **feature_differences, **risk_differences,
            "ledger": ledger_difference,
        },
        "feature_rows_compared": int(len(selected)),
        "ledger_hours_compared": int(len(ledger_clock)),
        "audited_source_hashes": dict(dependencies.hashes),
    }
    atomic_json(output, receipt)
    return receipt


def write_evaluation_payload(
    components: Components,
    market: Market,
    *,
    output_directory: Path,
    source_freeze_path: Path,
    acquisition_hashes: Mapping[str, str],
    dependencies: Dependencies,
    blinded_receipts: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    decision_clock = pd.date_range(
        "2025-08-01T00:00:00Z", "2026-06-30T22:00:00Z", freq="1h"
    )
    terminal = pd.Timestamp("2026-06-30T23:00:00Z")
    ledger_clock = decision_clock.append(pd.DatetimeIndex([terminal]))
    lookup = pd.Series(np.arange(len(components.clock)), index=components.clock)
    rows = lookup.reindex(ledger_clock).to_numpy(dtype=np.int64)
    decision_rows = rows[:-1]
    if len(decision_clock) != 8015 or len(ledger_clock) != 8016:
        raise ConfirmationGateError("evaluation clock differs")
    open_prices = pd.DataFrame(
        components.coin_open_prices, index=components.clock, columns=components.symbols
    )
    risk = dependencies.pipeline.estimate_daily_risk_inputs(
        open_prices,
        start=decision_clock[0],
        end_exclusive=terminal + pd.Timedelta(hours=1),
        lookback_days=60,
        minimum_complete_observations=720,
    )
    beta, covariance = _asof_risk(risk, ledger_clock, components.symbols)
    returns = components.coin_returns[rows].copy()
    returns[-1] = 0.0
    availability = components.coin_execution_available[rows].copy()
    terminal_mask = np.zeros(len(ledger_clock), dtype=bool)
    terminal_mask[-1] = True
    metadata = {
        "schema": EVALUATION_SCHEMA,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "forecast_payload_absent": True,
        "portfolio_metrics_computed": False,
        "decision_rows": 8015,
        "ledger_rows": 8016,
        "terminal_rule": "zero_pair_target_liquidation_at_observed_2026_06_30_23_open",
        **dict(acquisition_hashes),
    }
    payload_path = root / "evaluation_payload.npz"
    _write_npz(payload_path, {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        "decision_clock_ns": decision_clock.asi8,
        "ledger_clock_ns": ledger_clock.asi8,
        "symbols": np.asarray(components.symbols),
        "pair_ids": np.asarray(components.pair_ids),
        "pair_leg_indices": components.legs,
        "pair_weights": components.pair_weights[rows],
        "coin_open_prices": components.coin_open_prices[rows],
        "coin_open_to_open_returns": returns,
        "funding_at_boundary": components.funding_boundary[rows],
        "coin_availability": availability,
        "pair_decision_available": components.pair_decision_available[decision_rows],
        "pair_target_cashflows": components.target_cashflows[decision_rows],
        "pair_target_available": components.target_available[decision_rows],
        "forecast_feature_available": components.feature_available[decision_rows],
        "risk_beta": beta,
        "risk_covariance": covariance,
        "terminal_row_mask": terminal_mask,
    })
    expected_keys = [
        (row["name"], representation)
        for row in load_contract()["refits"]
        for representation in ("native_1m", "causal_15m_control")
    ]
    receipt_lookup = {
        (str(receipt["fold"]), str(receipt["representation"])): receipt
        for receipt in blinded_receipts
    }
    if set(receipt_lookup) != set(expected_keys) or len(blinded_receipts) != 8:
        raise ConfirmationGateError("evaluation manifest lacks the exact blinded-input family")
    blinded_bindings = []
    for fold, representation in expected_keys:
        receipt = receipt_lookup[(fold, representation)]
        digest = str(receipt.get("sha256", ""))
        if SHA_RE.fullmatch(digest) is None:
            raise ConfirmationGateError("blinded-input receipt hash is invalid")
        blinded_bindings.append({
            "fold": fold,
            "representation": representation,
            "relative_path": f"{fold}/{representation}.blinded.npz",
            "sha256": digest,
        })
    manifest = {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "status": "sealed_economic_and_execution_inputs_never_opened_by_forecast_stage",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_freeze_sha256": sha256_file(source_freeze_path),
        "payload_path": payload_path.name,
        "payload_sha256": sha256_file(payload_path),
        "payload_role": "confirmation_outcomes_and_execution_only",
        "clock_start": "2025-08-01T00:00:00Z",
        "clock_end_exclusive": "2026-06-30T23:00:00Z",
        "terminal_rule": "zero_pair_target_liquidation_at_observed_2026_06_30_23_open",
        "symbols": list(components.symbols),
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
    }
    manifest_path = root / "evaluation_payload.manifest.json"
    atomic_json(manifest_path, manifest)
    return payload_path, manifest_path


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--acquisition-manifest", type=Path, required=True)
    prepare.add_argument("--source-freeze", type=Path, required=True)
    prepare.add_argument("--blinded-output", type=Path, required=True)
    prepare.add_argument("--evaluation-output", type=Path, required=True)
    prepare.add_argument("--audited-root", type=Path, default=DEFAULT_AUDITED_ROOT)
    overlap = subparsers.add_parser("validate-overlap")
    overlap.add_argument("--cache-root", type=Path, required=True)
    overlap.add_argument("--development-manifest", type=Path, required=True)
    overlap.add_argument("--source-freeze", type=Path, required=True)
    overlap.add_argument("--audited-root", type=Path, default=DEFAULT_AUDITED_ROOT)
    overlap.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    verify_source_freeze(args.source_freeze)
    if args.command == "validate-overlap":
        receipt = validate_development_overlap(
            cache_root=args.cache_root,
            development_manifest=args.development_manifest,
            source_freeze_path=args.source_freeze,
            audited_root=args.audited_root,
            output=args.output,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    dependencies = audited_dependencies(args.audited_root)
    acquisition_rows, acquisition_hashes = verify_acquisition(
        args.acquisition_manifest, args.source_freeze
    )
    symbols = tuple(load_contract()["data"]["symbols_in_order"])
    market = load_market(
        cache_root=args.cache_root,
        acquisition_rows=acquisition_rows,
        symbols=symbols,
        dependencies=dependencies,
    )
    components = build_components(market, load_contract())
    blinded_destination = Path(args.blinded_output).resolve()
    evaluation_destination = Path(args.evaluation_output).resolve()
    if blinded_destination.exists() or evaluation_destination.exists():
        raise FileExistsError(
            blinded_destination if blinded_destination.exists() else evaluation_destination
        )
    blinded_destination.parent.mkdir(parents=True, exist_ok=True)
    evaluation_destination.parent.mkdir(parents=True, exist_ok=True)
    blinded_container = Path(tempfile.mkdtemp(
        prefix=".pairtoken-blinded-preparation-", dir=blinded_destination.parent
    ))
    evaluation_container = Path(tempfile.mkdtemp(
        prefix=".pairtoken-evaluation-preparation-", dir=evaluation_destination.parent
    ))
    blinded_staging = blinded_container / "payload"
    evaluation_staging = evaluation_container / "payload"
    try:
        receipts = write_blinded_inputs(
            components,
            output_root=blinded_staging,
            source_freeze_path=args.source_freeze,
            acquisition_hashes=acquisition_hashes,
        )
        write_evaluation_payload(
            components,
            market,
            output_directory=evaluation_staging,
            source_freeze_path=args.source_freeze,
            acquisition_hashes=acquisition_hashes,
            dependencies=dependencies,
            blinded_receipts=receipts,
        )
        os.replace(blinded_staging, blinded_destination)
        os.replace(evaluation_staging, evaluation_destination)
    finally:
        shutil.rmtree(blinded_container, ignore_errors=True)
        shutil.rmtree(evaluation_container, ignore_errors=True)
    payload = evaluation_destination / "evaluation_payload.npz"
    manifest = evaluation_destination / "evaluation_payload.manifest.json"
    print(json.dumps({
        "status": "eight_blinded_inputs_and_separate_evaluation_payload_complete",
        "blinded_inputs": len(receipts),
        "evaluation_payload": str(payload),
        "evaluation_manifest": str(manifest),
        "confirmation_metrics_computed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
