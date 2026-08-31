#!/usr/bin/env python3
"""Joint portfolio and risk evaluation for the paired neural forecasts.

The forecast runner intentionally stops at pair targets.  This module binds
those targets to the source-frozen multi-coin execution backend used by the
audited raw-four-hour experiment.  It evaluates the complete five-series,
six-fold and four-cost family together; there is no model or fold selector.

Raw execution prices and funding are opened only through the explicit
development acquisition allowlist.  Every requested archive month is checked
before a path is constructed, and no month after June 2025 is addressable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from paired_neural_core import (
    DEVELOPMENT_END_EXCLUSIVE,
    load_acquisition_manifest,
    load_components,
    require_development_month,
    sha256_file,
)
from paired_neural_models import FIXED_SEEDS, cycle


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
CONTRACT_PATH = HERE / "contract.json"
DEFAULT_AUDITED_ROOT = (
    WORKSPACE / "dependencies" / "pairs_research"
)
AUDITED_SOURCE_HASHES = {
    "crypto_multicoin_execution.py": "4a24afe9f5e24c8186325dbdb30b02f67214590ad396c0cc07a134b61f9f5655",
    "crypto_multicoin_pipeline.py": "db05fadaa330a2a19d172600b4dc6e95dbf045b7b5588d8d6aee915f46d391e5",
    "crypto_multicoin_data.py": "8d4904c324425853bdf574e2cd50916b100d5851e0d7d223809d55984455a2b6",
    "binance_um_data.py": "d45467fe331712b34ea6b0a34cc5de371b3bb4740ddabef24a1858169ce33717",
    "binance_vision.py": "fcb5b20886e9dbcf7fd9a7880ce1e7f3aca4e7653587a3329d53a09f0c40d3de",
}
SERIES = {
    "B00": ("endpoint_ridge_shared", "shared_endpoint", "endpoint_ridge", "none"),
    "N01": ("native_1m_lstm_seed_mean", "native_1m", "lstm_seed_mean", "all3"),
    "C01": (
        "causal_15m_lstm_seed_mean",
        "causal_15m_control",
        "lstm_seed_mean",
        "all3",
    ),
    "N02": (
        "native_1m_itransformer_seed_mean",
        "native_1m",
        "itransformer_seed_mean",
        "all3",
    ),
    "C02": (
        "causal_15m_itransformer_seed_mean",
        "causal_15m_control",
        "itransformer_seed_mean",
        "all3",
    ),
}
FORECAST_DIAGNOSTIC_IDENTITY_KEYS = frozenset(
    {
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
)
FORECAST_RUN_MANIFEST_KEYS = frozenset(
    {
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
)
FORECAST_SOURCE_PATHS = {
    "contract.json": CONTRACT_PATH,
    "paired_neural_core.py": HERE / "paired_neural_core.py",
    "paired_neural_models.py": HERE / "paired_neural_models.py",
    "run_paired_neural.py": HERE / "run_paired_neural.py",
    "parent_cycle_pilot.py": HERE.parent / "model" / "pairtoken_model.py",
    "parent_raw_4h_contract.json": HERE.parent / "contracts" / "universe.json",
}
RUNTIME_ENVIRONMENT_KEYS = frozenset(
    {
        "python_implementation",
        "python_version",
        "numpy_version",
        "pandas_version",
        "torch_version",
        "torch_cuda_version",
        "torch_cudnn_version",
        "requested_device",
        "cuda_available",
        "cuda_device_name",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cublas_workspace_config",
    }
)
CONTRASTS = {
    "primary_itransformer_native_minus_control": ("N02", "C02"),
    "supporting_lstm_native_minus_control": ("N01", "C01"),
}
BOOTSTRAP_DAYS = 7
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 0x13527D7
RISK_LOOKBACK_DAYS = 60
RISK_MINIMUM_COMPLETE_HOURS = 720
TOLERANCE = 1e-10


@dataclass(frozen=True)
class AuditedDependencies:
    PortfolioConstraints: Any
    backtest_multicoin_pair_ledger: Any
    estimate_daily_risk_inputs: Any
    VerifiedArchive: Any
    archive_relative_path: Any
    parse_checksum: Any
    read_single_csv: Any
    normalize_kline_frame: Any
    normalize_funding_frame: Any
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class MarketInputs:
    open_prices: pd.DataFrame
    last_closes: pd.DataFrame
    previous_minute_available: pd.DataFrame
    funding_rates: pd.DataFrame
    manifest_rows: pd.DataFrame
    loaded_months: tuple[str, ...]
    funding_archive_count: int
    missing_funding_archive_count: int


@dataclass(frozen=True)
class Forecast:
    series_id: str
    series: str
    fold: str
    clock: pd.DatetimeIndex
    pair_targets: np.ndarray
    path: Path
    sha256: str


@dataclass(frozen=True)
class EconomicInputs:
    clock: pd.DatetimeIndex
    symbols: tuple[str, ...]
    pair_ids: tuple[str, ...]
    pair_leg_indices: np.ndarray
    pair_weights: np.ndarray
    coin_open_prices: np.ndarray
    coin_open_to_open_returns: np.ndarray
    funding_at_boundary: np.ndarray
    funding_at_next_boundary: np.ndarray
    next_boundary_growth: np.ndarray
    coin_decision_available: np.ndarray
    coin_target_available: np.ndarray
    terminal_exit_mask: np.ndarray
    pair_decision_available: np.ndarray
    pair_target_available: np.ndarray
    pair_target_cashflows: np.ndarray
    metadata: Mapping[str, Any]


ECONOMIC_MEMBERS = {
    "metadata_json",
    "clock_ns",
    "symbols",
    "pair_ids",
    "pair_leg_indices",
    "pair_weights",
    "coin_open_prices",
    "coin_open_to_open_returns",
    "funding_at_boundary",
    "funding_at_next_boundary",
    "next_boundary_growth",
    "coin_decision_available",
    "coin_target_available",
    "terminal_exit_mask",
    "pair_decision_available",
    "pair_target_available",
    "pair_target_cashflows",
}
ECONOMIC_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "artifact_path",
        "artifact_sha256",
        "contract_sha256",
        "clock_start",
        "clock_end",
        "hourly_observations",
        "symbols",
        "pair_ids_sha256",
        "pairs",
        "pair_decision_available_cells",
        "pair_target_available_cells",
        "terminal_exit_cells",
        "funding_verified_archives_loaded",
        "funding_missing_archives",
        "funding_included_in_target",
        "pair_target_formula",
        "one_period_ledger_identity",
        "multi_period_ledger_identity",
        "transient_open_gap_report",
        "audited_source_hashes",
        "acquisition_manifest_sha256",
        "archive_availability_sha256",
        "artifact_hashes_sha256",
        "latest_raw_month",
        "july_or_later_path_count",
        "confirmation_rows_parsed",
    }
)
ECONOMIC_PROVENANCE_KEYS = frozenset(
    {
        "contract_sha256",
        "acquisition_manifest_sha256",
        "archive_availability_sha256",
        "artifact_hashes_sha256",
        "audited_source_hashes",
        "builder_sha256",
    }
)


def _utc(value: object, *, field: str = "timestamp") -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone aware")
    return timestamp.tz_convert("UTC")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
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
    _atomic_bytes(path, _canonical_json(value))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if (
        contract.get("contract_name")
        != "paired_native_1m_vs_causal_15m_neural_development"
    ):
        raise ValueError("paired contract identity differs")
    if tuple(map(float, contract["portfolio"]["cost_bps"])) != (0.0, 0.5, 1.5, 5.0):
        raise ValueError("paired cost family differs")
    if int(contract["candidate_budget"]["maximum_unique_strategy_return_series"]) != 5:
        raise ValueError("paired strategy budget differs")
    if contract["data"]["hard_cutoff_exclusive_utc"] != "2025-07-01T00:00:00Z":
        raise ValueError("paired confirmation firewall differs")
    return contract


def install_audited_dependencies(root: Path) -> AuditedDependencies:
    source = Path(root).expanduser().resolve() / "src/pairs_research"
    observed: dict[str, str] = {}
    for name, expected in AUDITED_SOURCE_HASHES.items():
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"audited dependency changed: {name}")
        observed[name] = digest
    import_root = source.parent
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    execution = importlib.import_module("pairs_research.crypto_multicoin_execution")
    pipeline = importlib.import_module("pairs_research.crypto_multicoin_pipeline")
    vision = importlib.import_module("pairs_research.binance_vision")
    um_data = importlib.import_module("pairs_research.binance_um_data")
    module_files = {
        "crypto_multicoin_execution.py": execution,
        "crypto_multicoin_pipeline.py": pipeline,
        "binance_vision.py": vision,
        "binance_um_data.py": um_data,
    }
    for name, module in module_files.items():
        observed_path = Path(module.__file__).resolve()
        expected_path = (source / name).resolve()
        if observed_path != expected_path:
            raise ValueError(
                f"imported module did not resolve to the verified snapshot: {name}"
            )
    return AuditedDependencies(
        PortfolioConstraints=execution.PortfolioConstraints,
        backtest_multicoin_pair_ledger=execution.backtest_multicoin_pair_ledger,
        estimate_daily_risk_inputs=pipeline.estimate_daily_risk_inputs,
        VerifiedArchive=vision.VerifiedArchive,
        archive_relative_path=vision.archive_relative_path,
        parse_checksum=vision.parse_checksum,
        read_single_csv=vision.read_single_csv,
        normalize_kline_frame=um_data.normalize_kline_frame,
        normalize_funding_frame=um_data.normalize_funding_frame,
        source_hashes=observed,
    )


def _month_labels(start: pd.Timestamp, end_inclusive: pd.Timestamp) -> tuple[str, ...]:
    first = _utc(start, field="market start")
    last = _utc(end_inclusive, field="market end")
    if last < first or last >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError("market request reaches the confirmation firewall")
    periods = pd.period_range(
        pd.Period(first.strftime("%Y-%m"), freq="M"),
        pd.Period(last.strftime("%Y-%m"), freq="M"),
        freq="M",
    )
    labels = tuple(str(value) for value in periods)
    for label in labels:
        require_development_month(label)
    return labels


def _verified_archive(
    row: Any,
    *,
    cache_root: Path,
    dependencies: AuditedDependencies,
) -> Any:
    month = require_development_month(str(row.month))
    dataset = str(row.dataset)
    symbol = str(row.symbol)
    if dataset not in {"klines", "fundingRate"}:
        raise ValueError("unexpected dataset in explicit allowlist")
    expected_relative = dependencies.archive_relative_path(
        symbol, month, dataset=dataset, interval="1m"
    )
    if expected_relative.as_posix() != str(row.archive_relative_path):
        raise ValueError("manifest archive path differs from the exact requested path")
    path = Path(cache_root) / expected_relative
    sidecar = path.with_name(f"{path.name}.CHECKSUM")
    expected_sidecar = expected_relative.parent / f"{expected_relative.name}.CHECKSUM"
    if expected_sidecar.as_posix() != str(row.checksum_relative_path):
        raise ValueError("manifest checksum path differs from the exact requested path")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != str(row.archive_sha256) or observed != str(
        row.official_checksum_sha256
    ):
        raise ValueError("archive hash differs from the manifest or official checksum")
    if sha256_file(sidecar) != str(row.checksum_sidecar_sha256):
        raise ValueError("checksum sidecar hash differs from the manifest")
    official = dependencies.parse_checksum(
        sidecar.read_text(encoding="ascii"), expected_filename=path.name
    )
    if official != observed:
        raise ValueError("official checksum differs from the archive")
    return dependencies.VerifiedArchive(
        dataset=dataset,
        symbol=symbol,
        month=month,
        path=path,
        sha256=observed,
        source_url=str(row.source_url),
    )


def _assert_frame_month(index: pd.DatetimeIndex, month: str, *, label: str) -> None:
    start = pd.Timestamp(f"{require_development_month(month)}-01T00:00:00Z")
    stop = start + pd.offsets.MonthBegin(1)
    if len(index) and (index.min() < start or index.max() >= stop):
        raise ValueError(f"{label} archive contains an out-of-month timestamp")
    if len(index) and index.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError(f"{label} archive crosses the confirmation firewall")


def load_market_inputs(
    *,
    cache_root: Path,
    manifest_directory: Path,
    symbols: Sequence[str],
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
    dependencies: AuditedDependencies,
) -> MarketInputs:
    """Load only explicitly allowlisted months, reduced immediately to hourly data."""

    manifest = load_acquisition_manifest(manifest_directory, require_full_contract=True)
    names = tuple(map(str, symbols))
    if names != manifest.universe:
        raise ValueError("component and acquisition universe axes differ")
    months = _month_labels(start, end_inclusive)
    availability_path = Path(manifest_directory).resolve() / "archive_availability.csv"
    rows = pd.read_csv(availability_path, dtype=str, keep_default_na=False)
    selected = rows.loc[
        rows["symbol"].isin(names)
        & rows["month"].isin(months)
        & rows["dataset"].isin(("klines", "fundingRate"))
    ].copy()
    expected_keys = {
        (symbol, month, dataset)
        for symbol in names
        for month in months
        for dataset in ("klines", "fundingRate")
    }
    observed_keys = set(
        selected[["symbol", "month", "dataset"]].itertuples(index=False, name=None)
    )
    if (
        observed_keys != expected_keys
        or selected.duplicated(["symbol", "month", "dataset"]).any()
    ):
        raise ValueError("explicit market allowlist is incomplete or duplicated")
    if (selected["month"] > "2025-06").any():
        raise PermissionError("selected market allowlist contains a reserved month")
    lookup = {
        (str(row.symbol), str(row.month), str(row.dataset)): row
        for row in selected.itertuples(index=False)
    }
    first_hour = _utc(start).floor("h")
    last_hour = _utc(end_inclusive).ceil("h")
    hour_grid = pd.date_range(first_hour, last_hour, freq="1h", tz="UTC")
    open_columns: dict[str, pd.Series] = {}
    close_columns: dict[str, pd.Series] = {}
    previous_columns: dict[str, pd.Series] = {}
    funding_by_symbol: dict[str, list[tuple[str, pd.Series | None]]] = {}
    funding_count = 0
    missing_funding_count = 0
    for symbol in names:
        hourly_opens: list[pd.Series] = []
        hourly_closes: list[pd.Series] = []
        observed_minute_blocks: list[pd.DatetimeIndex] = []
        funding_blocks: list[tuple[str, pd.Series | None]] = []
        for month in months:
            kline_row = lookup[(symbol, month, "klines")]
            if str(kline_row.availability) == "available_verified":
                archive = _verified_archive(
                    kline_row, cache_root=cache_root, dependencies=dependencies
                )
                raw = dependencies.read_single_csv(archive)
                normalized, _audit = dependencies.normalize_kline_frame(
                    raw, symbol=symbol, interval="1m"
                )
                _assert_frame_month(normalized.index, month, label="kline")
                opens = normalized.loc[normalized.index.minute == 0, "open"].astype(
                    float
                )
                closes = (
                    normalized["close"]
                    .astype(float)
                    .resample("1h", label="left", closed="left")
                    .last()
                )
                hourly_opens.append(opens)
                hourly_closes.append(closes)
                observed_minute_blocks.append(normalized.index)
            elif str(kline_row.availability) != "missing_remote_404":
                raise ValueError("unsupported kline availability status")

            funding_row = lookup[(symbol, month, "fundingRate")]
            if str(funding_row.availability) == "available_verified":
                archive = _verified_archive(
                    funding_row, cache_root=cache_root, dependencies=dependencies
                )
                series = dependencies.normalize_funding_frame(
                    dependencies.read_single_csv(archive), symbol=symbol
                )
                _assert_frame_month(series.index, month, label="funding")
                if (
                    len(series)
                    and (
                        (series.index.minute != 0)
                        | (series.index.second != 0)
                        | (series.index.microsecond != 0)
                    ).any()
                ):
                    raise ValueError(
                        "funding event is not on the hourly execution clock"
                    )
                funding_blocks.append((month, series.astype(float)))
                funding_count += 1
            elif str(funding_row.availability) == "missing_remote_404":
                funding_blocks.append((month, None))
                missing_funding_count += 1
            else:
                raise ValueError("unsupported funding availability status")
        open_columns[symbol] = (
            pd.concat(hourly_opens).sort_index().reindex(hour_grid)
            if hourly_opens
            else pd.Series(np.nan, index=hour_grid)
        )
        close_columns[symbol] = (
            pd.concat(hourly_closes).sort_index().reindex(hour_grid)
            if hourly_closes
            else pd.Series(np.nan, index=hour_grid)
        )
        observed_minutes = (
            observed_minute_blocks[0].append(observed_minute_blocks[1:])
            if observed_minute_blocks
            else pd.DatetimeIndex([], tz="UTC")
        )
        previous_columns[symbol] = pd.Series(
            (hour_grid - pd.Timedelta(minutes=1)).isin(observed_minutes),
            index=hour_grid,
            dtype=bool,
        )
        funding_by_symbol[symbol] = funding_blocks

    all_event_times: set[pd.Timestamp] = set()
    for blocks in funding_by_symbol.values():
        for _month, series in blocks:
            if series is not None:
                all_event_times.update(map(pd.Timestamp, series.index))
    funding_clock = pd.DatetimeIndex(sorted(all_event_times), tz="UTC")
    funding_columns: dict[str, pd.Series] = {}
    for symbol, blocks in funding_by_symbol.items():
        result = pd.Series(0.0, index=funding_clock, dtype=float, name=symbol)
        for month, series in blocks:
            month_start = pd.Timestamp(f"{month}-01T00:00:00Z")
            month_stop = month_start + pd.offsets.MonthBegin(1)
            mask = (result.index >= month_start) & (result.index < month_stop)
            if series is None:
                result.loc[mask] = np.nan
            else:
                result.loc[mask] = series.reindex(result.index[mask]).fillna(0.0)
        funding_columns[symbol] = result
    market = MarketInputs(
        open_prices=pd.DataFrame(open_columns, index=hour_grid).rename_axis(
            "timestamp"
        ),
        last_closes=pd.DataFrame(close_columns, index=hour_grid).rename_axis(
            "timestamp"
        ),
        previous_minute_available=pd.DataFrame(previous_columns, index=hour_grid)
        .astype(bool)
        .rename_axis("timestamp"),
        funding_rates=pd.DataFrame(funding_columns, index=funding_clock).rename_axis(
            "timestamp"
        ),
        manifest_rows=selected.reset_index(drop=True),
        loaded_months=months,
        funding_archive_count=funding_count,
        missing_funding_archive_count=missing_funding_count,
    )
    if market.open_prices.index.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError("market panel reaches the confirmation firewall")
    return market


def execution_slice(
    market: MarketInputs, clock: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return causal availability, fixed-quantity returns and sparse funding."""

    times = pd.DatetimeIndex(clock).tz_convert("UTC")
    if times.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError("execution clock reaches the confirmation firewall")
    opens = market.open_prices.reindex(times)
    next_opens = market.open_prices.reindex(times + pd.Timedelta(hours=1))
    next_opens.index = times
    terminal_marks = market.last_closes.reindex(times)
    exits = next_opens.combine_first(terminal_marks)
    availability = (
        market.previous_minute_available.reindex(times).eq(True)
        & opens.notna()
        & opens.gt(0.0)
    )
    unmarkable = availability & (exits.isna() | exits.le(0.0))
    if unmarkable.any().any():
        row, column = np.argwhere(unmarkable.to_numpy())[0]
        raise ValueError(
            f"causally available position has no exit mark: {times[row]} "
            f"{unmarkable.columns[column]}"
        )
    returns = exits.div(opens).sub(1.0).where(availability)
    if ((returns <= -1.0) & availability).any().any():
        raise ValueError("available fixed-quantity return is not greater than -100%")
    funding = market.funding_rates.loc[market.funding_rates.index.isin(times)].copy()
    return returns, availability, funding


def static_axes() -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    parent_path = HERE.parent / "contracts" / "universe.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    symbols = tuple(map(str, parent["universe"]["static_symbols_in_rank_order"]))
    if len(symbols) != 24 or len(set(symbols)) != 24:
        raise ValueError("parent universe is not the static Top-24")
    legs = np.asarray(
        [
            (second, first)
            for first in range(len(symbols))
            for second in range(first + 1, len(symbols))
        ],
        dtype=np.int64,
    )
    pair_ids = tuple(f"{symbols[signal]}__{symbols[hedge]}" for signal, hedge in legs)
    if len(pair_ids) != 276:
        raise AssertionError("static universe did not generate 276 oriented pairs")
    return symbols, pair_ids, legs


def _rolling_pair_weights_from_executable_returns(
    returns: pd.DataFrame,
    *,
    legs: np.ndarray,
    lookback_hours: int,
    minimum_hours: int,
    beta_clip: tuple[float, float],
) -> np.ndarray:
    """Fit the declared hedge on completed open-to-open log returns.

    The return stored at row ``u`` spans open ``u`` to open ``u+1``.  A shift
    of one therefore makes the newest formation observation at decision ``t``
    the completed interval ``t-1 -> t``.
    """

    simple = pd.DataFrame(returns, dtype=float)
    log_returns = np.log1p(simple.where(simple > -1.0))
    lagged = log_returns.shift(1)
    result = np.full((len(simple), len(legs), 2), np.nan, dtype=np.float64)
    for pair, (signal, hedge) in enumerate(np.asarray(legs, dtype=np.int64)):
        y = lagged.iloc[:, int(signal)]
        x = lagged.iloc[:, int(hedge)]
        covariance = (
            pd.concat({"x": x, "y": y}, axis=1)
            .rolling(int(lookback_hours), min_periods=int(minimum_hours))
            .cov()
            .unstack()
            .loc[:, ("x", "y")]
        )
        variance = x.rolling(int(lookback_hours), min_periods=int(minimum_hours)).var(
            ddof=1
        )
        beta = covariance.div(variance.where(variance > 1e-12)).clip(*beta_clip)
        denominator = 1.0 + beta.abs()
        result[:, pair, 0] = (1.0 / denominator).to_numpy(dtype=float)
        result[:, pair, 1] = (-beta / denominator).to_numpy(dtype=float)
    return result


def build_economic_inputs(
    *,
    market: MarketInputs,
    symbols: tuple[str, ...],
    pair_ids: tuple[str, ...],
    pair_leg_indices: np.ndarray,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> EconomicInputs:
    clock = market.open_prices.index
    if (
        clock.min() < pd.Timestamp("2021-01-01T00:00:00Z")
        or clock.max() >= DEVELOPMENT_END_EXCLUSIVE
    ):
        raise PermissionError("economic clock lies outside the development firewall")
    opens = market.open_prices.reindex(columns=list(symbols)).astype(float)
    next_opens = opens.reindex(clock + pd.Timedelta(hours=1))
    next_opens.index = clock
    terminal_marks = market.last_closes.reindex(clock).reindex(columns=list(symbols))
    # A source termination is a terminal event only when a later requested
    # hour exists.  The global June cutoff itself must not be converted into a
    # shortened pseudo-hour.
    later_hour_is_inside = (clock + pd.Timedelta(hours=1)) <= clock.max()
    terminal_exit = pd.DataFrame(False, index=clock, columns=symbols)
    for symbol in symbols:
        valid = opens[symbol].dropna()
        if valid.empty:
            continue
        final_source_open = valid.index[-1]
        terminal_exit.loc[:, symbol] = (
            next_opens[symbol].isna()
            & terminal_marks[symbol].notna()
            & clock.to_series(index=clock).eq(final_source_open)
            & later_hour_is_inside
        )
    exits = next_opens.where(next_opens.notna(), terminal_marks.where(terminal_exit))
    decision_available = (
        market.previous_minute_available.reindex(clock)
        .reindex(columns=list(symbols))
        .eq(True)
        & opens.notna()
        & opens.gt(0.0)
    )
    unclassified_gap = decision_available & next_opens.isna() & ~terminal_exit
    # The final global cutoff has no permitted t+1 outcome and is masked.  A
    # transient interior gap is not a terminal event: its unmeasurable outcome
    # is masked below without changing causal decision availability, filling a
    # price, or shortening the one-hour holding period.
    inside_next_hour = pd.DataFrame(
        np.broadcast_to(
            np.asarray(later_hour_is_inside)[:, None], decision_available.shape
        ),
        index=clock,
        columns=symbols,
    )
    transient_gap = unclassified_gap & inside_next_hour
    simple_returns = exits.div(opens).sub(1.0).where(decision_available)
    if ((simple_returns <= -1.0) & decision_available).any().any():
        raise ValueError("economic input contains a return at or below -100%")

    funding_boundary = pd.DataFrame(0.0, index=clock, columns=symbols)
    observed_boundary = market.funding_rates.reindex(clock)
    event_at_boundary = clock.isin(market.funding_rates.index)
    funding_boundary.loc[event_at_boundary, :] = observed_boundary.loc[
        event_at_boundary, list(symbols)
    ]
    # If the previous interval was terminally marked, the position was closed
    # at that mark and cannot owe funding at this boundary.
    prior_terminal = terminal_exit.shift(1, fill_value=False)
    funding_boundary = funding_boundary.mask(prior_terminal, 0.0)
    funding_next = funding_boundary.reindex(clock + pd.Timedelta(hours=1))
    funding_next.index = clock
    funding_next = funding_next.mask(terminal_exit, 0.0)
    growth = 1.0 + simple_returns
    target_available = (
        decision_available
        & simple_returns.notna()
        & funding_next.notna()
        & growth.gt(0.0)
    )
    weights = _rolling_pair_weights_from_executable_returns(
        simple_returns,
        legs=pair_leg_indices,
        lookback_hours=int(contract["target"]["hedge_estimation"]["lookback_hours"]),
        minimum_hours=int(
            contract["target"]["hedge_estimation"]["minimum_complete_hours"]
        ),
        beta_clip=tuple(contract["target"]["hedge_estimation"]["beta_clip"]),
    )
    pair_decision = np.zeros((len(clock), len(pair_ids)), dtype=bool)
    pair_target = np.zeros_like(pair_decision)
    cashflows = np.zeros((len(clock), len(pair_ids)), dtype=np.float64)
    returns_array = simple_returns.to_numpy(dtype=float)
    funding_next_array = funding_next.to_numpy(dtype=float)
    growth_array = growth.to_numpy(dtype=float)
    decision_array = decision_available.to_numpy(dtype=bool)
    target_array = target_available.to_numpy(dtype=bool)
    for pair, (signal, hedge) in enumerate(
        np.asarray(pair_leg_indices, dtype=np.int64)
    ):
        finite_weight = np.isfinite(weights[:, pair]).all(axis=1)
        pair_decision[:, pair] = (
            finite_weight & decision_array[:, signal] & decision_array[:, hedge]
        )
        pair_target[:, pair] = (
            pair_decision[:, pair] & target_array[:, signal] & target_array[:, hedge]
        )
        price = (
            weights[:, pair, 0] * returns_array[:, signal]
            + weights[:, pair, 1] * returns_array[:, hedge]
        )
        next_funding = -(
            weights[:, pair, 0]
            * growth_array[:, signal]
            * funding_next_array[:, signal]
            + weights[:, pair, 1]
            * growth_array[:, hedge]
            * funding_next_array[:, hedge]
        )
        values = price + next_funding
        cashflows[:, pair] = np.where(pair_target[:, pair], values, 0.0)
    if not np.isfinite(cashflows).all():
        raise ValueError("masked pair target cashflows are nonfinite")
    if np.any(pair_target & ~pair_decision):
        raise AssertionError(
            "target availability is not a subset of decision availability"
        )
    transient_array = transient_gap.to_numpy(dtype=bool)
    if (
        np.isfinite(returns_array[transient_array]).any()
        or terminal_exit.to_numpy(dtype=bool)[transient_array].any()
        or target_array[transient_array].any()
        or not decision_array[transient_array].all()
    ):
        raise AssertionError(
            "transient gaps were filled, shortened, outcome-enabled, or decision-masked"
        )
    pair_transient = np.zeros_like(pair_target)
    legs = np.asarray(pair_leg_indices, dtype=np.int64)
    for pair, (signal, hedge) in enumerate(legs):
        pair_transient[:, pair] = pair_decision[:, pair] & (
            transient_array[:, signal] | transient_array[:, hedge]
        )
    if np.any(pair_transient & pair_target):
        raise AssertionError("a transient-gap pair label remained available")
    test_rows = np.zeros(len(clock), dtype=bool)
    for fold in contract["folds"]:
        test_rows |= (clock >= _utc(fold["test_start"])) & (
            clock < _utc(fold["test_end_exclusive"])
        )
    oos_coin_gap_cells = int(transient_array[test_rows].sum())
    oos_pair_label_cells = int(pair_transient[test_rows].sum())
    if oos_coin_gap_cells != 0 or oos_pair_label_cells != 0:
        raise ValueError(
            "a transient open gap reaches a declared 2023/24 test-fold clock"
        )
    events: list[dict[str, Any]] = []
    for row in np.flatnonzero(transient_array.any(axis=1)):
        affected_symbols = [
            symbol
            for column, symbol in enumerate(symbols)
            if transient_array[row, column]
        ]
        events.append(
            {
                "decision_timestamp": clock[row].isoformat(),
                "missing_next_open_timestamp": (
                    clock[row] + pd.Timedelta(hours=1)
                ).isoformat(),
                "symbols": affected_symbols,
                "coin_gap_cells": len(affected_symbols),
                "pair_label_cells_masked": int(pair_transient[row].sum()),
            }
        )
    transient_report = {
        "policy": "no_fill_no_shortening_mask_outcome_only",
        "no_fill": True,
        "no_shortening": True,
        "feature_and_decision_availability_unchanged": True,
        "coin_gap_cells": int(transient_array.sum()),
        "pair_label_cells_masked": int(pair_transient.sum()),
        "oos_coin_gap_cells": oos_coin_gap_cells,
        "oos_pair_label_cells": oos_pair_label_cells,
        "events": events,
    }
    governed_gap_policy = contract["data"].get("transient_open_gap_policy")
    if not isinstance(governed_gap_policy, dict):
        raise ValueError("contract lacks the transient-open-gap policy")
    observed_events = [
        {
            "decision_timestamp": event["decision_timestamp"],
            "missing_next_open_timestamp": event["missing_next_open_timestamp"],
            "symbols": event["symbols"],
        }
        for event in events
    ]
    if (
        governed_gap_policy.get("rule") != transient_report["policy"]
        or governed_gap_policy.get("expected_coin_gap_cells")
        != transient_report["coin_gap_cells"]
        or governed_gap_policy.get("expected_pair_label_cells_masked")
        != transient_report["pair_label_cells_masked"]
        or governed_gap_policy.get("expected_events") != observed_events
        or governed_gap_policy.get("maximum_oos_coin_gap_cells") != 0
        or governed_gap_policy.get("maximum_oos_pair_label_cells") != 0
    ):
        raise ValueError("observed transient open gaps differ from the contract")
    metadata = {
        "schema": "paired_hourly_economic_inputs_v2",
        "clock_start": clock[0].isoformat(),
        "clock_end": clock[-1].isoformat(),
        "hard_cutoff_exclusive": DEVELOPMENT_END_EXCLUSIVE.isoformat(),
        "price_timing": "arithmetic open t to next open t+1; source terminal uses final in-hour close",
        "funding_timing": "action at t excludes funding[t] and includes funding[t+1] on grown fixed quantity; terminal exit includes no next-boundary funding",
        "pair_cashflow_formula": "sum_l w[t,l]*r[t,l] - sum_l w[t,l]*(1+r[t,l])*funding[t+1,l]",
        "funding_sign": "positive rate is paid by a long and received by a short",
        "hedge_formula": "signal=1/(1+abs(beta)); hedge=-beta/(1+abs(beta))",
        "hedge_information": "720 prior completed hourly open-to-open log returns, minimum 168, one-row shift, beta clipped [-3,3]",
        "symbols": list(symbols),
        "pair_ids": list(pair_ids),
        "transient_open_gap_report": transient_report,
        "provenance": dict(provenance),
    }
    return EconomicInputs(
        clock=clock,
        symbols=symbols,
        pair_ids=pair_ids,
        pair_leg_indices=np.asarray(pair_leg_indices, dtype=np.int64),
        pair_weights=weights,
        coin_open_prices=opens.to_numpy(dtype=np.float64),
        coin_open_to_open_returns=returns_array,
        funding_at_boundary=funding_boundary.to_numpy(dtype=np.float64),
        funding_at_next_boundary=funding_next_array,
        next_boundary_growth=growth_array,
        coin_decision_available=decision_array,
        coin_target_available=target_array,
        terminal_exit_mask=terminal_exit.to_numpy(dtype=bool),
        pair_decision_available=pair_decision,
        pair_target_available=pair_target,
        pair_target_cashflows=cashflows,
        metadata=metadata,
    )


def save_economic_inputs(path: Path, economic: EconomicInputs) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(
            json.dumps(economic.metadata, sort_keys=True, separators=(",", ":"))
        ),
        clock_ns=economic.clock.asi8,
        symbols=np.asarray(economic.symbols),
        pair_ids=np.asarray(economic.pair_ids),
        pair_leg_indices=economic.pair_leg_indices,
        pair_weights=economic.pair_weights,
        coin_open_prices=economic.coin_open_prices,
        coin_open_to_open_returns=economic.coin_open_to_open_returns,
        funding_at_boundary=economic.funding_at_boundary,
        funding_at_next_boundary=economic.funding_at_next_boundary,
        next_boundary_growth=economic.next_boundary_growth,
        coin_decision_available=economic.coin_decision_available,
        coin_target_available=economic.coin_target_available,
        terminal_exit_mask=economic.terminal_exit_mask,
        pair_decision_available=economic.pair_decision_available,
        pair_target_available=economic.pair_target_available,
        pair_target_cashflows=economic.pair_target_cashflows,
    )
    temporary.replace(destination)


def load_economic_inputs(
    path: Path, manifest_path: Path | None = None
) -> EconomicInputs:
    source = Path(path).expanduser().resolve()
    manifest_file = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else source.with_suffix(".manifest.json")
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != ECONOMIC_MANIFEST_KEYS:
        raise ValueError("economic input manifest key set differs")
    if (
        manifest.get("schema") != "paired_hourly_economic_inputs_manifest_v2"
        or manifest.get("status")
        != "complete_immutable_shared_training_and_ledger_input"
    ):
        raise ValueError("economic input manifest schema or status differs")
    if Path(str(manifest.get("artifact_path"))).expanduser().resolve() != source:
        raise ValueError("economic input manifest names a different artifact path")
    if manifest.get("artifact_sha256") != sha256_file(source):
        raise ValueError("economic input artifact differs from its manifest")
    if manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("economic input is not bound to the paired contract")
    receipt = source.with_suffix(".sha256")
    expected_receipt = f"{manifest['artifact_sha256']}  {source.name}\n"
    if not receipt.is_file() or receipt.read_text(encoding="ascii") != expected_receipt:
        raise ValueError("economic input SHA-256 receipt differs")
    with np.load(source, allow_pickle=False) as stored:
        if set(stored.files) != ECONOMIC_MEMBERS:
            raise ValueError("economic input NPZ members differ")
        metadata = json.loads(str(stored["metadata_json"].item()))
        economic = EconomicInputs(
            clock=pd.to_datetime(stored["clock_ns"], utc=True),
            symbols=tuple(stored["symbols"].astype(str)),
            pair_ids=tuple(stored["pair_ids"].astype(str)),
            pair_leg_indices=np.asarray(stored["pair_leg_indices"], dtype=np.int64),
            pair_weights=np.asarray(stored["pair_weights"], dtype=np.float64),
            coin_open_prices=np.asarray(stored["coin_open_prices"], dtype=np.float64),
            coin_open_to_open_returns=np.asarray(
                stored["coin_open_to_open_returns"], dtype=np.float64
            ),
            funding_at_boundary=np.asarray(
                stored["funding_at_boundary"], dtype=np.float64
            ),
            funding_at_next_boundary=np.asarray(
                stored["funding_at_next_boundary"], dtype=np.float64
            ),
            next_boundary_growth=np.asarray(
                stored["next_boundary_growth"], dtype=np.float64
            ),
            coin_decision_available=np.asarray(
                stored["coin_decision_available"], dtype=bool
            ),
            coin_target_available=np.asarray(
                stored["coin_target_available"], dtype=bool
            ),
            terminal_exit_mask=np.asarray(stored["terminal_exit_mask"], dtype=bool),
            pair_decision_available=np.asarray(
                stored["pair_decision_available"], dtype=bool
            ),
            pair_target_available=np.asarray(
                stored["pair_target_available"], dtype=bool
            ),
            pair_target_cashflows=np.asarray(
                stored["pair_target_cashflows"], dtype=np.float64
            ),
            metadata=metadata,
        )
    if metadata.get("schema") != "paired_hourly_economic_inputs_v2":
        raise ValueError("economic input metadata schema differs")
    if economic.clock.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError("economic input reaches the confirmation firewall")
    time_count, coin_count, pair_count = (
        len(economic.clock),
        len(economic.symbols),
        len(economic.pair_ids),
    )
    shapes = {
        "pair_leg_indices": (pair_count, 2),
        "pair_weights": (time_count, pair_count, 2),
        "coin_open_prices": (time_count, coin_count),
        "coin_open_to_open_returns": (time_count, coin_count),
        "funding_at_boundary": (time_count, coin_count),
        "funding_at_next_boundary": (time_count, coin_count),
        "next_boundary_growth": (time_count, coin_count),
        "coin_decision_available": (time_count, coin_count),
        "coin_target_available": (time_count, coin_count),
        "terminal_exit_mask": (time_count, coin_count),
        "pair_decision_available": (time_count, pair_count),
        "pair_target_available": (time_count, pair_count),
        "pair_target_cashflows": (time_count, pair_count),
    }
    for name, expected in shapes.items():
        if getattr(economic, name).shape != expected:
            raise ValueError(f"economic input member has wrong shape: {name}")
    if (
        not economic.clock.is_monotonic_increasing
        or economic.clock.has_duplicates
        or (
            len(economic.clock) > 1
            and not np.all(np.diff(economic.clock.asi8) == pd.Timedelta(hours=1).value)
        )
    ):
        raise ValueError("economic input clock is not a strict hourly axis")
    if (
        len(set(economic.symbols)) != coin_count
        or len(set(economic.pair_ids)) != pair_count
    ):
        raise ValueError("economic symbol or pair axis contains duplicates")
    if (economic.pair_leg_indices < 0).any() or (
        economic.pair_leg_indices >= coin_count
    ).any():
        raise ValueError("economic pair-leg indices lie outside the coin axis")
    expected_pair_hash = hashlib.sha256(
        "\n".join(economic.pair_ids).encode()
    ).hexdigest()
    manifest_axes = {
        "clock_start": economic.clock[0].isoformat(),
        "clock_end": economic.clock[-1].isoformat(),
        "hourly_observations": time_count,
        "symbols": list(economic.symbols),
        "pair_ids_sha256": expected_pair_hash,
        "pairs": pair_count,
        "pair_decision_available_cells": int(economic.pair_decision_available.sum()),
        "pair_target_available_cells": int(economic.pair_target_available.sum()),
        "terminal_exit_cells": int(economic.terminal_exit_mask.sum()),
    }
    for field, expected in manifest_axes.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"economic input manifest axis/content count differs: {field}"
            )
    contract = load_contract()
    acquisition_hashes = {
        field: contract["data"][field]
        for field in (
            "acquisition_manifest_sha256",
            "archive_availability_sha256",
            "artifact_hashes_sha256",
        )
    }
    for field, expected in acquisition_hashes.items():
        if manifest.get(field) != expected:
            raise ValueError(f"economic input acquisition binding differs: {field}")
    if manifest.get("audited_source_hashes") != AUDITED_SOURCE_HASHES:
        raise ValueError("economic input audited-source binding differs")
    if (
        manifest.get("latest_raw_month") != "2025-06"
        or manifest.get("july_or_later_path_count") != 0
        or manifest.get("confirmation_rows_parsed") is not False
        or manifest.get("funding_included_in_target") is not True
    ):
        raise ValueError("economic input firewall or funding policy differs")
    for field in ("funding_verified_archives_loaded", "funding_missing_archives"):
        if not isinstance(manifest.get(field), int) or manifest[field] < 0:
            raise ValueError(
                f"economic input funding archive count is invalid: {field}"
            )
    provenance = metadata.get("provenance")
    expected_provenance = {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        **acquisition_hashes,
        "audited_source_hashes": AUDITED_SOURCE_HASHES,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != ECONOMIC_PROVENANCE_KEYS
        or provenance != expected_provenance
    ):
        raise ValueError("economic input metadata provenance differs")
    expected_metadata = {
        "hard_cutoff_exclusive": DEVELOPMENT_END_EXCLUSIVE.isoformat(),
        "symbols": list(economic.symbols),
        "pair_ids": list(economic.pair_ids),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"economic input metadata differs: {field}")
    if manifest.get("pair_target_formula") != metadata.get("pair_cashflow_formula"):
        raise ValueError(
            "economic input target formula differs between metadata and manifest"
        )
    one_period = manifest.get("one_period_ledger_identity")
    expected_cases = {
        "ordinary_with_next_boundary_funding",
        "ordinary_without_funding_event",
        "terminal_delisting_mark",
    }
    if not isinstance(one_period, dict) or set(one_period) != expected_cases:
        raise ValueError("economic one-period identity report differs")
    for label, report in one_period.items():
        if (
            not isinstance(report, dict)
            or report.get("passed") is not True
            or not np.isfinite(report.get("absolute_error", np.nan))
            or float(report["absolute_error"]) > 1e-12
            or report.get("pair_id") not in economic.pair_ids
        ):
            raise ValueError(f"economic one-period identity failed: {label}")
    multi_period = manifest.get("multi_period_ledger_identity")
    if (
        not isinstance(multi_period, dict)
        or multi_period.get("passed") is not True
        or multi_period.get("terminal_zero_target_included") is not True
        or multi_period.get("contains_nonzero_next_boundary_funding") is not True
        or not np.isfinite(multi_period.get("absolute_error", np.nan))
        or float(multi_period["absolute_error"]) > 1e-12
        or multi_period.get("pair_id") not in economic.pair_ids
    ):
        raise ValueError("economic multi-period terminal identity failed")
    transient_report = manifest.get("transient_open_gap_report")
    if transient_report != metadata.get("transient_open_gap_report") or not isinstance(
        transient_report, dict
    ):
        raise ValueError(
            "economic transient-gap report differs between manifest and artifact"
        )
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
    if set(transient_report) != expected_report_keys or any(
        transient_report.get(field) is not True
        for field in (
            "no_fill",
            "no_shortening",
            "feature_and_decision_availability_unchanged",
        )
    ):
        raise ValueError("economic transient-gap report schema or policy differs")
    rebuilt_transient = np.zeros_like(economic.coin_decision_available)
    rebuilt_transient[:-1] = (
        economic.coin_decision_available[:-1]
        & ~np.isfinite(economic.coin_open_prices[1:])
        & ~economic.terminal_exit_mask[:-1]
    )
    if (
        np.isfinite(economic.coin_open_to_open_returns[rebuilt_transient]).any()
        or economic.coin_target_available[rebuilt_transient].any()
        or economic.terminal_exit_mask[rebuilt_transient].any()
        or not economic.coin_decision_available[rebuilt_transient].all()
    ):
        raise ValueError(
            "economic transient gap was filled, shortened, or decision-masked"
        )
    rebuilt_pair_transient = np.zeros_like(economic.pair_decision_available)
    for pair, (signal, hedge) in enumerate(economic.pair_leg_indices):
        rebuilt_pair_transient[:, pair] = economic.pair_decision_available[:, pair] & (
            rebuilt_transient[:, signal] | rebuilt_transient[:, hedge]
        )
    test_rows = np.zeros(time_count, dtype=bool)
    for fold in contract["folds"]:
        test_rows |= (economic.clock >= _utc(fold["test_start"])) & (
            economic.clock < _utc(fold["test_end_exclusive"])
        )
    rebuilt_events: list[dict[str, Any]] = []
    for row in np.flatnonzero(rebuilt_transient.any(axis=1)):
        affected_symbols = [
            symbol
            for column, symbol in enumerate(economic.symbols)
            if rebuilt_transient[row, column]
        ]
        rebuilt_events.append(
            {
                "decision_timestamp": economic.clock[row].isoformat(),
                "missing_next_open_timestamp": (
                    economic.clock[row] + pd.Timedelta(hours=1)
                ).isoformat(),
                "symbols": affected_symbols,
                "coin_gap_cells": len(affected_symbols),
                "pair_label_cells_masked": int(rebuilt_pair_transient[row].sum()),
            }
        )
    rebuilt_report_values = {
        "policy": "no_fill_no_shortening_mask_outcome_only",
        "coin_gap_cells": int(rebuilt_transient.sum()),
        "pair_label_cells_masked": int(rebuilt_pair_transient.sum()),
        "oos_coin_gap_cells": int(rebuilt_transient[test_rows].sum()),
        "oos_pair_label_cells": int(rebuilt_pair_transient[test_rows].sum()),
        "events": rebuilt_events,
    }
    for field, expected in rebuilt_report_values.items():
        if transient_report.get(field) != expected:
            raise ValueError(f"economic transient-gap report differs: {field}")
    if (
        transient_report["oos_coin_gap_cells"] != 0
        or transient_report["oos_pair_label_cells"] != 0
    ):
        raise ValueError("economic transient gaps reach a declared test fold")
    governed_gap_policy = contract["data"].get("transient_open_gap_policy")
    governed_events = [
        {
            "decision_timestamp": event["decision_timestamp"],
            "missing_next_open_timestamp": event["missing_next_open_timestamp"],
            "symbols": event["symbols"],
        }
        for event in rebuilt_events
    ]
    if (
        not isinstance(governed_gap_policy, dict)
        or governed_gap_policy.get("rule") != transient_report["policy"]
        or governed_gap_policy.get("expected_coin_gap_cells")
        != transient_report["coin_gap_cells"]
        or governed_gap_policy.get("expected_pair_label_cells_masked")
        != transient_report["pair_label_cells_masked"]
        or governed_gap_policy.get("expected_events") != governed_events
        or governed_gap_policy.get("maximum_oos_coin_gap_cells") != 0
        or governed_gap_policy.get("maximum_oos_pair_label_cells") != 0
    ):
        raise ValueError("economic transient gaps differ from the paired contract")
    if np.any(economic.pair_target_available & ~economic.pair_decision_available):
        raise ValueError(
            "economic target availability is not a subset of decision availability"
        )
    if np.any(economic.terminal_exit_mask & (economic.funding_at_next_boundary != 0.0)):
        raise ValueError("economic terminal exits carry next-boundary funding")
    if len(economic.clock) > 1 and not np.array_equal(
        economic.funding_at_next_boundary[:-1],
        economic.funding_at_boundary[1:],
        equal_nan=True,
    ):
        raise ValueError(
            "economic next-boundary funding is not the shifted boundary series"
        )
    if not np.array_equal(
        economic.next_boundary_growth[economic.coin_target_available],
        1.0 + economic.coin_open_to_open_returns[economic.coin_target_available],
    ):
        raise ValueError("economic next-boundary growth identity differs")
    rebuilt = np.zeros_like(economic.pair_target_cashflows)
    for pair, (signal, hedge) in enumerate(economic.pair_leg_indices):
        price = (
            economic.pair_weights[:, pair, 0]
            * economic.coin_open_to_open_returns[:, signal]
            + economic.pair_weights[:, pair, 1]
            * economic.coin_open_to_open_returns[:, hedge]
        )
        funding = -(
            economic.pair_weights[:, pair, 0]
            * economic.next_boundary_growth[:, signal]
            * economic.funding_at_next_boundary[:, signal]
            + economic.pair_weights[:, pair, 1]
            * economic.next_boundary_growth[:, hedge]
            * economic.funding_at_next_boundary[:, hedge]
        )
        rebuilt[:, pair] = np.where(
            economic.pair_target_available[:, pair], price + funding, 0.0
        )
    if not np.allclose(rebuilt, economic.pair_target_cashflows, rtol=1e-12, atol=1e-14):
        raise ValueError(
            "stored economic pair target does not reproduce from primitives"
        )
    return economic


def assert_one_period_ledger_identity(
    economic: EconomicInputs,
    dependencies: AuditedDependencies,
) -> dict[str, Any]:
    """Prove stored targets equal the audited ledger's fixed-quantity cashflow.

    The proof executes one ordinary interval and one source-terminal interval,
    each followed by a zero-target boundary row.  Compounding the two audited
    gross ledger rows must reproduce the stored unit-pair target exactly.
    """

    reports: dict[str, Any] = {}
    terminal_by_pair = np.zeros_like(economic.pair_target_available)
    for pair, (signal, hedge) in enumerate(economic.pair_leg_indices):
        terminal_by_pair[:, pair] = (
            economic.terminal_exit_mask[:, signal]
            | economic.terminal_exit_mask[:, hedge]
        )
    funding_by_pair = np.zeros_like(economic.pair_target_available)
    for pair, (signal, hedge) in enumerate(economic.pair_leg_indices):
        funding_by_pair[:, pair] = (
            np.abs(economic.funding_at_next_boundary[:, signal]) > 0.0
        ) | (np.abs(economic.funding_at_next_boundary[:, hedge]) > 0.0)
    cases = {
        "ordinary_with_next_boundary_funding": (
            economic.pair_target_available & ~terminal_by_pair & funding_by_pair
        ),
        "ordinary_without_funding_event": (
            economic.pair_target_available & ~terminal_by_pair & ~funding_by_pair
        ),
        "terminal_delisting_mark": economic.pair_target_available & terminal_by_pair,
    }
    for label, eligible in cases.items():
        candidates = np.argwhere(eligible)
        candidates = candidates[candidates[:, 0] + 1 < len(economic.clock)]
        if len(candidates) == 0:
            raise ValueError(f"economic input lacks an identity-test case: {label}")
        row, pair = map(
            int, candidates[-1] if label.startswith("terminal") else candidates[0]
        )
        test_clock = economic.clock.take([row, row + 1])
        pair_id = economic.pair_ids[pair]
        pair_targets = pd.DataFrame(
            [1.0, 0.0], index=test_clock, columns=[pair_id], dtype=float
        )
        hedge = np.zeros((2, len(economic.symbols), 1), dtype=float)
        signal, hedge_leg = map(int, economic.pair_leg_indices[pair])
        hedge[:, signal, 0] = economic.pair_weights[row, pair, 0]
        hedge[:, hedge_leg, 0] = economic.pair_weights[row, pair, 1]
        returns = pd.DataFrame(
            economic.coin_open_to_open_returns[[row, row + 1]],
            index=test_clock,
            columns=economic.symbols,
        )
        availability = pd.DataFrame(
            economic.coin_decision_available[[row, row + 1]],
            index=test_clock,
            columns=economic.symbols,
        )
        funding = pd.DataFrame(
            [economic.funding_at_boundary[row], economic.funding_at_next_boundary[row]],
            index=test_clock,
            columns=economic.symbols,
        )
        # At the second row the stored next-boundary vector is by definition
        # the boundary funding owed by the position opened in the first row.
        constraints = dependencies.PortfolioConstraints(
            maximum_total_gross=2.0,
            maximum_absolute_coin_weight=2.0,
            maximum_absolute_dollar_net=2.0,
        )
        ledger = dependencies.backtest_multicoin_pair_ledger(
            pair_targets,
            hedge,
            returns,
            funding,
            transaction_cost_bps=0.0,
            coin_symbols=economic.symbols,
            availability=availability,
            constraints=constraints,
            liquidate_at_end=True,
        )
        observed = float(
            np.prod(1.0 + ledger.bars["gross_return"].to_numpy(dtype=float)) - 1.0
        )
        expected = float(economic.pair_target_cashflows[row, pair])
        error = observed - expected
        if not np.isclose(observed, expected, rtol=1e-12, atol=1e-14):
            raise ValueError(
                f"stored target and audited ledger differ for {label}: "
                f"expected={expected:.17g} observed={observed:.17g}"
            )
        reports[label] = {
            "timestamp": economic.clock[row].isoformat(),
            "pair_id": pair_id,
            "stored_target_cashflow": expected,
            "audited_ledger_compounded_gross_return": observed,
            "absolute_error": abs(error),
            "passed": True,
        }
    return reports


def assert_multi_period_ledger_identity(
    economic: EconomicInputs,
    dependencies: AuditedDependencies,
    *,
    decision_count: int = 12,
) -> dict[str, Any]:
    """Reconcile a changing one-pair action path through its terminal funding row."""

    selected: tuple[int, int] | None = None
    for pair, (signal, hedge_leg) in enumerate(economic.pair_leg_indices):
        eligible = (
            economic.pair_target_available[:, pair]
            & ~economic.terminal_exit_mask[:, signal]
            & ~economic.terminal_exit_mask[:, hedge_leg]
        )
        funding = np.abs(economic.funding_at_next_boundary[:, signal]) + np.abs(
            economic.funding_at_next_boundary[:, hedge_leg]
        )
        for start in np.flatnonzero(eligible):
            stop = int(start) + int(decision_count)
            if stop >= len(economic.clock):
                continue
            if eligible[start:stop].all() and np.any(funding[start:stop] > 0.0):
                selected = (int(start), int(pair))
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError(
            "economic input lacks a continuous multi-period funding identity case"
        )
    start, pair = selected
    decision_rows = np.arange(start, start + int(decision_count), dtype=np.int64)
    ledger_rows = np.arange(start, start + int(decision_count) + 1, dtype=np.int64)
    clock = economic.clock.take(ledger_rows)
    # Deterministic changes in sign and size ensure that the ledger actually
    # rebalances rather than degenerating into a buy-and-hold identity.
    pattern = np.asarray([0.40, -0.25, 0.15, -0.35, 0.30, -0.10], dtype=float)
    actions = np.resize(pattern, int(decision_count))
    q = np.concatenate([actions, [0.0]])
    pair_id = economic.pair_ids[pair]
    pair_targets = pd.DataFrame(q, index=clock, columns=[pair_id])
    hedge = np.zeros((len(clock), len(economic.symbols), 1), dtype=float)
    signal, hedge_leg = map(int, economic.pair_leg_indices[pair])
    hedge[:, signal, 0] = economic.pair_weights[ledger_rows, pair, 0]
    hedge[:, hedge_leg, 0] = economic.pair_weights[ledger_rows, pair, 1]
    returns = pd.DataFrame(
        economic.coin_open_to_open_returns[ledger_rows],
        index=clock,
        columns=economic.symbols,
    )
    availability = pd.DataFrame(
        economic.coin_decision_available[ledger_rows],
        index=clock,
        columns=economic.symbols,
    )
    funding = pd.DataFrame(
        economic.funding_at_boundary[ledger_rows],
        index=clock,
        columns=economic.symbols,
    )
    constraints = dependencies.PortfolioConstraints(
        maximum_total_gross=2.0,
        maximum_absolute_coin_weight=2.0,
        maximum_absolute_dollar_net=2.0,
    )
    ledger = dependencies.backtest_multicoin_pair_ledger(
        pair_targets,
        hedge,
        returns,
        funding,
        transaction_cost_bps=0.0,
        coin_symbols=economic.symbols,
        availability=availability,
        constraints=constraints,
        liquidate_at_end=False,
    )
    expected = float(
        np.prod(1.0 + actions * economic.pair_target_cashflows[decision_rows, pair])
        - 1.0
    )
    observed = float(
        np.prod(1.0 + ledger.bars["gross_return"].to_numpy(dtype=float)) - 1.0
    )
    if not np.isclose(observed, expected, rtol=1e-12, atol=1e-14):
        raise ValueError(
            "multi-period stored targets do not reconcile to the audited terminal ledger"
        )
    terminal_close = ledger.coin_ledger.reset_index().loc[
        lambda value: value["timestamp"].eq(clock[-1]), "close_notional"
    ]
    if not np.allclose(terminal_close.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("multi-period identity ledger did not finish flat")
    return {
        "timestamp_start": clock[0].isoformat(),
        "terminal_timestamp": clock[-1].isoformat(),
        "pair_id": pair_id,
        "decisions": int(decision_count),
        "contains_nonzero_next_boundary_funding": True,
        "expected_compounded_action_return": expected,
        "audited_ledger_compounded_gross_return": observed,
        "absolute_error": abs(observed - expected),
        "terminal_zero_target_included": True,
        "passed": True,
    }


def prepare_economic(args: argparse.Namespace) -> int:
    contract = load_contract()
    dependencies = install_audited_dependencies(args.pairs_research_root)
    symbols, pair_ids, legs = static_axes()
    manifest_root = Path(args.acquisition_manifest).expanduser().resolve()
    acquisition_hashes = {
        "acquisition_manifest_sha256": sha256_file(
            manifest_root / "acquisition_manifest.json"
        ),
        "archive_availability_sha256": sha256_file(
            manifest_root / "archive_availability.csv"
        ),
        "artifact_hashes_sha256": sha256_file(manifest_root / "artifact_hashes.json"),
    }
    for key, digest in acquisition_hashes.items():
        if contract["data"].get(key) != digest:
            raise ValueError(f"acquisition binding differs: {key}")
    market = load_market_inputs(
        cache_root=args.cache_root,
        manifest_directory=manifest_root,
        symbols=symbols,
        start=pd.Timestamp("2021-01-01T00:00:00Z"),
        end_inclusive=pd.Timestamp("2025-06-30T23:00:00Z"),
        dependencies=dependencies,
    )
    provenance = {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        **acquisition_hashes,
        "audited_source_hashes": dict(dependencies.source_hashes),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    economic = build_economic_inputs(
        market=market,
        symbols=symbols,
        pair_ids=pair_ids,
        pair_leg_indices=legs,
        contract=contract,
        provenance=provenance,
    )
    identity = {
        "one_period": assert_one_period_ledger_identity(economic, dependencies),
        "multi_period_with_terminal_row": assert_multi_period_ledger_identity(
            economic, dependencies
        ),
    }
    save_economic_inputs(args.output, economic)
    artifact = Path(args.output).expanduser().resolve()
    manifest = {
        "schema": "paired_hourly_economic_inputs_manifest_v2",
        "status": "complete_immutable_shared_training_and_ledger_input",
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "clock_start": economic.clock[0].isoformat(),
        "clock_end": economic.clock[-1].isoformat(),
        "hourly_observations": len(economic.clock),
        "symbols": list(economic.symbols),
        "pair_ids_sha256": hashlib.sha256(
            "\n".join(economic.pair_ids).encode()
        ).hexdigest(),
        "pairs": len(economic.pair_ids),
        "pair_decision_available_cells": int(economic.pair_decision_available.sum()),
        "pair_target_available_cells": int(economic.pair_target_available.sum()),
        "terminal_exit_cells": int(economic.terminal_exit_mask.sum()),
        "funding_verified_archives_loaded": market.funding_archive_count,
        "funding_missing_archives": market.missing_funding_archive_count,
        "funding_included_in_target": True,
        "pair_target_formula": economic.metadata["pair_cashflow_formula"],
        "one_period_ledger_identity": identity["one_period"],
        "multi_period_ledger_identity": identity["multi_period_with_terminal_row"],
        "transient_open_gap_report": economic.metadata["transient_open_gap_report"],
        "audited_source_hashes": dict(dependencies.source_hashes),
        **acquisition_hashes,
        "latest_raw_month": max(market.loaded_months),
        "july_or_later_path_count": 0,
        "confirmation_rows_parsed": False,
    }
    _atomic_json(artifact.with_suffix(".manifest.json"), manifest)
    _atomic_bytes(
        artifact.with_suffix(".sha256"),
        f"{manifest['artifact_sha256']}  {artifact.name}\n".encode("ascii"),
    )
    # Reloading independently checks every stored target against primitives.
    load_economic_inputs(artifact)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _forecast_filename(representation: str, model: str, seed: str) -> str:
    return f"{representation}__{model}__seed-{seed}.npz"


def _current_forecast_source_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, path in FORECAST_SOURCE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"forecast-defining source is missing: {path}")
        observed[name] = sha256_file(path)
    return observed


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_forecast_manifest_keys(
    contract: Mapping[str, Any],
) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        keys.add((fold, "shared_endpoint", "endpoint_ridge", "none"))
        for representation in ("native_1m", "causal_15m_control"):
            for model in ("lstm", "itransformer"):
                for seed in FIXED_SEEDS:
                    keys.add((fold, representation, model, str(seed)))
                keys.add((fold, representation, f"{model}_seed_mean", "all3"))
    return keys


def _expected_forecast_identity(
    *,
    run_manifest: Mapping[str, Any],
    fold: str,
    representation: str,
    model: str,
    seed: str,
    role: str,
) -> dict[str, Any]:
    sources = run_manifest["source_hashes"]
    identity = {
        "schema": "paired_neural_forecast_v2",
        "contract_sha256": run_manifest["contract_sha256"],
        "component_sha256": run_manifest["component_cache_sha256"],
        "pair_axis_sha256": run_manifest["pair_axis_sha256"],
        "core_source_sha256": sources["paired_neural_core.py"],
        "models_source_sha256": sources["paired_neural_models.py"],
        "runner_source_sha256": sources["run_paired_neural.py"],
        "source_bundle_sha256": run_manifest["source_bundle_sha256"],
        "runtime_environment_sha256": run_manifest["runtime_environment_sha256"],
        "economic_artifact_sha256": run_manifest["economic_artifact_sha256"],
        "economic_manifest_sha256": run_manifest["economic_manifest_sha256"],
        "fold": str(fold),
        "representation": str(representation),
        "model": str(model),
        "seed": str(seed),
        "role": str(role),
    }
    if set(identity) != FORECAST_DIAGNOSTIC_IDENTITY_KEYS:
        raise AssertionError("internal forecast diagnostic identity contract differs")
    return identity


def _validate_forecast_diagnostics(
    diagnostics: object,
    *,
    expected_identity: Mapping[str, Any],
    expected_extras: Mapping[str, Any],
    source: Path | str,
) -> None:
    if set(expected_identity) != FORECAST_DIAGNOSTIC_IDENTITY_KEYS:
        raise AssertionError("caller supplied an incomplete forecast identity")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"forecast diagnostics are not a mapping: {source}")
    expected_keys = set(expected_identity) | set(expected_extras)
    if set(diagnostics) != expected_keys:
        raise ValueError(f"forecast diagnostic schema differs: {source}")
    for field, expected in {**expected_identity, **expected_extras}.items():
        if diagnostics[field] != expected:
            raise ValueError(
                f"forecast diagnostics identity differs for {field}: {source}"
            )


def load_forecasts(
    *,
    forecast_output: Path,
    components: Any,
    component_path: Path,
    economic_path: Path,
    economic_manifest_path: Path,
    contract: Mapping[str, Any],
) -> dict[tuple[str, str], Forecast]:
    root = Path(forecast_output).resolve()
    run_manifest_path = root / "run_manifest.json"
    forecast_manifest_path = root / "forecast_manifest.csv"
    if not run_manifest_path.is_file() or not forecast_manifest_path.is_file():
        raise FileNotFoundError("forecast stage manifest is incomplete")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(run_manifest, dict)
        or set(run_manifest) != FORECAST_RUN_MANIFEST_KEYS
    ):
        raise ValueError("forecast run manifest key set differs")
    if (
        run_manifest.get("schema") != "paired_neural_forecast_run_manifest_v2"
        or run_manifest.get("status")
        != "forecast_stage_v2_complete_portfolio_ledger_pending"
    ):
        raise ValueError("forecast run manifest schema or status differs")
    contract_hash = sha256_file(CONTRACT_PATH)
    component_hash = sha256_file(component_path)
    economic_artifact_hash = sha256_file(economic_path)
    economic_manifest_hash = sha256_file(economic_manifest_path)
    if run_manifest.get("contract_sha256") != contract_hash:
        raise ValueError("forecast run is not bound to the current paired contract")
    if run_manifest.get("component_cache_sha256") != component_hash:
        raise ValueError("forecast run is not bound to the supplied components")
    expected_pair_axis_hash = hashlib.sha256(
        (
            json.dumps(list(map(str, components.pair_ids)), separators=(",", ":"))
            + "\n"
        ).encode()
    ).hexdigest()
    if run_manifest.get("pair_axis_sha256") != expected_pair_axis_hash:
        raise ValueError("forecast run pair-axis hash differs")
    if int(run_manifest.get("unique_strategy_series", -1)) != 5:
        raise ValueError("forecast run does not contain the exact strategy family")
    if run_manifest.get("economic_artifact_sha256") != economic_artifact_hash:
        raise ValueError("forecast run is not bound to the supplied economic artifact")
    if run_manifest.get("economic_manifest_sha256") != economic_manifest_hash:
        raise ValueError("forecast run is not bound to the supplied economic manifest")
    recorded_sources = run_manifest.get("source_hashes")
    current_sources = _current_forecast_source_hashes()
    if not isinstance(recorded_sources, dict) or recorded_sources != current_sources:
        raise ValueError(
            "forecast run source hashes differ from the current frozen sources"
        )
    expected_bundle_hash = hashlib.sha256(_canonical_json(current_sources)).hexdigest()
    if run_manifest.get("source_bundle_sha256") != expected_bundle_hash:
        raise ValueError("forecast run source-bundle hash differs")
    runtime_environment = run_manifest.get("runtime_environment")
    if (
        not isinstance(runtime_environment, dict)
        or set(runtime_environment) != RUNTIME_ENVIRONMENT_KEYS
    ):
        raise ValueError("forecast run runtime-environment schema differs")
    expected_environment_hash = hashlib.sha256(
        _canonical_json(runtime_environment)
    ).hexdigest()
    if run_manifest.get("runtime_environment_sha256") != expected_environment_hash:
        raise ValueError("forecast run runtime-environment hash differs")
    if not _is_sha256(run_manifest.get("discarded_benchmark_receipt_sha256")):
        raise ValueError("forecast run benchmark receipt hash is invalid")
    if (
        run_manifest.get("fixed_seeds") != list(FIXED_SEEDS)
        or run_manifest.get("seed_selection") is not False
    ):
        raise ValueError("forecast run seed policy differs")
    if (
        int(run_manifest.get("fold_count", -1)) != 6
        or int(run_manifest.get("forecast_artifact_count", -1)) != 102
        or int(run_manifest.get("fold_metric_rows", -1)) != 30
        or int(run_manifest.get("seed_metric_rows", -1)) != 72
        or run_manifest.get("confirmation_parsed") is not False
    ):
        raise ValueError("forecast run family counts or firewall differ")
    if (
        run_manifest.get("portfolio_backend_required")
        != "the existing audited multicoin pair ledger"
    ):
        raise ValueError("forecast run requires a different portfolio backend")
    if run_manifest.get("portfolio_settings") != contract["portfolio"]:
        raise ValueError("forecast run portfolio settings differ from the contract")
    summary_hashes = run_manifest.get("summary_sha256")
    expected_summaries = {
        "seed_metrics.csv",
        "fold_metrics.csv",
        "ic_by_decision.csv",
        "forecast_manifest.csv",
        "attempt_ledger.csv",
    }
    if (
        not isinstance(summary_hashes, dict)
        or set(summary_hashes) != expected_summaries
    ):
        raise ValueError("forecast summary hash family differs")
    for name, digest in summary_hashes.items():
        if not _is_sha256(digest):
            raise ValueError(f"forecast summary hash is invalid: {name}")
        if sha256_file(root / name) != digest:
            raise ValueError(f"forecast summary differs from run manifest: {name}")
    manifest = pd.read_csv(forecast_manifest_path, dtype=str, keep_default_na=False)
    keys = ["fold", "representation", "model", "seed"]
    if set(manifest.columns) != set(keys + ["path", "sha256"]):
        raise ValueError("forecast manifest schema differs")
    if manifest.duplicated(keys).any():
        raise ValueError("forecast manifest contains duplicate keys")
    manifest = manifest.set_index(keys)
    expected_manifest_keys = _expected_forecast_manifest_keys(contract)
    observed_manifest_keys = set(map(tuple, manifest.index.tolist()))
    if observed_manifest_keys != expected_manifest_keys or len(manifest) != 102:
        raise ValueError("forecast manifest is not the exact governed artifact family")
    if manifest["path"].duplicated().any():
        raise ValueError("forecast manifest reuses an artifact path")
    if not manifest["sha256"].map(_is_sha256).all():
        raise ValueError("forecast manifest contains an invalid artifact hash")
    component_lookup = pd.Series(
        np.arange(len(components.clock), dtype=np.int64), index=components.clock
    )
    loaded: dict[tuple[str, str], Forecast] = {}
    expected_folds = tuple(str(row["name"]) for row in contract["folds"])
    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        expected_clock = pd.date_range(
            _utc(fold_spec["test_start"]),
            _utc(fold_spec["test_end_exclusive"]) - pd.Timedelta(hours=1),
            freq="1h",
        )
        locations = component_lookup.reindex(expected_clock)
        if locations.isna().any():
            raise ValueError(f"component cache does not cover {fold}")
        feature_mask = components.feature_available[locations.to_numpy(dtype=np.int64)]
        for series_id, (series, representation, model, seed) in SERIES.items():
            key = (fold, representation, model, seed)
            if key not in manifest.index:
                raise ValueError(f"forecast manifest lacks {series_id}/{fold}")
            row = manifest.loc[key]
            path = (root / str(row["path"])).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "forecast manifest path escapes its output root"
                ) from error
            if not path.is_file() or sha256_file(path) != str(row["sha256"]):
                raise ValueError(f"forecast file differs from manifest: {path}")
            try:
                with np.load(path, allow_pickle=False) as stored:
                    required = {
                        "clock_ns",
                        "predictions",
                        "pair_targets",
                        "diagnostics_json",
                    }
                    if set(stored.files) != required:
                        raise ValueError(f"forecast NPZ members differ: {path}")
                    clock = pd.to_datetime(stored["clock_ns"], utc=True)
                    predictions = np.asarray(stored["predictions"], dtype=np.float32)
                    pair_targets = np.asarray(stored["pair_targets"], dtype=np.float32)
                    diagnostics = json.loads(str(stored["diagnostics_json"].item()))
            except (OSError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid forecast artifact: {path}") from error
            shape = (len(expected_clock), len(components.pair_ids))
            if (
                not clock.equals(expected_clock)
                or predictions.shape != shape
                or pair_targets.shape != shape
            ):
                raise ValueError(f"forecast axes differ: {series_id}/{fold}")
            if np.isinf(predictions).any() or not np.array_equal(
                np.isfinite(predictions), feature_mask
            ):
                raise ValueError(
                    f"forecast finite/missing mask differs: {series_id}/{fold}"
                )
            if not np.isfinite(pair_targets).all():
                raise ValueError(
                    f"forecast pair targets are nonfinite: {series_id}/{fold}"
                )
            if np.any((np.abs(pair_targets) > 0.0) & ~feature_mask):
                raise ValueError(
                    f"forecast trades an ineligible pair: {series_id}/{fold}"
                )
            role = (
                "shared_endpoint_only_baseline"
                if series_id == "B00"
                else "fixed_three_seed_arithmetic_mean"
            )
            expected_identity = _expected_forecast_identity(
                run_manifest=run_manifest,
                fold=fold,
                representation=representation,
                model=model,
                seed=seed,
                role=role,
            )
            expected_extras = (
                {}
                if series_id == "B00"
                else {
                    "seeds": list(FIXED_SEEDS),
                    "aggregation": "arithmetic_mean_never_best_seed",
                }
            )
            _validate_forecast_diagnostics(
                diagnostics,
                expected_identity=expected_identity,
                expected_extras=expected_extras,
                source=path,
            )
            rebuilt = cycle.rank_pair_targets(
                predictions,
                feature_mask,
                long_pairs=int(contract["portfolio"]["long_pairs"]),
                short_pairs=int(contract["portfolio"]["short_pairs"]),
                absolute_pair_target=float(
                    contract["portfolio"]["absolute_pair_target"]
                ),
            ).astype(np.float32)
            if not np.array_equal(pair_targets, rebuilt):
                raise ValueError(
                    f"stored pair targets do not reproduce from predictions: {series_id}/{fold}"
                )
            loaded[(series_id, fold)] = Forecast(
                series_id, series, fold, clock, pair_targets, path, str(row["sha256"])
            )
    expected = {(series_id, fold) for series_id in SERIES for fold in expected_folds}
    if set(loaded) != expected or len(loaded) != 30:
        raise AssertionError("forecast load is not the exact five-by-six family")
    return loaded


def hedge_tensor(
    components: Any, locations: np.ndarray, pair_targets: np.ndarray
) -> np.ndarray:
    weights = np.asarray(components.pair_weights[locations], dtype=float)
    if weights.shape != (len(locations), len(components.pair_ids), 2):
        raise ValueError("component hedge tensor has unexpected axes")
    active = np.abs(np.asarray(pair_targets, dtype=float)) > 0.0
    if np.any(active & ~np.isfinite(weights).all(axis=2)):
        raise ValueError("an active pair lacks finite causal hedge weights")
    tensor = np.zeros(
        (len(locations), len(components.symbols), len(components.pair_ids)),
        dtype=float,
    )
    for pair, (signal, hedge) in enumerate(
        np.asarray(components.pair_leg_indices, dtype=np.int64)
    ):
        tensor[:, signal, pair] = np.where(
            np.isfinite(weights[:, pair, 0]), weights[:, pair, 0], 0.0
        )
        tensor[:, hedge, pair] = np.where(
            np.isfinite(weights[:, pair, 1]), weights[:, pair, 1], 0.0
        )
    return tensor


def economic_hedge_tensor(
    economic: EconomicInputs, locations: np.ndarray, pair_targets: np.ndarray
) -> np.ndarray:
    weights = economic.pair_weights[locations]
    active = np.abs(np.asarray(pair_targets, dtype=float)) > 0.0
    if np.any(active & ~np.isfinite(weights).all(axis=2)):
        raise ValueError("an active pair lacks finite economic-input hedge weights")
    tensor = np.zeros(
        (len(locations), len(economic.symbols), len(economic.pair_ids)), dtype=float
    )
    for pair, (signal, hedge) in enumerate(economic.pair_leg_indices):
        tensor[:, signal, pair] = np.where(
            np.isfinite(weights[:, pair, 0]), weights[:, pair, 0], 0.0
        )
        tensor[:, hedge, pair] = np.where(
            np.isfinite(weights[:, pair, 1]), weights[:, pair, 1], 0.0
        )
    return tensor


def _asof_covariance(
    covariance_matrices: Mapping[pd.Timestamp, pd.DataFrame],
    clock: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> np.ndarray:
    effective = pd.DatetimeIndex(sorted(map(pd.Timestamp, covariance_matrices)))
    locations = effective.searchsorted(clock, side="right") - 1
    if (locations < 0).any():
        raise ValueError("covariance map lacks a causal first value")
    return np.stack(
        [
            covariance_matrices[effective[position]]
            .reindex(index=list(symbols), columns=list(symbols))
            .to_numpy(dtype=float)
            for position in locations
        ]
    )


def _daily_metrics(
    values: Iterable[float], *, turnover: float, gross_sum: float
) -> dict[str, Any]:
    daily = pd.Series(list(values), dtype=float)
    if daily.empty or not np.isfinite(daily).all():
        raise ValueError("performance metrics require finite daily returns")
    volatility = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    sharpe = (
        float(np.sqrt(365.0) * daily.mean() / volatility)
        if np.isfinite(volatility) and volatility > 0.0
        else float("nan")
    )
    compounded = float((1.0 + daily).prod())
    years = len(daily) / 365.0
    nav = (1.0 + daily).cumprod()
    peak = np.maximum.accumulate(np.concatenate(([1.0], nav.to_numpy(dtype=float))))[1:]
    return {
        "daily_observations": int(len(daily)),
        "mean_daily_return": float(daily.mean()),
        "annual_return": (
            float(compounded ** (1.0 / years) - 1.0)
            if years > 0 and compounded > 0
            else float("nan")
        ),
        "annual_volatility": float(volatility * np.sqrt(365.0)),
        "sharpe": sharpe,
        "maximum_drawdown": float(np.min(nav.to_numpy(dtype=float) / peak - 1.0)),
        "total_return": compounded - 1.0,
        "total_turnover": float(turnover),
        "turnover_per_day": float(turnover / len(daily)),
        "linearized_break_even_cost_bps": (
            float(10_000.0 * gross_sum / turnover) if turnover > 0 else float("nan")
        ),
    }


def _risk_audit(
    *,
    projection: Any,
    bars: pd.DataFrame,
    beta: pd.DataFrame,
    covariance: np.ndarray,
    constraints: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = projection.targets.astype(float)
    clock = pd.DatetimeIndex(targets.index)
    values = targets.to_numpy(dtype=float)
    beta_values = (
        beta.reindex(clock).reindex(columns=targets.columns).to_numpy(dtype=float)
    )
    long_gross = np.clip(values, 0.0, None).sum(axis=1)
    short_gross = -np.clip(values, None, 0.0).sum(axis=1)
    gross = long_gross + short_gross
    maximum_coin = np.abs(values).max(axis=1)
    dollar_net = values.sum(axis=1)
    btc_beta = np.einsum("tc,tc->t", values, beta_values)
    variance = np.einsum("ti,tij,tj->t", values, covariance, values)
    if (variance < -1e-12).any():
        raise ValueError("risk audit covariance implies negative variance")
    volatility = np.sqrt(np.maximum(variance, 0.0) * 365.0 * 24.0)
    denominator = np.square(np.abs(values)).sum(axis=1)
    effective_count = np.divide(
        np.square(gross), denominator, out=np.zeros_like(gross), where=denominator > 0.0
    )
    concentration = np.divide(
        maximum_coin, gross, out=np.zeros_like(gross), where=gross > 0.0
    )
    audit = pd.DataFrame(
        {
            "timestamp": clock,
            "long_gross": long_gross,
            "short_gross": short_gross,
            "total_gross": gross,
            "maximum_absolute_coin_weight": maximum_coin,
            "dollar_net": dollar_net,
            "btc_beta": btc_beta,
            "ex_ante_annualized_volatility": volatility,
            "volatility_scaling_factor": projection.diagnostics[
                "volatility_scale"
            ].to_numpy(dtype=float),
            "projection_scaling_factor": projection.diagnostics[
                "projection_scale"
            ].to_numpy(dtype=float),
            "disabled_pair_count": projection.diagnostics[
                "disabled_pair_count"
            ].to_numpy(dtype=float),
            "effective_coin_count": effective_count,
            "weight_concentration": concentration,
            "realized_gross_exposure": bars["gross_exposure"].to_numpy(dtype=float),
            "realized_net_exposure": bars["net_exposure"].to_numpy(dtype=float),
            "turnover": bars["turnover"].to_numpy(dtype=float),
            "transaction_cost": bars["transaction_cost"].to_numpy(dtype=float),
        }
    )
    audit["gross_violation"] = (
        audit["total_gross"] > constraints.maximum_total_gross + TOLERANCE
    )
    audit["coin_violation"] = (
        audit["maximum_absolute_coin_weight"]
        > constraints.maximum_absolute_coin_weight + TOLERANCE
    )
    audit["net_violation"] = (
        audit["dollar_net"].abs() > constraints.maximum_absolute_dollar_net + TOLERANCE
    )
    audit["beta_violation"] = (
        audit["btc_beta"].abs() > constraints.maximum_absolute_btc_beta + TOLERANCE
    )
    audit["volatility_violation"] = (
        audit["ex_ante_annualized_volatility"]
        > constraints.target_annualized_volatility + TOLERANCE
    )
    violation_columns = [column for column in audit if column.endswith("_violation")]
    audit["any_constraint_violation"] = audit[violation_columns].any(axis=1)
    summary = {
        "hourly_observations": len(audit),
        "maximum_total_gross": float(audit["total_gross"].max()),
        "maximum_absolute_coin_weight": float(
            audit["maximum_absolute_coin_weight"].max()
        ),
        "maximum_absolute_dollar_net": float(audit["dollar_net"].abs().max()),
        "maximum_absolute_btc_beta": float(audit["btc_beta"].abs().max()),
        "maximum_ex_ante_annualized_volatility": float(
            audit["ex_ante_annualized_volatility"].max()
        ),
        "mean_long_gross": float(audit["long_gross"].mean()),
        "mean_short_gross": float(audit["short_gross"].mean()),
        "mean_effective_coin_count": float(audit["effective_coin_count"].mean()),
        "median_effective_coin_count": float(audit["effective_coin_count"].median()),
        "mean_weight_concentration": float(audit["weight_concentration"].mean()),
        "maximum_weight_concentration": float(audit["weight_concentration"].max()),
        "all_constraint_violation_count": int(audit["any_constraint_violation"].sum()),
    }
    return audit, summary


def _validate_ledger_attribution(bars: pd.DataFrame, cost_bps: float) -> None:
    required = {
        "price_return",
        "funding_return",
        "gross_return",
        "net_return",
        "turnover",
        "transaction_cost",
    }
    if (
        not required.issubset(bars.columns)
        or not np.isfinite(bars[list(required)].to_numpy(dtype=float)).all()
    ):
        raise ValueError("audited ledger return attribution is incomplete")
    if not np.allclose(
        bars["gross_return"],
        bars["price_return"] + bars["funding_return"],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("gross return does not reconcile")
    if not np.allclose(
        bars["net_return"],
        bars["gross_return"] - bars["transaction_cost"],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("net return does not reconcile")
    if not np.allclose(
        bars["transaction_cost"],
        float(cost_bps) / 10_000.0 * bars["turnover"],
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("transaction cost does not equal one-way bps times turnover")


def economic_daily_attribution(
    ledger: Any, *, terminal_timestamp: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Compound the zero-target terminal row into the preceding economic day."""

    bars = ledger.bars.copy()
    clock = pd.DatetimeIndex(bars.index).tz_convert("UTC")
    terminal = _utc(terminal_timestamp)
    if clock[-1] != terminal:
        raise ValueError("ledger does not end on the declared terminal boundary")
    keys_array = clock.normalize().to_numpy(copy=True)
    keys_array[-1] = keys_array[-1] - np.timedelta64(1, "D")
    keys = pd.DatetimeIndex(keys_array).tz_convert("UTC")
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
            "price_return_contribution": bars["price_return"].to_numpy(dtype=float)
            * prior_growth,
            "funding_return_contribution": bars["funding_return"].to_numpy(dtype=float)
            * prior_growth,
            "transaction_cost_contribution": -bars["transaction_cost"].to_numpy(
                dtype=float
            )
            * prior_growth,
        },
        index=clock,
    )
    hourly["net_return_contribution"] = hourly.sum(axis=1)
    daily = hourly.groupby(keys).sum()
    daily["net_return"] = (1.0 + bars["net_return"]).groupby(keys).prod() - 1.0
    daily["gross_return"] = (1.0 + bars["gross_return"]).groupby(keys).prod() - 1.0
    daily["turnover"] = bars["turnover"].groupby(keys).sum()
    daily["trading_cost"] = bars["transaction_cost"].groupby(keys).sum()
    if not np.allclose(
        daily["net_return"], daily["net_return_contribution"], rtol=1e-12, atol=1e-14
    ):
        raise ValueError("economic daily return attribution does not reconcile")
    coin = ledger.coin_ledger.reset_index().copy()
    locations = clock.get_indexer(pd.DatetimeIndex(coin["timestamp"]))
    if (locations < 0).any():
        raise ValueError("coin ledger timestamp lies outside economic clock")
    coin["date"] = keys.take(locations).to_numpy()
    coin["price_return_contribution"] = (
        coin["price_return"].to_numpy(dtype=float) * prior_growth[locations]
    )
    coin["funding_return_contribution"] = (
        coin["funding_return"].to_numpy(dtype=float) * prior_growth[locations]
    )
    coin["transaction_cost_contribution"] = -(
        coin["transaction_cost"].to_numpy(dtype=float) * prior_growth[locations]
    )
    coin["net_return_contribution"] = coin[
        [
            "price_return_contribution",
            "funding_return_contribution",
            "transaction_cost_contribution",
        ]
    ].sum(axis=1)
    daily_coin = coin.groupby(["date", "coin"], as_index=False)[
        [
            "price_return_contribution",
            "funding_return_contribution",
            "transaction_cost_contribution",
            "net_return_contribution",
        ]
    ].sum()
    reconciled = daily_coin.groupby("date")["net_return_contribution"].sum()
    if not np.allclose(
        reconciled.reindex(daily.index), daily["net_return"], rtol=1e-12, atol=1e-14
    ):
        raise ValueError("economic daily coin attribution does not reconcile")
    return daily, daily_coin, keys


def moving_block_bootstrap(
    values: pd.DataFrame,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    block_days: int = BOOTSTRAP_DAYS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Year-stratified, non-circular seven-day bootstrap for Q2--Q4 paths."""

    required = {"fold", "date", "difference"}
    if not required.issubset(values.columns):
        raise ValueError("paired bootstrap columns are incomplete")
    data = values[list(required)].copy()
    data["date"] = pd.to_datetime(data["date"], utc=True).dt.normalize()
    data["difference"] = pd.to_numeric(data["difference"], errors="raise")
    if data.empty or not np.isfinite(data["difference"]).all():
        raise ValueError("paired bootstrap needs finite differences")
    if data.duplicated(["fold", "date"]).any():
        raise ValueError("paired bootstrap keys are duplicated")
    data["year"] = data["fold"].astype(str).str.slice(0, 4)
    strata: list[tuple[str, np.ndarray]] = []
    for year, chosen in data.sort_values(["date"]).groupby("year", sort=True):
        observed = pd.DatetimeIndex(chosen["date"])
        expected = pd.date_range(f"{year}-04-01", f"{year}-12-31", freq="1D", tz="UTC")
        if not observed.equals(expected):
            raise ValueError(f"paired bootstrap {year} is not the exact Q2-Q4 calendar")
        strata.append((str(year), chosen["difference"].to_numpy(dtype=float)))
    if tuple(year for year, _ in strata) != ("2023", "2024"):
        raise ValueError("paired bootstrap requires the two declared year strata")
    rng = np.random.default_rng(int(seed))
    sampled = np.empty(int(replications), dtype=float)
    offsets = np.arange(int(block_days), dtype=np.int64)
    total_count = sum(len(array) for _, array in strata)
    for draw in range(int(replications)):
        total = 0.0
        for _year, array in strata:
            blocks = math.ceil(len(array) / int(block_days))
            starts = rng.integers(0, len(array) - int(block_days) + 1, size=blocks)
            locations = starts[:, None] + offsets[None, :]
            total += float(array[locations.ravel()[: len(array)]].sum())
        sampled[draw] = total / total_count
    observed_mean = float(sum(array.sum() for _, array in strata) / total_count)
    return {
        "method": "year_stratified_non_circular_moving_block_bootstrap",
        "observations": int(total_count),
        "block_days": int(block_days),
        "replications": int(replications),
        "seed": int(seed),
        "observed_mean_daily_difference": observed_mean,
        "annualized_mean_difference": float(365.0 * observed_mean),
        "bootstrap_mean_daily_difference": float(sampled.mean()),
        "bootstrap_standard_error": float(sampled.std(ddof=1)),
        "one_sided_90pct_lower_bound_daily": float(np.quantile(sampled, 0.10)),
        "one_sided_90pct_lower_bound_annualized": float(
            365.0 * np.quantile(sampled, 0.10)
        ),
    }


def _one_metric(
    metrics: pd.DataFrame, *, series_id: str, cost_bps: float, scope: str = "aggregate"
) -> pd.Series:
    chosen = metrics.loc[
        metrics["series_id"].eq(series_id)
        & metrics["scope"].eq(scope)
        & np.isclose(metrics["cost_bps"].astype(float), cost_bps, rtol=0.0, atol=1e-12)
    ]
    if len(chosen) != 1:
        raise ValueError(f"metric row is not unique: {series_id}/{cost_bps}/{scope}")
    return chosen.iloc[0]


def evaluate(args: argparse.Namespace) -> int:
    contract = load_contract()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("portfolio output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    dependencies = install_audited_dependencies(args.pairs_research_root)
    components, component_metadata = load_components(args.components)
    if components.clock.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError("component cache reaches the confirmation firewall")
    economic = load_economic_inputs(args.economic_inputs, args.economic_manifest)
    if (
        components.symbols != economic.symbols
        or components.pair_ids != economic.pair_ids
        or not np.array_equal(components.pair_leg_indices, economic.pair_leg_indices)
        or not components.clock.equals(economic.clock)
    ):
        raise ValueError("component and economic-input axes differ")
    if not np.array_equal(
        np.asarray(components.pair_weights, dtype=np.float64),
        economic.pair_weights,
        equal_nan=True,
    ):
        raise ValueError("component hedge weights differ from the economic artifact")
    if not np.array_equal(
        np.asarray(components.target_cashflows, dtype=np.float64),
        economic.pair_target_cashflows,
        equal_nan=True,
    ):
        raise ValueError("component training labels differ from the economic artifact")
    if not np.array_equal(
        np.asarray(components.target_available, dtype=bool),
        economic.pair_target_available,
    ):
        raise ValueError("component target mask differs from the economic artifact")
    component_provenance = component_metadata.get("provenance", {})
    if not isinstance(component_provenance, dict) or component_provenance.get(
        "economic_artifact_sha256"
    ) != sha256_file(args.economic_inputs):
        raise ValueError("component cache is not content-bound to the economic input")
    forecasts = load_forecasts(
        forecast_output=args.forecast_output,
        components=components,
        component_path=Path(args.components),
        economic_path=Path(args.economic_inputs),
        economic_manifest_path=(
            Path(args.economic_manifest)
            if args.economic_manifest is not None
            else Path(args.economic_inputs).with_suffix(".manifest.json")
        ),
        contract=contract,
    )
    first_test = min(_utc(row["test_start"]) for row in contract["folds"])
    final_test_end = max(_utc(row["test_end_exclusive"]) for row in contract["folds"])
    economic_opens = pd.DataFrame(
        economic.coin_open_prices,
        index=economic.clock,
        columns=economic.symbols,
    )
    risk = dependencies.estimate_daily_risk_inputs(
        economic_opens,
        start=first_test,
        end_exclusive=final_test_end,
        lookback_days=RISK_LOOKBACK_DAYS,
        minimum_complete_observations=RISK_MINIMUM_COMPLETE_HOURS,
    )
    portfolio = contract["portfolio"]
    constraints = dependencies.PortfolioConstraints(
        maximum_total_gross=float(portfolio["maximum_total_gross"]),
        maximum_absolute_coin_weight=float(portfolio["maximum_absolute_coin_weight"]),
        maximum_absolute_dollar_net=float(portfolio["maximum_absolute_dollar_net"]),
        maximum_absolute_btc_beta=float(portfolio["maximum_absolute_btc_beta"]),
        target_annualized_volatility=float(portfolio["target_annualized_volatility"]),
    )
    component_lookup = pd.Series(
        np.arange(len(components.clock), dtype=np.int64), index=components.clock
    )
    economic_lookup = pd.Series(
        np.arange(len(economic.clock), dtype=np.int64), index=economic.clock
    )
    daily_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    hourly_risk_rows: list[pd.DataFrame] = []
    coin_rows: list[pd.DataFrame] = []
    total_ledgers = 0
    funding_unknown_flat_cells = 0
    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        exemplar = forecasts[("B00", fold)]
        decision_clock = exemplar.clock
        terminal_timestamp = _utc(fold_spec["test_end_exclusive"])
        if decision_clock[-1] + pd.Timedelta(hours=1) != terminal_timestamp:
            raise ValueError(
                "forecast decision clock does not precede terminal boundary"
            )
        clock = decision_clock.append(pd.DatetimeIndex([terminal_timestamp]))
        locations = component_lookup.reindex(decision_clock).to_numpy(dtype=np.int64)
        economic_locations = economic_lookup.reindex(clock)
        if economic_locations.isna().any():
            raise ValueError(f"economic input does not cover {fold}")
        economic_locations = economic_locations.to_numpy(dtype=np.int64)
        returns = pd.DataFrame(
            economic.coin_open_to_open_returns[economic_locations],
            index=clock,
            columns=economic.symbols,
        )
        availability = pd.DataFrame(
            economic.coin_decision_available[economic_locations],
            index=clock,
            columns=economic.symbols,
        )
        funding = pd.DataFrame(
            economic.funding_at_boundary[economic_locations],
            index=clock,
            columns=economic.symbols,
        )
        beta = risk.beta_frame(clock).reindex(columns=list(components.symbols))
        covariance = _asof_covariance(
            risk.covariance_matrices, clock, components.symbols
        )
        for series_id in SERIES:
            forecast = forecasts[(series_id, fold)]
            if not forecast.clock.equals(decision_clock):
                raise ValueError("series clocks differ inside a fold")
            decision_pair_targets = pd.DataFrame(
                forecast.pair_targets,
                index=decision_clock,
                columns=components.pair_ids,
                dtype=float,
            )
            pair_targets = decision_pair_targets.reindex(clock, fill_value=0.0)
            if not np.array_equal(
                pair_targets.iloc[-1].to_numpy(dtype=float),
                np.zeros(len(components.pair_ids)),
            ):
                raise AssertionError("terminal pair target is not zero")
            if np.any(
                (np.abs(forecast.pair_targets) > 0.0)
                & ~economic.pair_decision_available[economic_locations[:-1]]
            ):
                raise ValueError("forecast trades outside causal economic availability")
            hedge = economic_hedge_tensor(
                economic, economic_locations, pair_targets.to_numpy(dtype=float)
            )
            for cost_bps in map(float, portfolio["cost_bps"]):
                ledger = dependencies.backtest_multicoin_pair_ledger(
                    pair_targets,
                    hedge,
                    returns,
                    funding,
                    transaction_cost_bps=cost_bps,
                    coin_symbols=tuple(components.symbols),
                    availability=availability,
                    constraints=constraints,
                    btc_beta_loadings=beta,
                    covariance_matrices=risk.covariance_matrices,
                    liquidate_at_end=False,
                )
                total_ledgers += 1
                bars = ledger.bars.copy()
                if not bars.index.equals(clock):
                    raise ValueError("audited ledger changed the hourly clock")
                _validate_ledger_attribution(bars, cost_bps)
                coin_ledger = ledger.coin_ledger.reset_index()
                terminal = coin_ledger.loc[
                    coin_ledger["timestamp"].eq(clock[-1]), "close_notional"
                ]
                if not np.allclose(
                    terminal.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        "audited ledger did not liquidate every terminal coin"
                    )
                unknown = coin_ledger["funding_rate_at_boundary"].isna()
                carried = coin_ledger["carried_notional"].abs() > 1e-15
                if (unknown & carried).any():
                    raise ValueError(
                        "unknown funding was encountered on a carried position"
                    )
                funding_unknown_flat_cells += int((unknown & ~carried).sum())
                daily_table, daily_coin, dates = economic_daily_attribution(
                    ledger, terminal_timestamp=terminal_timestamp
                )
                for date in daily_table.index:
                    daily_rows.append(
                        {
                            "series_id": series_id,
                            "series": forecast.series,
                            "fold": fold,
                            "date": date,
                            "cost_bps": cost_bps,
                            "gross_return": float(
                                daily_table.loc[date, "gross_return"]
                            ),
                            "net_return": float(daily_table.loc[date, "net_return"]),
                            "price_return_contribution": float(
                                daily_table.loc[date, "price_return_contribution"]
                            ),
                            "funding_return_contribution": float(
                                daily_table.loc[date, "funding_return_contribution"]
                            ),
                            "transaction_cost_contribution": float(
                                daily_table.loc[date, "transaction_cost_contribution"]
                            ),
                            "turnover": float(daily_table.loc[date, "turnover"]),
                            "trading_cost": float(
                                daily_table.loc[date, "trading_cost"]
                            ),
                            "is_terminal_day": bool(date == daily_table.index[-1]),
                        }
                    )
                fold_metric_rows.append(
                    {
                        "scope": "fold",
                        "series_id": series_id,
                        "series": forecast.series,
                        "fold": fold,
                        "cost_bps": cost_bps,
                        **_daily_metrics(
                            daily_table["net_return"].to_numpy(dtype=float),
                            turnover=float(bars["turnover"].sum()),
                            gross_sum=float(bars["gross_return"].sum()),
                        ),
                        "total_funding_return": float(bars["funding_return"].sum()),
                        "total_price_return": float(bars["price_return"].sum()),
                        "total_transaction_cost": float(bars["transaction_cost"].sum()),
                    }
                )
                audit, audit_summary = _risk_audit(
                    projection=ledger.projection,
                    bars=bars,
                    beta=beta,
                    covariance=covariance,
                    constraints=constraints,
                )
                risk_rows.append(
                    {
                        "series_id": series_id,
                        "series": forecast.series,
                        "fold": fold,
                        "cost_bps": cost_bps,
                        **audit_summary,
                    }
                )
                if np.isclose(
                    cost_bps, float(portfolio["primary_cost_bps"]), rtol=0.0, atol=1e-12
                ):
                    audit.insert(0, "fold", fold)
                    audit.insert(0, "series", forecast.series)
                    audit.insert(0, "series_id", series_id)
                    audit.insert(3, "cost_bps", cost_bps)
                    hourly_risk_rows.append(audit)
                coin = daily_coin.copy()
                coin.insert(0, "cost_bps", cost_bps)
                coin.insert(0, "fold", fold)
                coin.insert(0, "series", forecast.series)
                coin.insert(0, "series_id", series_id)
                coin_rows.append(coin)
    if total_ledgers != 5 * 6 * 4:
        raise AssertionError(
            "portfolio evaluation did not run the exact 120-ledger family"
        )

    daily = (
        pd.DataFrame(daily_rows)
        .sort_values(["series_id", "cost_bps", "fold", "date"])
        .reset_index(drop=True)
    )
    if daily.duplicated(["series_id", "fold", "date", "cost_bps"]).any():
        raise ValueError("portfolio daily-return keys are duplicated")
    metrics_rows = list(fold_metric_rows)
    for series_id, (series, *_rest) in SERIES.items():
        for cost_bps in map(float, portfolio["cost_bps"]):
            chosen = daily.loc[
                daily["series_id"].eq(series_id)
                & np.isclose(daily["cost_bps"], cost_bps, rtol=0.0, atol=1e-12)
            ].sort_values(["fold", "date"])
            metrics_rows.append(
                {
                    "scope": "aggregate",
                    "series_id": series_id,
                    "series": series,
                    "fold": "ALL",
                    "cost_bps": cost_bps,
                    **_daily_metrics(
                        chosen["net_return"],
                        turnover=float(chosen["turnover"].sum()),
                        gross_sum=float(
                            pd.DataFrame(fold_metric_rows)
                            .loc[
                                pd.DataFrame(fold_metric_rows)["series_id"].eq(
                                    series_id
                                )
                                & np.isclose(
                                    pd.DataFrame(fold_metric_rows)["cost_bps"].astype(
                                        float
                                    ),
                                    cost_bps,
                                    rtol=0.0,
                                    atol=1e-12,
                                ),
                                "total_price_return",
                            ]
                            .sum()
                            + pd.DataFrame(fold_metric_rows)
                            .loc[
                                pd.DataFrame(fold_metric_rows)["series_id"].eq(
                                    series_id
                                )
                                & np.isclose(
                                    pd.DataFrame(fold_metric_rows)["cost_bps"].astype(
                                        float
                                    ),
                                    cost_bps,
                                    rtol=0.0,
                                    atol=1e-12,
                                ),
                                "total_funding_return",
                            ]
                            .sum()
                        ),
                    ),
                    "total_funding_return": float(
                        chosen["funding_return_contribution"].sum()
                    ),
                    "total_price_return": float(
                        chosen["price_return_contribution"].sum()
                    ),
                    "total_transaction_cost": float(chosen["trading_cost"].sum()),
                }
            )
    metrics = (
        pd.DataFrame(metrics_rows)
        .sort_values(["scope", "series_id", "cost_bps", "fold"])
        .reset_index(drop=True)
    )
    risk_summary = (
        pd.DataFrame(risk_rows)
        .sort_values(["series_id", "cost_bps", "fold"])
        .reset_index(drop=True)
    )
    hourly_risk = pd.concat(hourly_risk_rows, ignore_index=True)
    coin_contributions = pd.concat(coin_rows, ignore_index=True)

    paired_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for contrast, (native_id, control_id) in CONTRASTS.items():
        for cost_bps in map(float, portfolio["cost_bps"]):
            native = daily.loc[
                daily["series_id"].eq(native_id)
                & np.isclose(daily["cost_bps"], cost_bps, rtol=0.0, atol=1e-12)
            ].set_index(["fold", "date"])
            control = daily.loc[
                daily["series_id"].eq(control_id)
                & np.isclose(daily["cost_bps"], cost_bps, rtol=0.0, atol=1e-12)
            ].set_index(["fold", "date"])
            if not native.index.equals(control.index):
                raise ValueError(f"paired daily clocks differ: {contrast}/{cost_bps}")
            difference = native["net_return"] - control["net_return"]
            frame = difference.rename("difference").reset_index()
            for row in frame.itertuples(index=False):
                paired_rows.append(
                    {
                        "contrast": contrast,
                        "native_series_id": native_id,
                        "control_series_id": control_id,
                        "cost_bps": cost_bps,
                        "fold": row.fold,
                        "date": row.date,
                        "native_net_return": float(
                            native.loc[(row.fold, row.date), "net_return"]
                        ),
                        "control_net_return": float(
                            control.loc[(row.fold, row.date), "net_return"]
                        ),
                        "difference": float(row.difference),
                    }
                )
            bootstrap_rows.append(
                {
                    "contrast": contrast,
                    "native_series_id": native_id,
                    "control_series_id": control_id,
                    "cost_bps": cost_bps,
                    **moving_block_bootstrap(frame),
                }
            )
    paired = (
        pd.DataFrame(paired_rows)
        .sort_values(["contrast", "cost_bps", "fold", "date"])
        .reset_index(drop=True)
    )
    bootstraps = (
        pd.DataFrame(bootstrap_rows)
        .sort_values(["contrast", "cost_bps"])
        .reset_index(drop=True)
    )

    primary_cost = float(portfolio["primary_cost_bps"])
    stress_cost = float(portfolio["stress_cost_bps"])
    native_primary = _one_metric(metrics, series_id="N02", cost_bps=primary_cost)
    control_primary = _one_metric(metrics, series_id="C02", cost_bps=primary_cost)
    native_stress = _one_metric(metrics, series_id="N02", cost_bps=stress_cost)
    primary_bootstrap = bootstraps.loc[
        bootstraps["contrast"].eq("primary_itransformer_native_minus_control")
        & np.isclose(bootstraps["cost_bps"], primary_cost, rtol=0.0, atol=1e-12)
    ].iloc[0]
    ic = pd.read_csv(Path(args.forecast_output) / "fold_metrics.csv")
    native_ic = ic.loc[
        ic["representation"].eq("native_1m") & ic["model"].eq("itransformer")
    ].set_index("fold")["mean_ic"]
    control_ic = ic.loc[
        ic["representation"].eq("causal_15m_control") & ic["model"].eq("itransformer")
    ].set_index("fold")["mean_ic"]
    ic_delta = native_ic - control_ic
    if len(ic_delta) != 6 or ic_delta.isna().any():
        raise ValueError("primary fold IC contrast is incomplete")
    primary_difference = paired.loc[
        paired["contrast"].eq("primary_itransformer_native_minus_control")
        & np.isclose(paired["cost_bps"], primary_cost, rtol=0.0, atol=1e-12)
    ]
    fold_advantage = primary_difference.groupby("fold")["difference"].mean()
    best_fold = str(fold_advantage.idxmax())
    without_best = primary_difference.loc[
        ~primary_difference["fold"].eq(best_fold), "difference"
    ]
    primary_coin = (
        coin_contributions.loc[
            coin_contributions["series_id"].isin(("N02", "C02"))
            & np.isclose(
                coin_contributions["cost_bps"], primary_cost, rtol=0.0, atol=1e-12
            )
        ]
        .groupby(["series_id", "coin"])["net_return_contribution"]
        .sum()
        .unstack("series_id")
    )
    coin_advantage = primary_coin["N02"] - primary_coin["C02"]
    coin_denominator = float(coin_advantage.abs().sum())
    maximum_coin_fraction = (
        float(coin_advantage.abs().max() / coin_denominator)
        if coin_denominator > 0.0
        else float("nan")
    )
    violations = int(risk_summary["all_constraint_violation_count"].sum())
    gates = {
        "schema": "paired_neural_portfolio_gate_report_v1",
        "status": "development_comparison_only",
        "primary_contrast": "N02_native_1m_itransformer_minus_C02_causal_15m_itransformer",
        "gates": {
            "ic_delta_positive_in_at_least_four_folds": {
                "passed": int((ic_delta > 0.0).sum()) >= 4,
                "positive_folds": int((ic_delta > 0.0).sum()),
                "required": 4,
                "fold_differences": {
                    str(key): float(value) for key, value in ic_delta.items()
                },
            },
            "paired_daily_return_bootstrap": {
                "passed": float(primary_bootstrap["one_sided_90pct_lower_bound_daily"])
                > 0.0,
                "one_sided_90pct_lower_bound_daily": float(
                    primary_bootstrap["one_sided_90pct_lower_bound_daily"]
                ),
                "observed_mean_daily_difference": float(
                    primary_bootstrap["observed_mean_daily_difference"]
                ),
            },
            "sharpe_improvement_at_least_0_15": {
                "passed": float(native_primary["sharpe"] - control_primary["sharpe"])
                >= 0.15,
                "native_sharpe": float(native_primary["sharpe"]),
                "control_sharpe": float(control_primary["sharpe"]),
                "difference": float(
                    native_primary["sharpe"] - control_primary["sharpe"]
                ),
                "required": 0.15,
            },
            "stress_and_break_even": {
                "passed": float(native_stress["sharpe"]) > 0.0
                and float(native_stress["linearized_break_even_cost_bps"]) >= 1.5,
                "native_sharpe_at_1_5bp": float(native_stress["sharpe"]),
                "linearized_break_even_cost_bps": float(
                    native_stress["linearized_break_even_cost_bps"]
                ),
            },
            "zero_constraint_violations": {
                "passed": violations == 0,
                "violation_count_all_120_ledgers": violations,
            },
            "not_one_quarter_or_coin": {
                "passed": float(without_best.mean()) > 0.0
                and np.isfinite(maximum_coin_fraction)
                and maximum_coin_fraction <= 0.5,
                "removed_best_fold": best_fold,
                "mean_daily_advantage_without_best_fold": float(without_best.mean()),
                "maximum_single_coin_absolute_advantage_fraction": maximum_coin_fraction,
                "maximum_allowed_coin_fraction": 0.5,
            },
        },
        "supporting_lstm_cannot_replace_primary": True,
        "sample_role": "exploratory_development_comparison",
        "confirmation_authorized": False,
        "confirmation_rows_parsed": False,
    }
    gates["primary_all_pass"] = all(
        bool(value["passed"]) for value in gates["gates"].values()
    )

    _atomic_csv(output / "portfolio_daily_returns.csv", daily)
    _atomic_csv(output / "portfolio_metrics.csv", metrics)
    _atomic_csv(output / "paired_differences.csv", paired)
    _atomic_csv(output / "paired_bootstrap.csv", bootstraps)
    _atomic_csv(output / "risk_audit_summary.csv", risk_summary)
    _atomic_csv(output / "coin_contributions.csv", coin_contributions)
    hourly_path = output / "hourly_constraint_audit.parquet"
    hourly_risk.to_parquet(hourly_path, index=False)
    _atomic_json(output / "gate_report.json", gates)
    output_files = (
        "portfolio_daily_returns.csv",
        "portfolio_metrics.csv",
        "paired_differences.csv",
        "paired_bootstrap.csv",
        "risk_audit_summary.csv",
        "coin_contributions.csv",
        "hourly_constraint_audit.parquet",
        "gate_report.json",
    )
    receipt = {
        "schema": "paired_neural_portfolio_completion_receipt_v1",
        "status": "complete_exact_five_series_joint_evaluation",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "components": {
            "path": str(Path(args.components).resolve()),
            "sha256": sha256_file(args.components),
            "metadata_schema": component_metadata.get("schema"),
        },
        "economic_inputs": {
            "path": str(Path(args.economic_inputs).resolve()),
            "sha256": sha256_file(args.economic_inputs),
            "manifest_sha256": sha256_file(
                args.economic_manifest
                if args.economic_manifest is not None
                else Path(args.economic_inputs).with_suffix(".manifest.json")
            ),
            "schema": economic.metadata.get("schema"),
        },
        "forecast_output": str(Path(args.forecast_output).resolve()),
        "forecast_hashes": {
            f"{series_id}/{fold}": forecast.sha256
            for (series_id, fold), forecast in sorted(forecasts.items())
        },
        "audited_dependency_source_hashes": dict(dependencies.source_hashes),
        "family": {
            "series": list(SERIES),
            "folds": [str(row["name"]) for row in contract["folds"]],
            "cost_bps": list(map(float, portfolio["cost_bps"])),
            "ledger_count": total_ledgers,
            "selector_used": False,
        },
        "execution": {
            "decision_frequency": "hourly",
            "return": "one-hour open-to-next-open simple return with audited terminal in-hour close mark",
            "position_accounting": "fixed USD-M coin quantities between target changes",
            "pair_targets_netted_to_coins_before_constraints": True,
            "terminal_liquidation_included": True,
        },
        "funding": {
            "included": True,
            "source": "manifest-allowlisted verified Binance fundingRate archives",
            "verified_archives_loaded": int(
                json.loads(
                    Path(
                        args.economic_manifest
                        if args.economic_manifest is not None
                        else Path(args.economic_inputs).with_suffix(".manifest.json")
                    ).read_text(encoding="utf-8")
                )["funding_verified_archives_loaded"]
            ),
            "missing_remote_404_archives_in_requested_range": int(
                json.loads(
                    Path(
                        args.economic_manifest
                        if args.economic_manifest is not None
                        else Path(args.economic_inputs).with_suffix(".manifest.json")
                    ).read_text(encoding="utf-8")
                )["funding_missing_archives"]
            ),
            "unknown_funding_cells_encountered_while_flat": funding_unknown_flat_cells,
            "unknown_funding_cells_encountered_while_carried": 0,
            "approximation_to_zero_for_missing_archive": False,
            "limitation": "Funding is known only for available_verified archive months; an unknown event is allowed only while the corresponding coin is flat.",
        },
        "raw_data_firewall": {
            "loaded_months": "bound_by_economic_input_manifest",
            "latest_loaded_month": "2025-06",
            "hard_cutoff_exclusive": DEVELOPMENT_END_EXCLUSIVE.isoformat(),
            "july_or_later_path_count": 0,
            "confirmation_rows_parsed": False,
        },
        "outputs": {name: sha256_file(output / name) for name in output_files},
        "primary_all_pass": bool(gates["primary_all_pass"]),
        "claim_boundary": "development evidence only; no Signature, HFT attainability, or confirmation claim",
    }
    _atomic_json(output / "completion_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-economic")
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--acquisition-manifest", type=Path, required=True)
    prepare.add_argument(
        "--pairs-research-root", type=Path, default=DEFAULT_AUDITED_ROOT
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(function=prepare_economic)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--components", type=Path, required=True)
    evaluation.add_argument("--economic-inputs", type=Path, required=True)
    evaluation.add_argument("--economic-manifest", type=Path)
    evaluation.add_argument("--forecast-output", type=Path, required=True)
    evaluation.add_argument(
        "--pairs-research-root", type=Path, default=DEFAULT_AUDITED_ROOT
    )
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.set_defaults(function=evaluate)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
