"""Normalization and audits for Binance USD-M intraday research data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .binance_vision import (
    SUPPORTED_KLINE_INTERVALS,
    VerifiedArchive,
    read_single_csv,
    validate_kline_interval,
)


KLINE_COLUMNS = {
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
}
FUNDING_COLUMNS = {
    "calc_time",
    "funding_interval_hours",
    "last_funding_rate",
}
_INTERVAL_DELTAS = {label: pd.Timedelta(label) for label in SUPPORTED_KLINE_INTERVALS}
_FUNDING_MINUTE_SNAP_TOLERANCE = pd.Timedelta(seconds=1)


def _coerce_kline_interval(
    interval: str | pd.Timedelta,
) -> tuple[str, pd.Timedelta]:
    """Normalize the public interval forms while rejecting arbitrary grids."""

    if isinstance(interval, str):
        label = validate_kline_interval(interval)
        return label, _INTERVAL_DELTAS[label]
    if not isinstance(interval, pd.Timedelta):
        raise TypeError("interval must be one of '1m', '5m', '15m' or a Timedelta")
    for label, delta in _INTERVAL_DELTAS.items():
        if interval == delta:
            return label, delta
    allowed = ", ".join(SUPPORTED_KLINE_INTERVALS)
    raise ValueError(f"unsupported kline interval {interval}; expected one of {allowed}")


def _require_grid_aligned(
    timestamp: pd.Timestamp,
    interval: pd.Timedelta,
    label: str,
) -> None:
    if timestamp.value % interval.value != 0:
        raise ValueError(f"{label} must lie on the {interval} UTC bar-open grid")


@dataclass(frozen=True)
class SymbolDataAudit:
    symbol: str
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    observations: int
    expected_grid_rows: int
    missing_grid_rows: int
    duplicate_rows: int
    nonpositive_prices: int


@dataclass(frozen=True)
class FundingDataAudit:
    """Timestamp-canonicalization audit for one symbol's funding events.

    ``snapped_events`` counts events whose raw official timestamp was shifted
    to the nearest full minute.  ``max_abs_offset_ms`` is measured before that
    shift and is zero when every event was already minute-aligned.
    """

    symbol: str
    raw_events: int
    snapped_events: int
    max_abs_offset_ms: int


@dataclass(frozen=True)
class BinanceUMPanel:
    """Aligned USD-M prices, volumes and sparse funding rates in UTC."""

    open_prices: pd.DataFrame
    close_prices: pd.DataFrame
    quote_volumes: pd.DataFrame
    trade_counts: pd.DataFrame
    funding_rates: pd.DataFrame
    audits: tuple[SymbolDataAudit, ...]
    funding_audits: tuple[FundingDataAudit, ...] = ()


def _utc_from_milliseconds(values: pd.Series, label: str) -> pd.DatetimeIndex:
    numeric = pd.to_numeric(values, errors="raise")
    if (numeric % 1 != 0).any():
        raise ValueError(f"{label} must contain integer milliseconds")
    result = pd.to_datetime(numeric.astype("int64"), unit="ms", utc=True)
    if result.isna().any():
        raise ValueError(f"{label} contains invalid timestamps")
    return pd.DatetimeIndex(result)


def normalize_kline_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    interval: str | pd.Timedelta = "15m",
) -> tuple[pd.DataFrame, SymbolDataAudit]:
    """Validate one symbol's concatenated kline frame without filling gaps."""

    interval_label, interval_delta = _coerce_kline_interval(interval)
    missing = sorted(KLINE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"kline frame is missing columns: {missing}")
    value = frame.copy()
    value.index = _utc_from_milliseconds(value["open_time"], "open_time")
    value.index.name = "timestamp"
    duplicates = int(value.index.duplicated(keep=False).sum())
    if duplicates:
        raise ValueError(f"{symbol} contains duplicate kline timestamps")
    value = value.sort_index()
    if value.empty:
        raise ValueError(f"{symbol} kline frame is empty")
    interval_ns = int(interval_delta.value)
    if (value.index.asi8 % interval_ns != 0).any():
        raise ValueError(
            f"{symbol} open timestamps are off the {interval_label} grid"
        )
    close_time = _utc_from_milliseconds(value["close_time"], "close_time")
    expected_close = value.index + interval_delta - pd.Timedelta(milliseconds=1)
    if not np.array_equal(close_time.asi8, expected_close.asi8):
        raise ValueError(f"{symbol} close timestamps disagree with the bar interval")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "count",
    ]
    for column in numeric_columns:
        value[column] = pd.to_numeric(value[column], errors="raise")
    prices = value[["open", "high", "low", "close"]].to_numpy(dtype=float)
    nonpositive = int((~np.isfinite(prices) | (prices <= 0.0)).sum())
    if nonpositive:
        raise ValueError(f"{symbol} contains nonpositive or non-finite prices")
    if (value[["volume", "quote_volume", "count"]].to_numpy(dtype=float) < 0).any():
        raise ValueError(f"{symbol} contains negative activity fields")

    expected = pd.date_range(
        value.index[0], value.index[-1], freq=interval_delta, tz="UTC"
    )
    missing_rows = int(len(expected.difference(value.index)))
    audit = SymbolDataAudit(
        symbol=str(symbol).upper(),
        first_timestamp=value.index[0],
        last_timestamp=value.index[-1],
        observations=int(len(value)),
        expected_grid_rows=int(len(expected)),
        missing_grid_rows=missing_rows,
        duplicate_rows=duplicates,
        nonpositive_prices=nonpositive,
    )
    return value, audit


def _normalize_funding_frame_with_audit(
    frame: pd.DataFrame, *, symbol: str
) -> tuple[pd.Series, FundingDataAudit]:
    missing = sorted(FUNDING_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"funding frame is missing columns: {missing}")
    raw_timestamps = _utc_from_milliseconds(frame["calc_time"], "calc_time")
    timestamps = raw_timestamps.round("min")
    offsets_ns = np.abs(raw_timestamps.asi8 - timestamps.asi8)
    tolerance_ns = int(_FUNDING_MINUTE_SNAP_TOLERANCE.value)
    if (offsets_ns > tolerance_ns).any():
        maximum_ms = float(offsets_ns.max() / 1_000_000.0)
        raise ValueError(
            f"{symbol} funding timestamp is more than "
            f"{_FUNDING_MINUTE_SNAP_TOLERANCE.total_seconds():g}s from a full "
            f"minute (maximum offset {maximum_ms:g}ms)"
        )
    if timestamps.duplicated().any():
        raise ValueError(
            f"{symbol} contains duplicate funding timestamps after minute "
            "canonicalization"
        )
    rates = pd.to_numeric(frame["last_funding_rate"], errors="raise").astype(float)
    intervals = pd.to_numeric(frame["funding_interval_hours"], errors="raise")
    if not np.isfinite(rates).all() or not np.isfinite(intervals).all():
        raise ValueError(f"{symbol} funding fields must be finite")
    if (intervals <= 0).any():
        raise ValueError(f"{symbol} funding intervals must be positive")
    result = pd.Series(rates.to_numpy(), index=timestamps, name=str(symbol).upper())
    audit = FundingDataAudit(
        symbol=str(symbol).upper(),
        raw_events=int(len(raw_timestamps)),
        snapped_events=int(np.count_nonzero(offsets_ns)),
        max_abs_offset_ms=(
            int(offsets_ns.max() // 1_000_000) if len(offsets_ns) else 0
        ),
    )
    return result.sort_index(), audit


def normalize_funding_frame(frame: pd.DataFrame, *, symbol: str) -> pd.Series:
    """Return sparse funding rates on their canonical UTC settlement minutes.

    Binance's official funding archives occasionally encode a settlement a
    handful of milliseconds away from its minute boundary.  Such offsets are
    snapped only when they are at most one second from the nearest full minute;
    larger offsets are rejected instead of silently moving an economic event.
    """

    result, _ = _normalize_funding_frame_with_audit(frame, symbol=symbol)
    return result


def load_um_panel(
    archives: Iterable[VerifiedArchive],
    *,
    symbols: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str | pd.Timedelta = "15m",
) -> BinanceUMPanel:
    """Load verified archives into aligned panels on the requested full grid.

    The requested end is inclusive for bar *open* timestamps.  No price or
    return is forward-filled.  Funding remains sparse and is filled with zero
    only after aligning to an explicit execution event clock.
    """

    _, interval_delta = _coerce_kline_interval(interval)
    requested_symbols = tuple(str(symbol).upper() for symbol in symbols)
    if len(requested_symbols) == 0 or len(set(requested_symbols)) != len(
        requested_symbols
    ):
        raise ValueError("symbols must be a non-empty unique sequence")
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    start_time = (
        start_time.tz_localize("UTC")
        if start_time.tzinfo is None
        else start_time.tz_convert("UTC")
    )
    end_time = (
        end_time.tz_localize("UTC")
        if end_time.tzinfo is None
        else end_time.tz_convert("UTC")
    )
    if end_time < start_time:
        raise ValueError("end must not precede start")
    _require_grid_aligned(start_time, interval_delta, "start")
    _require_grid_aligned(end_time, interval_delta, "end")

    by_key: dict[tuple[str, str], list[VerifiedArchive]] = {}
    for archive in archives:
        by_key.setdefault((archive.symbol, archive.dataset), []).append(archive)

    klines: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.Series] = {}
    audits: list[SymbolDataAudit] = []
    funding_audits: list[FundingDataAudit] = []
    for symbol in requested_symbols:
        kline_archives = sorted(
            by_key.get((symbol, "klines"), []), key=lambda item: item.month
        )
        funding_archives = sorted(
            by_key.get((symbol, "fundingRate"), []), key=lambda item: item.month
        )
        if not kline_archives or not funding_archives:
            raise ValueError(f"missing kline or funding archives for {symbol}")
        raw_klines = pd.concat(
            [read_single_csv(archive) for archive in kline_archives],
            ignore_index=True,
        )
        normalized, audit = normalize_kline_frame(
            raw_klines,
            symbol=symbol,
            interval=interval_delta,
        )
        normalized = normalized.loc[
            (normalized.index >= start_time) & (normalized.index <= end_time)
        ]
        if normalized.empty:
            raise ValueError(f"no requested klines for {symbol}")
        # Recompute the range-specific gap audit used by the experiment.
        requested_grid = pd.date_range(
            start_time, end_time, freq=interval_delta, tz="UTC"
        )
        range_audit = SymbolDataAudit(
            symbol=symbol,
            first_timestamp=normalized.index[0],
            last_timestamp=normalized.index[-1],
            observations=int(len(normalized)),
            expected_grid_rows=int(len(requested_grid)),
            missing_grid_rows=int(len(requested_grid.difference(normalized.index))),
            duplicate_rows=audit.duplicate_rows,
            nonpositive_prices=audit.nonpositive_prices,
        )
        klines[symbol] = normalized
        audits.append(range_audit)

        raw_funding = pd.concat(
            [read_single_csv(archive) for archive in funding_archives],
            ignore_index=True,
        )
        series, funding_audit = _normalize_funding_frame_with_audit(
            raw_funding, symbol=symbol
        )
        funding[symbol] = series.loc[
            (series.index >= start_time)
            & (series.index <= end_time + interval_delta)
        ]
        funding_audits.append(funding_audit)

    grid = pd.date_range(start_time, end_time, freq=interval_delta, tz="UTC")

    def panel(column: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                symbol: klines[symbol][column].reindex(grid).astype(float)
                for symbol in requested_symbols
            },
            index=grid,
        ).rename_axis("timestamp")

    sparse_funding_index = pd.DatetimeIndex(
        sorted(set().union(*(series.index for series in funding.values())))
    )
    funding_panel = pd.DataFrame(
        {
            symbol: funding[symbol].reindex(sparse_funding_index).fillna(0.0)
            for symbol in requested_symbols
        },
        index=sparse_funding_index,
    ).rename_axis("timestamp")
    return BinanceUMPanel(
        open_prices=panel("open"),
        close_prices=panel("close"),
        quote_volumes=panel("quote_volume"),
        trade_counts=panel("count"),
        funding_rates=funding_panel,
        audits=tuple(audits),
        funding_audits=tuple(funding_audits),
    )


def next_open_returns(open_prices: pd.DataFrame) -> pd.DataFrame:
    """Return open-to-next-open simple returns indexed by the entry timestamp."""

    if open_prices.empty or not isinstance(open_prices.index, pd.DatetimeIndex):
        raise ValueError("open_prices needs a non-empty DatetimeIndex")
    if not open_prices.index.is_monotonic_increasing or not open_prices.index.is_unique:
        raise ValueError("open_prices index must be unique and increasing")
    values = open_prices.astype(float)
    returns = values.shift(-1).div(values).sub(1.0)
    # A return across a missing bar must remain missing; never infer a zero move.
    returns = returns.where(values.notna() & values.shift(-1).notna())
    returns.index.name = "timestamp"
    return returns
