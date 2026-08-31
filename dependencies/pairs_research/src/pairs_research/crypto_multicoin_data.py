"""Fail-closed data core for the frozen multi-coin crypto experiment.

This module deliberately stops before constructing a signal or a strategy
return.  It provides the immutable 2021 formation cohort, checksum-verified
archive inventory semantics, causal availability/liquidity masks and weekly
pair hedges.  In particular, a symbol disappearing from Binance Vision never
causes another symbol (for example ``POLUSDT`` for ``MATICUSDT``) to be spliced
onto its history.

All intervals use half-open UTC ranges.  The last research timestamp that may
be read is therefore ``2025-07-01 00:00 UTC`` *exclusive*.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from .binance_um_data import normalize_funding_frame, normalize_kline_frame
from .binance_vision import (
    ArchiveDataset,
    VerifiedArchive,
    download_month_archive,
    month_labels,
    read_single_csv,
    sha256_file,
)


FROZEN_TOP12: Final[tuple[str, ...]] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "SOLUSDT",
    "EOSUSDT",
    "MATICUSDT",
)
FORMATION_RULE_VERSION: Final = "plain_usdt_full_15m_calendar_v1"
FIRST_ARCHIVE_MONTH: Final = "2021-01"
LAST_ARCHIVE_MONTH: Final = "2025-06"
HARD_CUTOFF_EXCLUSIVE: Final = pd.Timestamp("2025-07-01T00:00:00Z")
DISCOVERY_START: Final = pd.Timestamp("2022-07-01T00:00:00Z")
DISCOVERY_END_EXCLUSIVE: Final = HARD_CUTOFF_EXCLUSIVE
BAR_INTERVAL: Final = pd.Timedelta(minutes=15)
BARS_PER_DAY: Final = 96


def _as_utc(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    if pd.isna(timestamp):
        raise ValueError(f"{label} is not a valid timestamp")
    return timestamp


def _strict_bool(series: pd.Series, *, label: str) -> pd.Series:
    """Parse a CSV boolean column without accepting arbitrary truthy text."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false")).all():
        raise ValueError(f"formation audit {label} is not strictly boolean")
    return normalized.eq("true")


@dataclass(frozen=True)
class FrozenCryptoUniverse:
    """The exact ordered Top-12 cohort and its formation liquidity evidence."""

    symbols: tuple[str, ...]
    liquidity_ranks: tuple[int, ...]
    median_daily_quote_volume: tuple[float, ...]
    audit_path: Path

    def __post_init__(self) -> None:
        if self.symbols != FROZEN_TOP12:
            raise ValueError("universe symbols do not equal the frozen ordered Top 12")
        if self.liquidity_ranks != tuple(range(1, 13)):
            raise ValueError("universe liquidity ranks must be exactly 1..12")
        if len(self.median_daily_quote_volume) != 12 or not np.isfinite(
            self.median_daily_quote_volume
        ).all():
            raise ValueError("formation liquidity values must be twelve finite values")

    @property
    def rank_by_symbol(self) -> dict[str, int]:
        return dict(zip(self.symbols, self.liquidity_ranks, strict=True))


def load_frozen_top12(path: str | Path) -> FrozenCryptoUniverse:
    """Load and strictly verify the frozen formation-audit CSV.

    Row order is part of the contract.  Sorting a modified file back into the
    expected order is intentionally forbidden because that could hide a
    changed formation ranking.
    """

    audit_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(audit_path)
    required = {
        "symbol",
        "archive_rule_version",
        "available_2021_month_count",
        "available_2021_months",
        "observed_bars",
        "unique_bars",
        "duplicate_bars",
        "outside_year_bars",
        "missing_bar_count",
        "full_interval_year",
        "median_daily_quote_volume",
        "formation_eligible",
        "liquidity_rank",
        "selected_top_n",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"formation audit is missing columns: {missing}")
    if len(frame) != len(FROZEN_TOP12):
        raise ValueError("formation audit must contain exactly twelve rows")

    symbols = tuple(frame["symbol"].astype(str).str.strip().str.upper())
    if symbols != FROZEN_TOP12:
        raise ValueError("formation audit symbol order differs from frozen Top 12")
    if frame["symbol"].duplicated().any():
        raise ValueError("formation audit contains duplicate symbols")
    if not frame["archive_rule_version"].eq(FORMATION_RULE_VERSION).all():
        raise ValueError("formation audit rule version differs from the frozen rule")

    ranks = pd.to_numeric(frame["liquidity_rank"], errors="raise")
    if not np.array_equal(ranks.to_numpy(dtype=float), np.arange(1.0, 13.0)):
        raise ValueError("formation audit liquidity ranks must be exactly 1..12")
    expected_months = "|".join(f"2021-{month:02d}" for month in range(1, 13))
    structural_checks = (
        pd.to_numeric(frame["available_2021_month_count"], errors="raise").eq(12)
        & frame["available_2021_months"].astype(str).eq(expected_months)
        & pd.to_numeric(frame["observed_bars"], errors="raise").eq(35_040)
        & pd.to_numeric(frame["unique_bars"], errors="raise").eq(35_040)
        & pd.to_numeric(frame["duplicate_bars"], errors="raise").eq(0)
        & pd.to_numeric(frame["outside_year_bars"], errors="raise").eq(0)
        & pd.to_numeric(frame["missing_bar_count"], errors="raise").eq(0)
        & _strict_bool(frame["full_interval_year"], label="full_interval_year")
        & _strict_bool(frame["formation_eligible"], label="formation_eligible")
        & _strict_bool(frame["selected_top_n"], label="selected_top_n")
    )
    if not structural_checks.all():
        bad = frame.loc[~structural_checks, "symbol"].tolist()
        raise ValueError(f"formation audit eligibility proof failed for {bad}")
    liquidity = pd.to_numeric(
        frame["median_daily_quote_volume"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(liquidity).all() or (liquidity <= 0.0).any():
        raise ValueError("formation liquidity must be finite and positive")
    if (np.diff(liquidity) > 0.0).any():
        raise ValueError("formation liquidity does not follow the declared rank order")
    return FrozenCryptoUniverse(
        symbols=symbols,
        liquidity_ranks=tuple(int(value) for value in ranks),
        median_daily_quote_volume=tuple(float(value) for value in liquidity),
        audit_path=audit_path,
    )


def frozen_outer_fold_starts() -> pd.DatetimeIndex:
    """Return 36 folds after the causal 90-day eligibility warm-up."""

    result = pd.date_range(
        DISCOVERY_START,
        DISCOVERY_END_EXCLUSIVE,
        freq="MS",
        inclusive="left",
    )
    if len(result) != 36:
        raise AssertionError("frozen discovery calendar must contain 36 folds")
    return result


@dataclass(frozen=True, order=True)
class OrientedCryptoPair:
    """One unordered pair with a deterministic regression orientation.

    The larger *numeric* 2021 liquidity rank is the signal/dependent leg and
    the smaller numeric rank is its hedge/explanatory leg.  Consequently the
    first pair retains the earlier ETH-on-BTC convention.
    """

    pair_number: int
    signal_symbol: str
    hedge_symbol: str
    signal_rank: int
    hedge_rank: int

    @property
    def pair_id(self) -> str:
        return f"{self.signal_symbol}__{self.hedge_symbol}"


def oriented_pairs(
    universe: FrozenCryptoUniverse,
) -> tuple[OrientedCryptoPair, ...]:
    """Return all 66 pairs exactly once in deterministic formation order."""

    ranks = universe.rank_by_symbol
    pairs: list[OrientedCryptoPair] = []
    for pair_number, (hedge, signal) in enumerate(
        combinations(universe.symbols, 2), start=1
    ):
        pairs.append(
            OrientedCryptoPair(
                pair_number=pair_number,
                signal_symbol=signal,
                hedge_symbol=hedge,
                signal_rank=ranks[signal],
                hedge_rank=ranks[hedge],
            )
        )
    result = tuple(pairs)
    if len(result) != 66 or len({pair.pair_id for pair in result}) != 66:
        raise AssertionError("frozen Top-12 must create 66 unique pairs")
    return result


ArchiveDownloader = Callable[..., VerifiedArchive | None]


@dataclass(frozen=True)
class MultiCoinArchiveBundle:
    """Verified archives and explicit per-symbol/month availability."""

    universe: FrozenCryptoUniverse
    archives: tuple[VerifiedArchive, ...]
    availability: pd.DataFrame
    first_month: str
    last_month: str

    def __post_init__(self) -> None:
        months = _validate_closed_month_range(self.first_month, self.last_month)
        validate_archive_availability(
            self.availability,
            symbols=self.universe.symbols,
            months=months,
        )


def _validate_closed_month_range(first_month: str, last_month: str) -> tuple[str, ...]:
    months = month_labels(first_month, last_month)
    if pd.Period(months[0], freq="M") < pd.Period(FIRST_ARCHIVE_MONTH, freq="M"):
        raise ValueError(f"first month must not precede {FIRST_ARCHIVE_MONTH}")
    if pd.Period(months[-1], freq="M") > pd.Period(LAST_ARCHIVE_MONTH, freq="M"):
        raise ValueError(
            f"last month must not exceed hard research month {LAST_ARCHIVE_MONTH}"
        )
    return months


def validate_archive_availability(
    availability: pd.DataFrame,
    *,
    symbols: Sequence[str] = FROZEN_TOP12,
    months: Sequence[str] | None = None,
) -> None:
    """Validate a complete inventory and reject any post-gap symbol splice."""

    required = {
        "symbol",
        "month",
        "kline_available",
        "funding_available",
        "month_complete",
    }
    missing_columns = sorted(required.difference(availability.columns))
    if missing_columns:
        raise ValueError(f"archive availability is missing columns: {missing_columns}")
    expected_symbols = tuple(str(symbol).upper() for symbol in symbols)
    expected_months = (
        tuple(str(pd.Period(month, freq="M")) for month in months)
        if months is not None
        else tuple(dict.fromkeys(availability["month"].astype(str)))
    )
    expected_index = pd.MultiIndex.from_product(
        [expected_symbols, expected_months], names=["symbol", "month"]
    )
    frame = availability.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["month"] = frame["month"].map(lambda value: str(pd.Period(value, freq="M")))
    if frame.duplicated(["symbol", "month"]).any():
        raise ValueError("archive inventory contains duplicate symbol/month rows")
    actual_index = pd.MultiIndex.from_frame(frame[["symbol", "month"]])
    if set(actual_index) != set(expected_index) or len(actual_index) != len(expected_index):
        raise ValueError("archive inventory is not the exact symbol/month Cartesian grid")

    for column in ("kline_available", "funding_available", "month_complete"):
        frame[column] = _strict_bool(frame[column], label=column)
    expected_complete = frame["kline_available"] & frame["funding_available"]
    if not frame["month_complete"].equals(expected_complete):
        raise ValueError("month_complete must equal kline_available AND funding_available")

    ordered = frame.set_index(["symbol", "month"]).reindex(expected_index)
    for symbol in expected_symbols:
        complete = ordered.loc[symbol, "month_complete"].to_numpy(dtype=bool)
        missing_positions = np.flatnonzero(~complete)
        if missing_positions.size and complete[missing_positions[0] + 1 :].any():
            first_gap = expected_months[int(missing_positions[0])]
            raise ValueError(
                f"{symbol} has a complete archive after gap {first_gap}; "
                "symbol splicing/re-entry is forbidden"
            )


def download_multicoin_archives(
    cache_directory: str | Path,
    universe: FrozenCryptoUniverse,
    *,
    first_month: str = FIRST_ARCHIVE_MONTH,
    last_month: str = LAST_ARCHIVE_MONTH,
    downloader: ArchiveDownloader = download_month_archive,
) -> MultiCoinArchiveBundle:
    """Download the closed archive range without failing the whole universe.

    A remote 404 becomes an unavailable *symbol/month*.  Integrity and
    transport errors still raise.  All months are inventoried before the
    no-re-entry rule is checked, so a missing middle month cannot silently
    concatenate two disjoint histories.
    """

    months = _validate_closed_month_range(first_month, last_month)
    cache = Path(cache_directory).expanduser().resolve()
    archives: list[VerifiedArchive] = []
    rows: list[dict[str, object]] = []
    for symbol in universe.symbols:
        for month in months:
            found: dict[ArchiveDataset, VerifiedArchive | None] = {}
            for dataset in ("klines", "fundingRate"):
                archive = downloader(
                    cache,
                    symbol,
                    month,
                    dataset=dataset,
                    interval="15m",
                    allow_missing=True,
                )
                if archive is not None:
                    if (
                        archive.symbol != symbol
                        or archive.month != month
                        or archive.dataset != dataset
                    ):
                        raise ValueError("downloader returned archive metadata for another key")
                    archives.append(archive)
                found[dataset] = archive
            kline_available = found["klines"] is not None
            funding_available = found["fundingRate"] is not None
            rows.append(
                {
                    "symbol": symbol,
                    "month": month,
                    "kline_available": kline_available,
                    "funding_available": funding_available,
                    "month_complete": kline_available and funding_available,
                }
            )
    availability = pd.DataFrame(rows)
    validate_archive_availability(
        availability, symbols=universe.symbols, months=months
    )
    return MultiCoinArchiveBundle(
        universe=universe,
        archives=tuple(archives),
        availability=availability,
        first_month=months[0],
        last_month=months[-1],
    )


def load_frozen_multicoin_archive_bundle(
    manifest_path: str | Path,
    availability_path: str | Path,
    universe: FrozenCryptoUniverse,
    *,
    cache_directory: str | Path,
    first_month: str = FIRST_ARCHIVE_MONTH,
    last_month: str = LAST_ARCHIVE_MONTH,
    verify_sha256: bool = True,
) -> MultiCoinArchiveBundle:
    """Reconstruct a bundle from an immutable, previously frozen inventory.

    This is the offline counterpart to :func:`download_multicoin_archives`.
    It never contacts Binance or a checksum sidecar.  Every manifest path must
    remain below the declared cache directory and, by default, every archive
    byte is re-hashed against the digest frozen in the manifest.  The manifest
    must contain exactly one archive for every dataset marked available in the
    companion symbol/month inventory -- neither missing nor surplus entries
    are accepted.
    """

    months = _validate_closed_month_range(first_month, last_month)
    manifest_file = Path(manifest_path).expanduser().resolve()
    availability_file = Path(availability_path).expanduser().resolve()
    cache = Path(cache_directory).expanduser().resolve()
    manifest = pd.read_csv(manifest_file)
    availability = pd.read_csv(availability_file)
    required = {"dataset", "symbol", "month", "path", "sha256", "source_url"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"archive manifest is missing columns: {missing}")
    if manifest.empty:
        raise ValueError("archive manifest cannot be empty")

    manifest = manifest.loc[:, sorted(required)].copy()
    manifest["dataset"] = manifest["dataset"].astype(str)
    if not manifest["dataset"].isin(("klines", "fundingRate")).all():
        raise ValueError("archive manifest contains an unsupported dataset")
    manifest["symbol"] = manifest["symbol"].astype(str).str.upper()
    manifest["month"] = manifest["month"].map(
        lambda value: str(pd.Period(value, freq="M"))
    )
    if manifest.duplicated(["symbol", "month", "dataset"]).any():
        raise ValueError("archive manifest contains duplicate archive keys")
    if not manifest["symbol"].isin(universe.symbols).all():
        raise ValueError("archive manifest contains a symbol outside the frozen universe")
    if not manifest["month"].isin(months).all():
        raise ValueError("archive manifest contains a month outside the frozen range")

    validate_archive_availability(
        availability,
        symbols=universe.symbols,
        months=months,
    )
    normalized_availability = availability.copy()
    normalized_availability["symbol"] = (
        normalized_availability["symbol"].astype(str).str.upper()
    )
    normalized_availability["month"] = normalized_availability["month"].map(
        lambda value: str(pd.Period(value, freq="M"))
    )
    for column in ("kline_available", "funding_available", "month_complete"):
        normalized_availability[column] = _strict_bool(
            normalized_availability[column], label=column
        )
    expected_keys: set[tuple[str, str, str]] = set()
    for row in normalized_availability.itertuples(index=False):
        if row.kline_available:
            expected_keys.add((row.symbol, row.month, "klines"))
        if row.funding_available:
            expected_keys.add((row.symbol, row.month, "fundingRate"))
    actual_keys = set(
        manifest[["symbol", "month", "dataset"]].itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        missing_keys = sorted(expected_keys.difference(actual_keys))[:5]
        surplus_keys = sorted(actual_keys.difference(expected_keys))[:5]
        raise ValueError(
            "archive manifest disagrees with availability inventory; "
            f"missing={missing_keys}, surplus={surplus_keys}"
        )

    archives: list[VerifiedArchive] = []
    for row in manifest.sort_values(["symbol", "month", "dataset"]).itertuples(
        index=False
    ):
        path = Path(row.path).expanduser().resolve()
        if not path.is_relative_to(cache):
            raise ValueError(f"archive path escapes the frozen cache: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"frozen archive is missing: {path}")
        expected_digest = str(row.sha256).strip().lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise ValueError(f"invalid frozen archive SHA-256 for {path}")
        if verify_sha256:
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest:
                raise ValueError(f"frozen archive checksum mismatch: {path}")
        archives.append(
            VerifiedArchive(
                dataset=row.dataset,
                symbol=row.symbol,
                month=row.month,
                path=path,
                sha256=expected_digest,
                source_url=str(row.source_url),
            )
        )
    return MultiCoinArchiveBundle(
        universe=universe,
        archives=tuple(archives),
        availability=normalized_availability,
        first_month=months[0],
        last_month=months[-1],
    )


@dataclass(frozen=True)
class MultiCoinDataPanel:
    """Aligned values plus causal masks; no field contains strategy PnL."""

    universe: FrozenCryptoUniverse
    open_prices: pd.DataFrame
    high_prices: pd.DataFrame
    low_prices: pd.DataFrame
    close_prices: pd.DataFrame
    quote_volumes: pd.DataFrame
    trade_counts: pd.DataFrame
    funding_rates: pd.DataFrame
    archive_available: pd.DataFrame
    bar_available: pd.DataFrame
    return_available: pd.DataFrame
    rolling_30d_bar_coverage: pd.DataFrame
    coverage_mask: pd.DataFrame
    rolling_90d_median_daily_quote_volume: pd.DataFrame
    liquidity_mask: pd.DataFrame
    tradable_mask: pd.DataFrame
    symbol_ends: pd.DataFrame
    archive_availability: pd.DataFrame

    def __post_init__(self) -> None:
        expected = list(self.universe.symbols)
        for name in (
            "open_prices",
            "high_prices",
            "low_prices",
            "close_prices",
            "quote_volumes",
            "trade_counts",
            "archive_available",
            "bar_available",
            "return_available",
            "rolling_30d_bar_coverage",
            "coverage_mask",
            "rolling_90d_median_daily_quote_volume",
            "liquidity_mask",
            "tradable_mask",
        ):
            frame = getattr(self, name)
            if list(frame.columns) != expected or not frame.index.equals(
                self.open_prices.index
            ):
                raise ValueError(f"{name} does not share the frozen panel axes")
        finite_ohlc = (
            self.open_prices.notna()
            & self.high_prices.notna()
            & self.low_prices.notna()
            & self.close_prices.notna()
        )
        invalid_ohlc = finite_ohlc & (
            self.high_prices.lt(self.open_prices)
            | self.high_prices.lt(self.close_prices)
            | self.low_prices.gt(self.open_prices)
            | self.low_prices.gt(self.close_prices)
            | self.high_prices.lt(self.low_prices)
        )
        if invalid_ohlc.to_numpy(dtype=bool).any():
            raise ValueError("high/low prices do not contain open and close")
        if tuple(self.symbol_ends.index) != self.universe.symbols:
            raise ValueError("symbol_ends is not in frozen universe order")

    def pair_tradable_mask(self, pair: OrientedCryptoPair) -> pd.Series:
        """Return the causal intersection of both exact contract histories."""

        return (
            self.tradable_mask[pair.signal_symbol]
            & self.tradable_mask[pair.hedge_symbol]
        ).rename(pair.pair_id)


def _validate_archive_key_and_month(
    archive: VerifiedArchive,
    frame: pd.DataFrame | pd.Series,
) -> None:
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("archive reader did not produce a timezone-aware index")
    start = pd.Period(archive.month, freq="M").start_time.tz_localize("UTC")
    stop = (pd.Period(archive.month, freq="M") + 1).start_time.tz_localize("UTC")
    if len(index) and (index >= HARD_CUTOFF_EXCLUSIVE).any():
        raise ValueError("archive contains observations at or beyond the hard cutoff")
    if len(index) and ((index < start).any() or (index >= stop).any()):
        raise ValueError(f"{archive.symbol} {archive.month} archive contains another month")


def _causal_masks(
    quote_volumes: pd.DataFrame,
    bar_available: pd.DataFrame,
    *,
    coverage_days: int,
    minimum_coverage: float,
    liquidity_days: int,
    minimum_daily_quote_volume: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if coverage_days < 1 or liquidity_days < 1:
        raise ValueError("mask lookbacks must be positive")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum coverage must lie in (0, 1]")
    if minimum_daily_quote_volume < 0.0:
        raise ValueError("minimum liquidity must be non-negative")

    coverage_bars = coverage_days * BARS_PER_DAY
    # shift(1) proves that the bar whose position is being decided is not used.
    coverage = (
        bar_available.astype(float)
        .shift(1)
        .rolling(coverage_bars, min_periods=coverage_bars)
        .mean()
    )
    coverage_mask = coverage.ge(minimum_coverage).fillna(False)

    valid_quote = quote_volumes.where(bar_available)
    daily_volume = valid_quote.resample("1D").sum(min_count=BARS_PER_DAY)
    # A calendar day with anything other than all 96 bars is not a liquidity
    # observation; min_count above leaves it NaN.
    trailing_daily_median = (
        daily_volume.rolling(liquidity_days, min_periods=liquidity_days)
        .median()
        .shift(1)
    )
    normalized_days = quote_volumes.index.normalize()
    liquidity = trailing_daily_median.reindex(normalized_days)
    liquidity.index = quote_volumes.index
    liquidity_mask = liquidity.ge(minimum_daily_quote_volume).fillna(False)
    return coverage, coverage_mask, liquidity, liquidity_mask


def load_multicoin_panel(
    bundle: MultiCoinArchiveBundle,
    *,
    start: str | pd.Timestamp = f"{FIRST_ARCHIVE_MONTH}-01T00:00:00Z",
    end_exclusive: str | pd.Timestamp = HARD_CUTOFF_EXCLUSIVE,
    coverage_days: int = 30,
    minimum_coverage: float = 0.995,
    liquidity_days: int = 90,
    minimum_daily_quote_volume: float = 25_000_000.0,
) -> MultiCoinDataPanel:
    """Load exact 15-minute histories with per-symbol terminal missingness.

    Prices are never forward-filled.  A funding archive that is absent makes
    the entire symbol-month unavailable rather than being interpreted as a
    month of zero funding.
    """

    first = _as_utc(start, label="start")
    stop = _as_utc(end_exclusive, label="end_exclusive")
    if first.value % BAR_INTERVAL.value or stop.value % BAR_INTERVAL.value:
        raise ValueError("panel boundaries must lie on the 15-minute UTC grid")
    if stop <= first:
        raise ValueError("end_exclusive must follow start")
    if first < pd.Timestamp(f"{bundle.first_month}-01T00:00:00Z"):
        raise ValueError("start precedes the archive bundle")
    if stop > HARD_CUTOFF_EXCLUSIVE:
        raise ValueError("end_exclusive exceeds the hard research cutoff")
    bundle_stop = (pd.Period(bundle.last_month, freq="M") + 1).start_time.tz_localize(
        "UTC"
    )
    if stop > bundle_stop:
        raise ValueError("end_exclusive exceeds the archive bundle")

    inventory = bundle.availability.copy()
    for column in ("kline_available", "funding_available", "month_complete"):
        inventory[column] = _strict_bool(inventory[column], label=column)
    complete_keys = {
        (row.symbol, row.month)
        for row in inventory.itertuples(index=False)
        if row.month_complete
    }
    by_key: dict[tuple[str, str, str], VerifiedArchive] = {}
    availability_lookup = inventory.set_index(["symbol", "month"])
    for archive in bundle.archives:
        if archive.symbol not in bundle.universe.symbols:
            raise ValueError(
                f"archive symbol {archive.symbol} is not in the frozen universe; "
                "contract substitution/splicing is forbidden"
            )
        key = (archive.symbol, archive.month, archive.dataset)
        if key in by_key:
            raise ValueError(f"duplicate archive key: {key}")
        if (archive.symbol, archive.month) not in availability_lookup.index:
            raise ValueError(f"archive key lies outside the frozen inventory: {key}")
        availability_column = (
            "kline_available"
            if archive.dataset == "klines"
            else "funding_available"
        )
        if not bool(
            availability_lookup.at[(archive.symbol, archive.month), availability_column]
        ):
            raise ValueError(f"archive contradicts unavailable inventory key: {key}")
        by_key[key] = archive

    raw_klines: dict[str, list[pd.DataFrame]] = {
        symbol: [] for symbol in bundle.universe.symbols
    }
    raw_funding: dict[str, list[pd.Series]] = {
        symbol: [] for symbol in bundle.universe.symbols
    }
    for symbol, month in sorted(
        complete_keys,
        key=lambda item: (bundle.universe.symbols.index(item[0]), item[1]),
    ):
        kline_key = (symbol, month, "klines")
        funding_key = (symbol, month, "fundingRate")
        if kline_key not in by_key or funding_key not in by_key:
            raise ValueError(f"complete inventory row lacks verified archives: {symbol} {month}")
        kline_archive = by_key[kline_key]
        funding_archive = by_key[funding_key]
        kline_frame, _ = normalize_kline_frame(
            read_single_csv(kline_archive), symbol=symbol, interval="15m"
        )
        funding = normalize_funding_frame(
            read_single_csv(funding_archive), symbol=symbol
        )
        _validate_archive_key_and_month(kline_archive, kline_frame)
        _validate_archive_key_and_month(funding_archive, funding)
        raw_klines[symbol].append(kline_frame)
        raw_funding[symbol].append(funding)

    grid = pd.date_range(first, stop, freq=BAR_INTERVAL, inclusive="left")
    klines: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.Series] = {}
    for symbol in bundle.universe.symbols:
        if not raw_klines[symbol]:
            raise ValueError(f"no complete archive month is available for {symbol}")
        joined = pd.concat(raw_klines[symbol]).sort_index()
        if joined.index.duplicated().any():
            raise ValueError(f"{symbol} contains duplicate bars across archives")
        # Retain pre-start rows for an explicit delisting end and causal
        # lookback audits.  Reindexing below still exposes only the requested
        # half-open grid and therefore cannot leak these rows into a return.
        klines[symbol] = joined.loc[joined.index < stop]
        funding[symbol] = pd.concat(raw_funding[symbol]).sort_index().loc[
            lambda value: (value.index >= first) & (value.index < stop)
        ]

    def value_panel(column: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                symbol: klines[symbol][column].reindex(grid).astype(float)
                for symbol in bundle.universe.symbols
            },
            index=grid,
        ).rename_axis("timestamp")

    opens = value_panel("open")
    highs = value_panel("high")
    lows = value_panel("low")
    closes = value_panel("close")
    quote_volumes = value_panel("quote_volume")
    trade_counts = value_panel("count")

    month_complete = inventory.pivot(
        index="month", columns="symbol", values="month_complete"
    ).reindex(columns=bundle.universe.symbols)
    month_labels_for_grid = pd.Index(grid.strftime("%Y-%m"), name="month")
    archive_available = month_complete.reindex(month_labels_for_grid)
    archive_available.index = grid
    archive_available = archive_available.fillna(False).astype(bool)
    archive_available.index.name = "timestamp"

    bar_available = (
        opens.notna()
        & highs.notna()
        & lows.notna()
        & closes.notna()
        & quote_volumes.notna()
        & trade_counts.notna()
        & archive_available
    )
    return_available = bar_available & bar_available.shift(-1, fill_value=False)
    coverage, coverage_mask, liquidity, liquidity_mask = _causal_masks(
        quote_volumes,
        bar_available,
        coverage_days=coverage_days,
        minimum_coverage=minimum_coverage,
        liquidity_days=liquidity_days,
        minimum_daily_quote_volume=minimum_daily_quote_volume,
    )
    tradable = archive_available & bar_available & coverage_mask & liquidity_mask

    event_index = pd.DatetimeIndex(
        sorted(set().union(*(series.index for series in funding.values()))),
        tz="UTC",
    )
    funding_panel = pd.DataFrame(index=event_index, columns=bundle.universe.symbols)
    for symbol in bundle.universe.symbols:
        series = funding[symbol]
        for timestamp in event_index:
            month = timestamp.strftime("%Y-%m")
            available = bool(
                month in month_complete.index
                and month_complete.at[month, symbol]
            )
            if available:
                funding_panel.at[timestamp, symbol] = float(series.get(timestamp, 0.0))
    funding_panel = funding_panel.astype(float).rename_axis("timestamp")

    end_rows: list[dict[str, object]] = []
    final_expected_open = stop - BAR_INTERVAL
    for symbol in bundle.universe.symbols:
        observed = klines[symbol].index
        last_open = observed[-1] if len(observed) else pd.NaT
        ended_early = bool(pd.notna(last_open) and last_open < final_expected_open)
        symbol_inventory = inventory.loc[
            inventory["symbol"].eq(symbol)
        ].sort_values("month")
        complete_months = symbol_inventory.loc[
            symbol_inventory["month_complete"], "month"
        ].astype(str)
        incomplete_months = symbol_inventory.loc[
            ~symbol_inventory["month_complete"], "month"
        ].astype(str)
        end_rows.append(
            {
                "symbol": symbol,
                "last_complete_bar_open": last_open,
                "last_complete_bar_close": (
                    last_open + BAR_INTERVAL - pd.Timedelta(milliseconds=1)
                    if pd.notna(last_open)
                    else pd.NaT
                ),
                "first_unavailable_open": (
                    last_open + BAR_INTERVAL if ended_early else pd.NaT
                ),
                "ended_before_requested_cutoff": ended_early,
                "last_complete_archive_month": (
                    complete_months.iloc[-1] if len(complete_months) else ""
                ),
                "first_incomplete_archive_month": (
                    incomplete_months.iloc[0] if len(incomplete_months) else ""
                ),
                "terminal_reason": (
                    "source_coverage_end" if ended_early else "requested_cutoff"
                ),
                "replacement_symbol": "",
            }
        )
    symbol_ends = pd.DataFrame(end_rows).set_index("symbol").reindex(
        bundle.universe.symbols
    )
    symbol_ends.index.name = "symbol"

    return MultiCoinDataPanel(
        universe=bundle.universe,
        open_prices=opens,
        high_prices=highs,
        low_prices=lows,
        close_prices=closes,
        quote_volumes=quote_volumes,
        trade_counts=trade_counts,
        funding_rates=funding_panel,
        archive_available=archive_available,
        bar_available=bar_available,
        return_available=return_available,
        rolling_30d_bar_coverage=coverage,
        coverage_mask=coverage_mask,
        rolling_90d_median_daily_quote_volume=liquidity,
        liquidity_mask=liquidity_mask,
        tradable_mask=tradable,
        symbol_ends=symbol_ends,
        archive_availability=inventory,
    )


def synchronized_hourly_log_returns(open_prices: pd.DataFrame) -> pd.DataFrame:
    """Sample exact hourly boundary opens and compute log returns without fill."""

    if open_prices.empty or not isinstance(open_prices.index, pd.DatetimeIndex):
        raise ValueError("open_prices must have a non-empty DatetimeIndex")
    if open_prices.index.tz is None:
        raise ValueError("open_prices must be timezone-aware")
    if not open_prices.index.is_monotonic_increasing or not open_prices.index.is_unique:
        raise ValueError("open_prices index must be unique and increasing")
    if (open_prices.index.asi8 % BAR_INTERVAL.value != 0).any():
        raise ValueError("open_prices is not on the 15-minute UTC grid")
    values = open_prices.astype(float)
    values = values.where(values > 0.0)
    hourly = values.loc[
        (values.index.minute == 0)
        & (values.index.second == 0)
        & (values.index.microsecond == 0)
    ]
    returns = np.log(hourly).diff()
    returns.index.name = "timestamp"
    return returns


@dataclass(frozen=True)
class MultiCoinPairHedge:
    effective_at: pd.Timestamp
    pair: OrientedCryptoPair
    sample_start_exclusive: pd.Timestamp
    sample_end_exclusive: pd.Timestamp
    formation_observations: int
    raw_beta: float
    beta: float
    signal_weight: float
    hedge_weight: float
    estimable: bool
    pair_available: bool
    pair_liquid: bool
    tradable: bool

    @property
    def weights(self) -> dict[str, float]:
        return {
            self.pair.signal_symbol: self.signal_weight,
            self.pair.hedge_symbol: self.hedge_weight,
        }


def _mask_at(
    mask: pd.DataFrame | None,
    timestamp: pd.Timestamp,
    symbols: tuple[str, str],
    *,
    default: bool,
) -> bool:
    if mask is None:
        return default
    missing = set(symbols).difference(mask.columns)
    if missing:
        raise ValueError(f"mask is missing symbols: {sorted(missing)}")
    if timestamp not in mask.index:
        return False
    values = mask.loc[timestamp, list(symbols)]
    return bool(values.notna().all() and values.astype(bool).all())


def estimate_weekly_multicoin_hedges(
    open_prices: pd.DataFrame,
    pairs: Sequence[OrientedCryptoPair],
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    formation_days: int = 30,
    refit_days: int = 7,
    minimum_hourly_observations: int = 500,
    beta_floor: float = 0.25,
    beta_cap: float = 2.5,
    availability_mask: pd.DataFrame | None = None,
    liquidity_mask: pd.DataFrame | None = None,
) -> tuple[MultiCoinPairHedge, ...]:
    """Estimate all pair betas from synchronized prior hourly log returns.

    One row is returned for every pair/effective-time combination.  Failed
    estimates carry NaN weights and ``tradable=False``; they are never silently
    dropped from the family.
    """

    if formation_days < 1 or refit_days < 1:
        raise ValueError("formation_days and refit_days must be positive")
    if minimum_hourly_observations < 2:
        raise ValueError("minimum_hourly_observations must be at least two")
    if not 0.0 < beta_floor <= beta_cap:
        raise ValueError("beta bounds are invalid")
    if len(pairs) != 66 or len({pair.pair_id for pair in pairs}) != 66:
        raise ValueError("the frozen hedge family must contain exactly 66 pairs")
    expected_symbols = set(FROZEN_TOP12)
    if len(open_prices.columns) != 12 or set(open_prices.columns) != expected_symbols:
        raise ValueError("open price columns must equal the frozen Top 12")
    expected_pair_keys = tuple(
        (signal, hedge)
        for hedge, signal in combinations(FROZEN_TOP12, 2)
    )
    actual_pair_keys = tuple(
        (pair.signal_symbol, pair.hedge_symbol) for pair in pairs
    )
    if actual_pair_keys != expected_pair_keys:
        raise ValueError("pair order/orientation differs from the frozen 66-pair family")

    first = _as_utc(start, label="start")
    stop = _as_utc(end_exclusive, label="end_exclusive")
    if stop <= first:
        raise ValueError("end_exclusive must follow start")
    if stop > HARD_CUTOFF_EXCLUSIVE:
        raise ValueError("hedge end exceeds the hard research cutoff")
    effective_times = pd.date_range(first, stop, freq=f"{refit_days}D", inclusive="left")
    hourly_returns = synchronized_hourly_log_returns(
        open_prices.loc[:, list(FROZEN_TOP12)]
    )

    estimates: list[MultiCoinPairHedge] = []
    formation_delta = pd.Timedelta(days=formation_days)
    for effective in effective_times:
        sample_start = effective - formation_delta
        # A log return stamped h uses the interval (h-1h, h].  Requiring h to
        # be strictly after the left boundary keeps that interval inside the
        # 30-day formation window.  h < effective excludes the live boundary.
        window = hourly_returns.loc[
            (hourly_returns.index > sample_start)
            & (hourly_returns.index < effective)
        ]
        for pair in pairs:
            symbols = (pair.signal_symbol, pair.hedge_symbol)
            sample = window.loc[:, list(symbols)].dropna()
            observations = int(len(sample))
            raw_beta = np.nan
            beta = np.nan
            signal_weight = np.nan
            hedge_weight = np.nan
            estimable = False
            if observations >= minimum_hourly_observations:
                signal_values = sample[pair.signal_symbol].to_numpy(dtype=float)
                hedge_values = sample[pair.hedge_symbol].to_numpy(dtype=float)
                variance = float(np.var(hedge_values, ddof=1))
                if np.isfinite(variance) and variance > 0.0:
                    covariance = float(
                        np.cov(signal_values, hedge_values, ddof=1)[0, 1]
                    )
                    raw_beta = covariance / variance
                    if np.isfinite(raw_beta):
                        beta = float(np.clip(raw_beta, beta_floor, beta_cap))
                        denominator = 1.0 + abs(beta)
                        signal_weight = 1.0 / denominator
                        hedge_weight = -beta / denominator
                        estimable = True
            finite_at_effective = bool(
                effective in open_prices.index
                and open_prices.loc[effective, list(symbols)].notna().all()
            )
            pair_available = finite_at_effective and _mask_at(
                availability_mask,
                effective,
                symbols,
                default=True,
            )
            pair_liquid = _mask_at(
                liquidity_mask,
                effective,
                symbols,
                default=True,
            )
            estimates.append(
                MultiCoinPairHedge(
                    effective_at=effective,
                    pair=pair,
                    sample_start_exclusive=sample_start,
                    sample_end_exclusive=effective,
                    formation_observations=observations,
                    raw_beta=float(raw_beta),
                    beta=float(beta),
                    signal_weight=float(signal_weight),
                    hedge_weight=float(hedge_weight),
                    estimable=estimable,
                    pair_available=pair_available,
                    pair_liquid=pair_liquid,
                    tradable=bool(estimable and pair_available and pair_liquid),
                )
            )
    return tuple(estimates)


def hedge_estimates_frame(
    estimates: Iterable[MultiCoinPairHedge],
) -> pd.DataFrame:
    """Flatten hedge records into a deterministic, audit-friendly table."""

    rows = [
        {
            "effective_at": item.effective_at,
            "pair_number": item.pair.pair_number,
            "pair_id": item.pair.pair_id,
            "signal_symbol": item.pair.signal_symbol,
            "hedge_symbol": item.pair.hedge_symbol,
            "signal_rank": item.pair.signal_rank,
            "hedge_rank": item.pair.hedge_rank,
            "sample_start_exclusive": item.sample_start_exclusive,
            "sample_end_exclusive": item.sample_end_exclusive,
            "formation_observations": item.formation_observations,
            "raw_beta": item.raw_beta,
            "beta": item.beta,
            "signal_weight": item.signal_weight,
            "hedge_weight": item.hedge_weight,
            "estimable": item.estimable,
            "pair_available": item.pair_available,
            "pair_liquid": item.pair_liquid,
            "tradable": item.tradable,
        }
        for item in estimates
    ]
    if not rows:
        raise ValueError("hedge estimate collection is empty")
    return pd.DataFrame(rows).sort_values(
        ["effective_at", "pair_number"], ignore_index=True
    )
