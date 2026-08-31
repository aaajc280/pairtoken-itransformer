"""Causal data and feature construction for the paired one-minute experiment.

The raw-data reader is intentionally fail closed.  It constructs only explicit
development-month paths and never walks, globs, or recursively inventories the
archive root.  A timestamp request is validated before any filesystem method is
called.  The panel itself is factorized by coin so the full Top-24 cache does
not materialize a ``time x 276 pairs x 232 features`` tensor on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd


DEVELOPMENT_START = pd.Timestamp("2021-01-01T00:00:00Z")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2025-07-01T00:00:00Z")
FIRST_MONTH = "2021-01"
LAST_MONTH = "2025-06"
MINUTES_PER_HOUR = 60
CONTROL_BLOCK_MINUTES = 15
ENDPOINT_HISTORY_HOURS = 168
RAW_DIMENSION = 232
STATE_DIMENSION = 4
PAIR_PATH_DIMENSION = 60
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_CHECKSUM = re.compile(r"^([0-9a-fA-F]{64})\s+([^/\\]+)\s*$")


class ConfirmationFirewallError(PermissionError):
    """Raised before raw-path access when a request crosses June 2025."""


def utc(value: object, *, field: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError(f"{field} must be timezone aware")
    return result.tz_convert("UTC")


def development_months(start: object, end_exclusive: object) -> tuple[str, ...]:
    """Return months intersecting ``[start,end)`` after firewall validation."""

    first = utc(start, field="start")
    stop = utc(end_exclusive, field="end_exclusive")
    if stop <= first:
        raise ValueError("end_exclusive must be later than start")
    if first < DEVELOPMENT_START:
        raise ConfirmationFirewallError("start predates the development boundary")
    if stop > DEVELOPMENT_END_EXCLUSIVE:
        raise ConfirmationFirewallError("request crosses the confirmation firewall")
    last_included = stop - pd.Timedelta(1, unit="ns")
    periods = pd.period_range(
        pd.Period(first.strftime("%Y-%m"), freq="M"),
        pd.Period(last_included.strftime("%Y-%m"), freq="M"),
        freq="M",
    )
    labels = tuple(str(period) for period in periods)
    if not labels or min(labels) < FIRST_MONTH or max(labels) > LAST_MONTH:
        raise ConfirmationFirewallError("resolved archive month is outside development")
    return labels


def require_development_month(month: str) -> str:
    if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(month)) is None:
        raise ValueError(f"invalid month label: {month!r}")
    if month < FIRST_MONTH or month > LAST_MONTH:
        raise ConfirmationFirewallError(
            f"month {month} is outside development; final allowed month is {LAST_MONTH}"
        )
    return month


def kline_archive_path(root: Path, *, symbol: str, month: str) -> Path:
    """Construct one exact path only after validating its month."""

    safe_month = require_development_month(month)
    if re.fullmatch(r"[A-Z0-9]+USDT", str(symbol)) is None:
        raise ValueError(f"invalid plain-USDT symbol: {symbol!r}")
    filename = f"{symbol}-1m-{safe_month}.zip"
    return (
        Path(root)
        / "futures"
        / "um"
        / "monthly"
        / "klines"
        / str(symbol)
        / "1m"
        / filename
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _official_checksum(archive: Path) -> str:
    sidecar = archive.with_name(f"{archive.name}.CHECKSUM")
    if not sidecar.is_file():
        raise FileNotFoundError(f"official checksum sidecar missing: {sidecar}")
    match = _CHECKSUM.fullmatch(sidecar.read_text(encoding="ascii").strip())
    if match is None or match.group(2) != archive.name:
        raise ValueError(f"invalid official checksum sidecar: {sidecar}")
    digest = match.group(1).lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError("invalid official SHA-256")
    return digest


@dataclass(frozen=True)
class ManifestArchiveRecord:
    symbol: str
    month: str
    availability: str
    archive_relative_path: str
    archive_sha256: str
    official_checksum_sha256: str
    checksum_relative_path: str
    checksum_sidecar_sha256: str


@dataclass(frozen=True)
class VerifiedAcquisitionManifest:
    directory: Path
    first_month: str
    last_month: str
    universe: tuple[str, ...]
    kline_records: dict[tuple[str, str], ManifestArchiveRecord]
    artifact_hashes: dict[str, str]


def load_acquisition_manifest(
    directory: Path,
    *,
    require_full_contract: bool = True,
) -> VerifiedAcquisitionManifest:
    """Verify the acquisition receipt and return its complete kline allowlist.

    Only the six named manifest files are addressed.  The manifest directory
    and the raw cache are never recursively enumerated.
    """

    root = Path(directory).expanduser().resolve()
    hash_receipt = root / "artifact_hashes.sha256"
    hashes_path = root / "artifact_hashes.json"
    manifest_path = root / "acquisition_manifest.json"
    availability_path = root / "archive_availability.csv"
    requested_path = root / "requested_inventory.csv"
    archive_manifest_path = root / "archive_manifest.csv"
    for path in (
        hash_receipt,
        hashes_path,
        manifest_path,
        availability_path,
        requested_path,
        archive_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    receipt = hash_receipt.read_text(encoding="ascii").strip().split()
    if len(receipt) != 2 or receipt[1] != "artifact_hashes.json" or not _SHA256.fullmatch(receipt[0]):
        raise ValueError("invalid acquisition artifact-hash receipt")
    if sha256_file(hashes_path) != receipt[0].lower():
        raise ValueError("acquisition artifact-hash receipt differs")
    hash_payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    artifact_hashes = hash_payload.get("files")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("acquisition artifact hash map is missing")
    named = {
        "acquisition_manifest.json": manifest_path,
        "archive_availability.csv": availability_path,
        "archive_manifest.csv": archive_manifest_path,
        "requested_inventory.csv": requested_path,
    }
    if set(artifact_hashes) != set(named):
        raise ValueError("acquisition artifact hash family differs")
    for name, path in named.items():
        if sha256_file(path) != str(artifact_hashes[name]):
            raise ValueError(f"acquisition manifest artifact differs: {name}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "development_archive_acquisition_complete":
        raise ValueError("acquisition is not complete")
    if payload.get("archive_csv_bodies_parsed") is not False:
        raise ValueError("acquisition receipt already parsed raw bodies")
    firewall = payload.get("confirmation_firewall", {})
    if firewall.get("hard_cutoff_exclusive_utc") != "2025-07-01T00:00:00Z":
        raise ConfirmationFirewallError("acquisition firewall differs")
    if any(
        firewall.get(name) is not False
        for name in (
            "months_after_2025_06_may_be_downloaded",
            "months_after_2025_06_may_be_enumerated",
            "months_after_2025_06_may_be_read",
        )
    ):
        raise ConfirmationFirewallError("acquisition permits reserved-month access")
    grid = payload.get("request_grid", {})
    first_month = str(grid.get("first_month"))
    last_month = str(grid.get("final_allowed_month"))
    require_development_month(first_month)
    require_development_month(last_month)
    if last_month != LAST_MONTH:
        raise ValueError("acquisition does not end at June 2025")
    if require_full_contract and first_month != FIRST_MONTH:
        raise ValueError("production run requires the full January-2021 manifest")
    universe = tuple(map(str, payload.get("universe", ())))
    if len(universe) != 24 or len(set(universe)) != 24:
        raise ValueError("acquisition universe is not the fixed Top-24")
    availability = pd.read_csv(availability_path, dtype=str, keep_default_na=False)
    requested = pd.read_csv(requested_path, dtype=str, keep_default_na=False)
    available_manifest = pd.read_csv(
        archive_manifest_path, dtype=str, keep_default_na=False
    )
    required_columns = {
        "symbol",
        "month",
        "dataset",
        "interval",
        "availability",
        "archive_relative_path",
        "archive_sha256",
        "official_checksum_sha256",
        "checksum_relative_path",
        "checksum_sidecar_sha256",
    }
    if not required_columns.issubset(availability.columns):
        raise ValueError("archive availability schema differs")
    identity_columns = [
        "ordinal",
        "liquidity_rank",
        "symbol",
        "month",
        "dataset",
        "interval",
    ]
    if not set(identity_columns).issubset(requested.columns):
        raise ValueError("requested inventory schema differs")
    expected_count = int(grid.get("archive_request_count", -1))
    if len(requested) != expected_count or len(availability) != expected_count:
        raise ValueError("acquisition request count differs")
    if not requested[identity_columns].equals(availability[identity_columns]):
        raise ValueError("requested and availability identities differ")
    available_rows = availability.loc[availability["availability"].eq("available_verified")]
    if not available_manifest.equals(available_rows.reset_index(drop=True)):
        raise ValueError("available-only archive manifest differs")
    if len(available_rows) != int(payload.get("available_verified_count", -1)):
        raise ValueError("available archive count differs")
    missing_rows = availability.loc[availability["availability"].eq("missing_remote_404")]
    if len(missing_rows) != int(payload.get("missing_remote_404_count", -1)):
        raise ValueError("404 archive count differs")
    if (availability["month"] > LAST_MONTH).any():
        raise ConfirmationFirewallError("manifest contains a reserved month")
    kline = availability.loc[availability["dataset"].eq("klines")].copy()
    if not kline["interval"].eq("1m").all():
        raise ValueError("manifest contains a non-1m kline request")
    if kline.duplicated(["symbol", "month"]).any():
        raise ValueError("manifest kline identities are duplicated")
    records: dict[tuple[str, str], ManifestArchiveRecord] = {}
    for row in kline.itertuples(index=False):
        require_development_month(str(row.month))
        if str(row.symbol) not in universe:
            raise ValueError("manifest kline symbol lies outside the fixed universe")
        status = str(row.availability)
        if status not in {"available_verified", "missing_remote_404"}:
            raise ValueError(f"unsupported archive availability: {status}")
        record = ManifestArchiveRecord(
            symbol=str(row.symbol),
            month=str(row.month),
            availability=status,
            archive_relative_path=str(row.archive_relative_path),
            archive_sha256=str(row.archive_sha256),
            official_checksum_sha256=str(row.official_checksum_sha256),
            checksum_relative_path=str(row.checksum_relative_path),
            checksum_sidecar_sha256=str(row.checksum_sidecar_sha256),
        )
        if status == "missing_remote_404" and any(
            (
                record.archive_relative_path,
                record.archive_sha256,
                record.official_checksum_sha256,
                record.checksum_relative_path,
                record.checksum_sidecar_sha256,
            )
        ):
            raise ValueError("404 manifest row unexpectedly binds an archive")
        records[(record.symbol, record.month)] = record
    return VerifiedAcquisitionManifest(
        root,
        first_month,
        last_month,
        universe,
        records,
        {str(key): str(value) for key, value in artifact_hashes.items()},
    )


def _missing_month_series(*, symbol: str, month: str) -> pd.Series:
    start = pd.Timestamp(f"{require_development_month(month)}-01T00:00:00Z")
    stop = start + pd.offsets.MonthBegin(1)
    index = pd.date_range(start, stop - pd.Timedelta(1, unit="min"), freq="1min", tz="UTC")
    return pd.Series(np.nan, index=index, name=symbol, dtype=np.float64)


def read_kline_month(
    root: Path,
    *,
    symbol: str,
    month: str,
    record: ManifestArchiveRecord | None = None,
) -> pd.Series:
    """Read one explicit, checksum-verified development archive.

    The expected CSV member is addressed directly; archive members are not
    enumerated.  No forward/backward filling is performed.
    """

    safe_month = require_development_month(month)
    archive_path = kline_archive_path(root, symbol=symbol, month=safe_month)
    if record is not None:
        if (record.symbol, record.month) != (symbol, safe_month):
            raise ValueError("manifest record identity differs from archive request")
        if record.availability != "available_verified":
            raise ValueError("only available_verified records may be opened")
        expected_relative = archive_path.relative_to(Path(root)).as_posix()
        if record.archive_relative_path != expected_relative:
            raise ValueError("manifest archive path differs from the exact expected path")
        expected_sidecar = archive_path.with_name(f"{archive_path.name}.CHECKSUM")
        if record.checksum_relative_path != expected_sidecar.relative_to(Path(root)).as_posix():
            raise ValueError("manifest checksum path differs from the exact expected path")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    expected = _official_checksum(archive_path)
    observed = sha256_file(archive_path)
    if observed != expected:
        raise ValueError(f"archive checksum differs from official sidecar: {archive_path}")
    if record is not None:
        sidecar = archive_path.with_name(f"{archive_path.name}.CHECKSUM")
        expected_bindings = {
            "archive_sha256": observed,
            "official_checksum_sha256": expected,
            "checksum_sidecar_sha256": sha256_file(sidecar),
        }
        for field, actual in expected_bindings.items():
            if getattr(record, field) != actual:
                raise ValueError(f"manifest {field} differs from cache")
    member = archive_path.name.removesuffix(".zip") + ".csv"
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        try:
            handle = archive.open(member, mode="r")
        except KeyError as error:
            raise ValueError(f"expected archive member is missing: {member}") from error
        with handle:
            frame = pd.read_csv(
                handle,
                header=None,
                usecols=[0, 4],
                names=["timestamp", "close"],
                dtype=str,
            )
    numeric_time = pd.to_numeric(frame["timestamp"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    valid = numeric_time.notna() & close.notna() & (close > 0.0)
    numeric_time = numeric_time.loc[valid].astype(np.int64)
    close = close.loc[valid].astype(np.float64)
    if numeric_time.empty:
        raise ValueError(f"archive contains no numeric kline rows: {archive_path}")
    unit = "us" if int(numeric_time.abs().max()) >= 100_000_000_000_000 else "ms"
    index = pd.to_datetime(numeric_time.to_numpy(), unit=unit, utc=True)
    result = pd.Series(close.to_numpy(), index=index, name=symbol, dtype=np.float64)
    if not result.index.is_monotonic_increasing or not result.index.is_unique:
        raise ValueError(f"kline clock is not strictly increasing: {archive_path}")
    period_start = pd.Timestamp(f"{safe_month}-01T00:00:00Z")
    period_stop = period_start + pd.offsets.MonthBegin(1)
    if result.index.min() < period_start or result.index.max() >= period_stop:
        raise ValueError(f"archive contains a timestamp outside its declared month: {archive_path}")
    if result.index.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise ConfirmationFirewallError("parsed kline timestamp crosses the firewall")
    return result


def load_development_closes(
    root: Path,
    *,
    manifest_directory: Path,
    symbols: Sequence[str],
    start: object,
    end_exclusive: object,
    require_full_contract_manifest: bool = True,
) -> pd.DataFrame:
    """Load manifest-authorized development archives without filling 404s."""

    first = utc(start, field="start")
    stop = utc(end_exclusive, field="end_exclusive")
    months = development_months(first, stop)  # validation precedes filesystem I/O
    manifest = load_acquisition_manifest(
        manifest_directory, require_full_contract=require_full_contract_manifest
    )
    names = tuple(map(str, symbols))
    if not names or len(names) != len(set(names)):
        raise ValueError("symbols must be non-empty and unique")
    if any(symbol not in manifest.universe for symbol in names):
        raise ValueError("requested symbol lies outside the manifest universe")
    columns: list[pd.Series] = []
    for symbol in names:
        blocks: list[pd.Series] = []
        for month in months:
            record = manifest.kline_records.get((symbol, month))
            if record is None:
                raise ValueError(f"manifest lacks requested kline row: {symbol}/{month}")
            if record.availability == "missing_remote_404":
                # This is the only permitted source of an all-missing month.
                blocks.append(_missing_month_series(symbol=symbol, month=month))
            elif record.availability == "available_verified":
                blocks.append(
                    read_kline_month(
                        root,
                        symbol=symbol,
                        month=month,
                        record=record,
                    )
                )
            else:  # load_acquisition_manifest already rejects this; defensive boundary.
                raise ValueError(f"unsupported manifest status: {record.availability}")
        series = pd.concat(blocks).sort_index()
        if not series.index.is_unique:
            raise ValueError(f"duplicate minute timestamp for {symbol}")
        columns.append(series.loc[(series.index >= first) & (series.index < stop)].rename(symbol))
    frame = pd.concat(columns, axis=1).sort_index()
    frame.index = pd.DatetimeIndex(frame.index, name="minute_open_at")
    if frame.index.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise ConfirmationFirewallError("loaded close panel crosses the firewall")
    return frame.reindex(columns=list(names))


def causal_15m_control(paths: np.ndarray) -> np.ndarray:
    """Remove within-block path variation while retaining all 15m endpoints."""

    values = np.asarray(paths, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != MINUTES_PER_HOUR:
        raise ValueError("path axis must contain exactly 60 one-minute increments")
    reshaped = values.reshape(*values.shape[:-1], 4, CONTROL_BLOCK_MINUTES)
    block_means = reshaped.mean(axis=-1, keepdims=True, dtype=np.float64)
    result = np.broadcast_to(block_means, reshaped.shape).reshape(values.shape).astype(np.float32)
    native_blocks = reshaped.sum(axis=-1, dtype=np.float64)
    control_blocks = result.reshape(*values.shape[:-1], 4, CONTROL_BLOCK_MINUTES).sum(
        axis=-1, dtype=np.float64
    )
    if not np.allclose(native_blocks, control_blocks, atol=2e-9, rtol=2e-6):
        raise AssertionError("causal 15m control changed a block endpoint")
    return result


def axis_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        (json.dumps(list(map(str, values)), separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _validate_transient_open_gap_report(
    report: object,
    *,
    clock: pd.DatetimeIndex,
    symbols: tuple[str, ...],
    legs: np.ndarray,
    opens: np.ndarray,
    returns: np.ndarray,
    coin_decision: np.ndarray,
    coin_target: np.ndarray,
    terminal: np.ndarray,
    pair_decision: np.ndarray,
    pair_target: np.ndarray,
) -> None:
    expected_keys = {
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
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("economic transient-gap report schema differs")
    if report["policy"] != "no_fill_no_shortening_mask_outcome_only" or any(
        report[field] is not True
        for field in (
            "no_fill",
            "no_shortening",
            "feature_and_decision_availability_unchanged",
        )
    ):
        raise ValueError("economic transient-gap policy differs")
    transient = np.zeros_like(coin_decision, dtype=bool)
    transient[:-1] = (
        coin_decision[:-1]
        & ~np.isfinite(opens[1:])
        & ~terminal[:-1]
    )
    if (
        np.isfinite(returns[transient]).any()
        or coin_target[transient].any()
        or terminal[transient].any()
        or not coin_decision[transient].all()
    ):
        raise ValueError("economic transient gap was filled, shortened, or decision-masked")
    pair_transient = np.zeros_like(pair_decision, dtype=bool)
    for pair, (signal, hedge) in enumerate(legs):
        pair_transient[:, pair] = pair_decision[:, pair] & (
            transient[:, signal] | transient[:, hedge]
        )
    if np.any(pair_transient & pair_target):
        raise ValueError("economic transient-gap pair label remained available")
    oos = np.zeros(len(clock), dtype=bool)
    for start, stop in (
        ("2023-04-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        ("2024-04-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    ):
        oos |= (clock >= pd.Timestamp(start)) & (clock < pd.Timestamp(stop))
    events: list[dict[str, Any]] = []
    for row in np.flatnonzero(transient.any(axis=1)):
        affected = [
            symbol for column, symbol in enumerate(symbols) if transient[row, column]
        ]
        events.append(
            {
                "decision_timestamp": clock[row].isoformat(),
                "missing_next_open_timestamp": (
                    clock[row] + pd.Timedelta(hours=1)
                ).isoformat(),
                "symbols": affected,
                "coin_gap_cells": len(affected),
                "pair_label_cells_masked": int(pair_transient[row].sum()),
            }
        )
    rebuilt = {
        "policy": "no_fill_no_shortening_mask_outcome_only",
        "no_fill": True,
        "no_shortening": True,
        "feature_and_decision_availability_unchanged": True,
        "coin_gap_cells": int(transient.sum()),
        "pair_label_cells_masked": int(pair_transient.sum()),
        "oos_coin_gap_cells": int(transient[oos].sum()),
        "oos_pair_label_cells": int(pair_transient[oos].sum()),
        "events": events,
    }
    if report != rebuilt:
        raise ValueError("economic transient-gap report differs from stored primitives")


@dataclass(frozen=True)
class HourlyEconomicInputs:
    """Exact one-hour target primitives shared with the audited ledger."""

    clock: pd.DatetimeIndex
    symbols: tuple[str, ...]
    pair_ids: tuple[str, ...]
    pair_leg_indices: np.ndarray
    pair_weights: np.ndarray
    coin_open_prices: np.ndarray
    coin_open_to_open_returns: np.ndarray
    funding_at_boundary: np.ndarray
    next_boundary_growth: np.ndarray
    funding_at_next_boundary: np.ndarray
    coin_decision_available: np.ndarray
    coin_target_available: np.ndarray
    pair_decision_available: np.ndarray
    pair_target_cashflows: np.ndarray
    pair_target_available: np.ndarray
    terminal_exit_mask: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        clock = pd.DatetimeIndex(self.clock)
        if clock.tz is None:
            raise ValueError("economic clock must be timezone aware")
        clock = clock.tz_convert("UTC").rename("decision_at")
        if not clock.is_monotonic_increasing or not clock.is_unique:
            raise ValueError("economic clock must be unique and increasing")
        if len(clock) and clock[-1] >= DEVELOPMENT_END_EXCLUSIVE:
            raise ConfirmationFirewallError("economic clock crosses the firewall")
        symbols = tuple(map(str, self.symbols))
        pairs = tuple(map(str, self.pair_ids))
        rows, coins, pair_count = len(clock), len(symbols), len(pairs)
        legs = np.asarray(self.pair_leg_indices, dtype=np.int64)
        weights = np.asarray(self.pair_weights, dtype=np.float64)
        opens = np.asarray(self.coin_open_prices, dtype=np.float64)
        returns = np.asarray(self.coin_open_to_open_returns, dtype=np.float64)
        funding_boundary = np.asarray(self.funding_at_boundary, dtype=np.float64)
        growth = np.asarray(self.next_boundary_growth, dtype=np.float64)
        funding = np.asarray(self.funding_at_next_boundary, dtype=np.float64)
        coin_decision = np.asarray(self.coin_decision_available, dtype=bool)
        coin_target = np.asarray(self.coin_target_available, dtype=bool)
        pair_decision = np.asarray(self.pair_decision_available, dtype=bool)
        targets = np.asarray(self.pair_target_cashflows, dtype=np.float64)
        pair_target = np.asarray(self.pair_target_available, dtype=bool)
        terminal = np.asarray(self.terminal_exit_mask, dtype=bool)
        if legs.shape != (pair_count, 2) or (legs < 0).any() or (legs >= coins).any():
            raise ValueError("economic pair-leg axis differs")
        if weights.shape != (rows, pair_count, 2):
            raise ValueError("economic pair weights have wrong axes")
        for name, values in (
            ("coin_open_prices", opens),
            ("coin_open_to_open_returns", returns),
            ("funding_at_boundary", funding_boundary),
            ("next_boundary_growth", growth),
            ("funding_at_next_boundary", funding),
        ):
            if values.shape != (rows, coins):
                raise ValueError(f"{name} has wrong axes")
        if any(value.shape != (rows, coins) for value in (coin_decision, coin_target, terminal)):
            raise ValueError("economic coin mask has wrong axes")
        if any(value.shape != (rows, pair_count) for value in (pair_decision, targets, pair_target)):
            raise ValueError("economic pair target/mask has wrong axes")
        if np.any(pair_target & ~pair_decision):
            raise ValueError("economic target availability widens decision availability")
        pair_coin_decision = coin_decision[:, legs].all(axis=2)
        pair_coin_target = coin_target[:, legs].all(axis=2)
        if np.any(pair_decision & ~pair_coin_decision):
            raise ValueError("pair decision availability exceeds coin availability")
        if np.any(pair_target & ~pair_coin_target):
            raise ValueError("pair target availability exceeds outcome availability")
        if not np.isfinite(weights[pair_decision]).all():
            raise ValueError("available economic pair weights are nonfinite")
        gross = np.abs(weights).sum(axis=2)
        if not np.allclose(gross[pair_decision], 1.0, atol=1e-7, rtol=1e-7):
            raise ValueError("available economic pair weights are not gross normalized")
        if not np.isfinite(opens[coin_decision]).all():
            raise ValueError("available economic decision opens are nonfinite")
        if not np.isfinite(returns[coin_target]).all() or not np.isfinite(growth[coin_target]).all():
            raise ValueError("available economic price outcomes are nonfinite")
        if not np.allclose(
            growth[coin_target], 1.0 + returns[coin_target], atol=1e-12, rtol=1e-12
        ):
            raise ValueError("next-boundary growth differs from arithmetic return")
        if not np.isfinite(funding[coin_target]).all():
            raise ValueError("available next-boundary funding is nonfinite")
        # funding_at_boundary belongs to the position carried into t, not to
        # the new action label.  The audited ledger may retain NaN here for a
        # manifest-missing archive and later proves that the coin was flat.
        if np.any(terminal & (np.nan_to_num(funding, nan=0.0) != 0.0)):
            raise ValueError("terminal-close exit cannot carry next-boundary funding")
        leg_growth = growth[:, legs]
        leg_funding = funding[:, legs]
        rebuilt = np.sum(
            weights * (leg_growth - 1.0) - weights * leg_growth * leg_funding,
            axis=2,
        )
        if not np.isfinite(targets[pair_target]).all():
            raise ValueError("available economic pair target is nonfinite")
        if not np.allclose(
            targets[pair_target], rebuilt[pair_target], atol=2e-12, rtol=2e-10
        ):
            raise ValueError("economic pair target differs from executable rebuild")
        if not np.all(targets[~pair_target] == 0.0):
            raise ValueError("unavailable economic pair targets must be stored as zero")
        metadata = dict(self.metadata)
        if metadata.get("schema") != "paired_hourly_economic_inputs_v2":
            raise ValueError("economic input metadata schema differs")
        if metadata.get("hard_cutoff_exclusive") != DEVELOPMENT_END_EXCLUSIVE.isoformat():
            raise ConfirmationFirewallError("economic metadata cutoff differs")
        if metadata.get("symbols") != list(symbols) or metadata.get("pair_ids") != list(pairs):
            raise ValueError("economic metadata axes differ")
        transient_report = metadata.get("transient_open_gap_report")
        if transient_report is not None:
            _validate_transient_open_gap_report(
                transient_report,
                clock=clock,
                symbols=symbols,
                legs=legs,
                opens=opens,
                returns=returns,
                coin_decision=coin_decision,
                coin_target=coin_target,
                terminal=terminal,
                pair_decision=pair_decision,
                pair_target=pair_target,
            )
        object.__setattr__(self, "clock", clock)


def load_hourly_economic_inputs(
    path: Path, manifest_path: Path
) -> tuple[HourlyEconomicInputs, dict[str, object]]:
    """Load the evaluator's sole v2 economic artifact, failing closed.

    This adapter deliberately has no v1 or best-effort compatibility path.  The
    target used by the neural fit and the target replayed by the audited ledger
    must be the same content-addressed object.
    """

    source = Path(path).expanduser().resolve()
    manifest_source = Path(manifest_path).expanduser().resolve()
    if not source.is_file() or not manifest_source.is_file():
        raise FileNotFoundError(source if not source.is_file() else manifest_source)
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    required_manifest = {
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
    if set(manifest) != required_manifest:
        raise ValueError("economic input manifest schema differs")
    if manifest["schema"] != "paired_hourly_economic_inputs_manifest_v2":
        raise ValueError("economic input schema differs")
    if manifest["status"] != "complete_immutable_shared_training_and_ledger_input":
        raise ValueError("economic inputs are not complete")
    if manifest["artifact_sha256"] != sha256_file(source):
        raise ValueError("economic input hash differs from manifest")
    if manifest["confirmation_rows_parsed"] is not False or manifest[
        "july_or_later_path_count"
    ] != 0:
        raise ConfirmationFirewallError("economic input manifest parsed confirmation")
    if manifest["latest_raw_month"] != LAST_MONTH:
        raise ConfirmationFirewallError("economic input latest raw month differs")
    if manifest["funding_included_in_target"] is not True:
        raise ValueError("economic targets omit funding")
    if manifest["pair_target_formula"] != (
        "sum_l w[t,l]*r[t,l] - sum_l w[t,l]*(1+r[t,l])*funding[t+1,l]"
    ):
        raise ValueError("economic target formula differs")
    identity = manifest["one_period_ledger_identity"]
    if set(identity) != {
        "ordinary_with_next_boundary_funding",
        "ordinary_without_funding_event",
        "terminal_delisting_mark",
    } or any(
        not isinstance(report, dict) or report.get("passed") is not True
        for report in identity.values()
    ):
        raise ValueError("economic artifact lacks all one-period ledger identity proofs")
    multi_identity = manifest["multi_period_ledger_identity"]
    if (
        not isinstance(multi_identity, dict)
        or multi_identity.get("passed") is not True
        or multi_identity.get("terminal_zero_target_included") is not True
    ):
        raise ValueError("economic artifact lacks the multi-period ledger identity proof")
    with np.load(source, allow_pickle=False) as stored:
        expected_members = {
            "metadata_json",
            "clock_ns",
            "symbols",
            "pair_ids",
            "pair_leg_indices",
            "pair_weights",
            "coin_open_prices",
            "coin_open_to_open_returns",
            "funding_at_boundary",
            "next_boundary_growth",
            "funding_at_next_boundary",
            "coin_decision_available",
            "coin_target_available",
            "pair_decision_available",
            "pair_target_cashflows",
            "pair_target_available",
            "terminal_exit_mask",
        }
        if set(stored.files) != expected_members:
            raise ValueError("economic input NPZ members differ")
        metadata = json.loads(str(stored["metadata_json"].item()))
        result = HourlyEconomicInputs(
            clock=pd.to_datetime(stored["clock_ns"], utc=True),
            symbols=tuple(stored["symbols"].astype(str)),
            pair_ids=tuple(stored["pair_ids"].astype(str)),
            pair_leg_indices=stored["pair_leg_indices"],
            pair_weights=stored["pair_weights"],
            coin_open_prices=stored["coin_open_prices"],
            coin_open_to_open_returns=stored["coin_open_to_open_returns"],
            funding_at_boundary=stored["funding_at_boundary"],
            next_boundary_growth=stored["next_boundary_growth"],
            funding_at_next_boundary=stored["funding_at_next_boundary"],
            coin_decision_available=stored["coin_decision_available"],
            coin_target_available=stored["coin_target_available"],
            pair_decision_available=stored["pair_decision_available"],
            pair_target_cashflows=stored["pair_target_cashflows"],
            pair_target_available=stored["pair_target_available"],
            terminal_exit_mask=stored["terminal_exit_mask"],
            metadata=metadata,
        )
    if manifest["clock_start"] != result.clock[0].isoformat() or manifest[
        "clock_end"
    ] != result.clock[-1].isoformat():
        raise ValueError("economic input clock differs from manifest")
    if manifest["hourly_observations"] != len(result.clock):
        raise ValueError("economic observation count differs from manifest")
    if manifest["symbols"] != list(result.symbols):
        raise ValueError("economic symbol axis differs from manifest")
    if manifest["pairs"] != len(result.pair_ids):
        raise ValueError("economic pair count differs from manifest")
    newline_pair_hash = hashlib.sha256("\n".join(result.pair_ids).encode()).hexdigest()
    if manifest["pair_ids_sha256"] != newline_pair_hash:
        raise ValueError("economic pair axis differs from manifest")
    if manifest["pair_decision_available_cells"] != int(result.pair_decision_available.sum()):
        raise ValueError("economic decision-availability count differs")
    if manifest["pair_target_available_cells"] != int(result.pair_target_available.sum()):
        raise ValueError("economic target-availability count differs")
    if manifest["terminal_exit_cells"] != int(result.terminal_exit_mask.sum()):
        raise ValueError("economic terminal-exit count differs")
    sources = manifest["audited_source_hashes"]
    if not isinstance(sources, dict) or not sources or any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in sources.values()
    ):
        raise ValueError("economic source-hash family is invalid")
    provenance = result.metadata.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("audited_source_hashes") != sources:
        raise ValueError("economic artifact and manifest audited sources differ")
    if manifest["transient_open_gap_report"] != result.metadata.get(
        "transient_open_gap_report"
    ):
        raise ValueError("economic transient-gap report differs between artifact and manifest")
    return result, manifest


@dataclass(frozen=True)
class PairedComponents:
    """Factorized hourly panel shared by both path representations."""

    clock: pd.DatetimeIndex
    symbols: tuple[str, ...]
    pair_ids: tuple[str, ...]
    pair_leg_indices: np.ndarray
    coin_minute_returns: np.ndarray
    coin_hourly_returns: np.ndarray
    pair_weights: np.ndarray
    pair_path_endpoints: np.ndarray
    target_cashflows: np.ndarray
    feature_available: np.ndarray
    target_available: np.ndarray

    def __post_init__(self) -> None:
        clock = pd.DatetimeIndex(self.clock)
        if clock.tz is None:
            raise ValueError("component clock must be UTC aware")
        clock = clock.tz_convert("UTC").rename("decision_at")
        if not clock.is_monotonic_increasing or not clock.is_unique:
            raise ValueError("component clock must be unique and increasing")
        if len(clock) and (clock.asi8 % 3_600_000_000_000 != 0).any():
            raise ValueError("component clock must lie on the hourly grid")
        if len(clock) and clock[-1] >= DEVELOPMENT_END_EXCLUSIVE:
            raise ConfirmationFirewallError("component clock crosses the firewall")
        rows, coins, pairs = len(clock), len(self.symbols), len(self.pair_ids)
        legs = np.asarray(self.pair_leg_indices, dtype=np.int64)
        minute = np.asarray(self.coin_minute_returns, dtype=np.float32)
        hourly = np.asarray(self.coin_hourly_returns, dtype=np.float32)
        weights = np.asarray(self.pair_weights, dtype=np.float32)
        endpoints = np.asarray(self.pair_path_endpoints, dtype=np.float32)
        cashflows = np.asarray(self.target_cashflows, dtype=np.float32)
        features = np.asarray(self.feature_available, dtype=bool)
        targets = np.asarray(self.target_available, dtype=bool)
        if legs.shape != (pairs, 2) or (legs < 0).any() or (legs >= coins).any():
            raise ValueError("pair leg indices are invalid")
        if minute.shape != (rows, coins, MINUTES_PER_HOUR):
            raise ValueError("coin minute-return cube has wrong axes")
        if hourly.shape != (rows, coins):
            raise ValueError("coin hourly returns have wrong axes")
        if weights.shape != (rows, pairs, 2):
            raise ValueError("pair weights have wrong axes")
        if endpoints.shape != (rows, pairs):
            raise ValueError("pair endpoint matrix has wrong axes")
        if cashflows.shape != (rows, pairs):
            raise ValueError("target cashflow matrix has wrong axes")
        if features.shape != endpoints.shape or targets.shape != endpoints.shape:
            raise ValueError("availability masks have wrong axes")
        if not np.isfinite(cashflows[targets]).all():
            raise ValueError("available targets must be finite")
        object.__setattr__(self, "clock", clock)

    @property
    def label_mask(self) -> np.ndarray:
        return self.feature_available & self.target_available

    def _pair_paths(self, episode_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(episode_rows, dtype=np.int64)
        signal = self.pair_leg_indices[:, 0]
        hedge = self.pair_leg_indices[:, 1]
        signal_paths = self.coin_minute_returns[rows[:, None], signal[None, :], :]
        hedge_paths = self.coin_minute_returns[rows[:, None], hedge[None, :], :]
        weights = self.pair_weights[rows]
        result = (
            weights[:, :, 0, None] * signal_paths
            + weights[:, :, 1, None] * hedge_paths
        )
        return np.asarray(result, dtype=np.float32)

    def features(self, rows: Sequence[int] | np.ndarray, *, representation: str) -> np.ndarray:
        selected = np.asarray(rows, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("feature rows must be a non-empty vector")
        if selected.min() < ENDPOINT_HISTORY_HOURS or selected.max() >= len(self.clock):
            raise IndexError("feature row lacks history or lies outside the panel")
        paths = self._pair_paths(selected - 1)
        # Missing/delisted legs can leave NaN weights in an unavailable pair.
        # The pair is masked from both models, so canonicalize its path before
        # applying either representation.  In particular, the control's
        # endpoint invariant must be checked only on executable feature cells,
        # not fail merely because an already-masked path contains NaNs.
        mask = self.feature_available[selected]
        paths = np.where(mask[:, :, None], paths, 0.0).astype(np.float32)
        if representation == "causal_15m_control":
            paths = causal_15m_control(paths)
        elif representation != "native_1m":
            raise KeyError(f"unknown representation: {representation}")
        history = np.stack(
            [self.pair_path_endpoints[row - ENDPOINT_HISTORY_HOURS : row].T for row in selected],
            axis=0,
        ).astype(np.float32)
        weights = self.pair_weights[selected]
        hours = self.clock.take(selected).hour.to_numpy(dtype=np.float64)
        phase = 2.0 * np.pi * hours / 24.0
        time = np.column_stack((np.sin(phase), np.cos(phase))).astype(np.float32)
        time = np.broadcast_to(time[:, None, :], (len(selected), len(self.pair_ids), 2))
        raw = np.concatenate((paths, history, weights, time), axis=2).astype(np.float32)
        if raw.shape != (len(selected), len(self.pair_ids), RAW_DIMENSION):
            raise AssertionError("materialized raw feature dimension differs")
        return np.where(mask[:, :, None], raw, 0.0).astype(np.float32)


def _rolling_pair_weights(
    hourly: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]],
    lookback_hours: int,
    minimum_hours: int,
    beta_clip: tuple[float, float],
) -> np.ndarray:
    rows = len(hourly)
    weights = np.full((rows, len(pairs), 2), np.nan, dtype=np.float32)
    for pair_number, (signal, hedge) in enumerate(pairs):
        y = hourly[signal]
        x = hourly[hedge]
        x_lag = x.shift(1)
        y_lag = y.shift(1)
        rolling = pd.concat({"x": x_lag, "y": y_lag}, axis=1).rolling(
            int(lookback_hours), min_periods=int(minimum_hours)
        )
        covariance = rolling.cov().unstack().loc[:, ("x", "y")]
        variance = x_lag.rolling(int(lookback_hours), min_periods=int(minimum_hours)).var(ddof=1)
        beta = covariance.div(variance.where(variance > 1e-12)).clip(*beta_clip)
        scale = 1.0 + beta.abs()
        weights[:, pair_number, 0] = (1.0 / scale).to_numpy(dtype=np.float32)
        weights[:, pair_number, 1] = (-beta / scale).to_numpy(dtype=np.float32)
    return weights


def build_components_from_closes(
    closes: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    economic_inputs: HourlyEconomicInputs | None = None,
    allow_synthetic_target: bool = False,
    hedge_lookback_hours: int = 720,
    minimum_hedge_hours: int = 168,
    beta_clip: tuple[float, float] = (-3.0, 3.0),
) -> PairedComponents:
    """Construct the factorized path panel and attach exact economic labels.

    Production callers must supply the evaluator's content-verified
    ``economic_inputs``.  At decision row ``t`` the most recent path is episode
    ``t-1`` and is formed with the weights that were in force at the start of
    that episode, ``w[t-1]``.  The common endpoint history is likewise strictly
    prior to ``t``.  Feature availability depends only on those past paths and
    current decision information; it never depends on the future-outcome mask.

    ``allow_synthetic_target`` exists only for unit and mechanical smoke tests.
    It must be opted into explicitly and is never used by production prepare.
    """

    frame = pd.DataFrame(closes, dtype=np.float64).copy()
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("at least two coin close series are required")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("minute closes must be timezone aware")
    index = index.tz_convert("UTC")
    if index.max() >= DEVELOPMENT_END_EXCLUSIVE:
        raise ConfirmationFirewallError("input closes cross the firewall")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("minute close clock must be unique and increasing")
    frame.index = index
    symbols = tuple(map(str, frame.columns))
    if economic_inputs is None and not allow_synthetic_target:
        raise ValueError(
            "production component construction requires audited hourly economic inputs"
        )
    if economic_inputs is not None:
        if economic_inputs.symbols != symbols:
            raise ValueError("close and economic symbol axes differ")
        economic_legs = np.asarray(economic_inputs.pair_leg_indices, dtype=np.int64)
        oriented = tuple((symbols[signal], symbols[hedge]) for signal, hedge in economic_legs)
        if pairs is not None and tuple(map(tuple, pairs)) != oriented:
            raise ValueError("requested and economic pair axes differ")
        pair_ids = tuple(economic_inputs.pair_ids)
        expected_pair_ids = tuple(f"{signal}__{hedge}" for signal, hedge in oriented)
        if pair_ids != expected_pair_ids:
            raise ValueError("economic pair identifiers differ from their leg axis")
        clock = economic_inputs.clock
    else:
        if pairs is None:
            oriented = tuple(
                (symbols[second], symbols[first])
                for first in range(len(symbols))
                for second in range(first + 1, len(symbols))
            )
        else:
            oriented = tuple((str(signal), str(hedge)) for signal, hedge in pairs)
        pair_ids = tuple(f"{signal}__{hedge}" for signal, hedge in oriented)
        first_hour = index.min().floor("h")
        last_hour = index.max().floor("h")
        clock = pd.date_range(first_hour, last_hour, freq="1h", tz="UTC", name="decision_at")
    if not oriented or len(oriented) != len(set(oriented)):
        raise ValueError("pairs must be non-empty and unique")
    if any(signal not in symbols or hedge not in symbols or signal == hedge for signal, hedge in oriented):
        raise ValueError("pair leg is missing or duplicated")
    if len(clock) == 0:
        raise ValueError("hourly component clock is empty")
    first_hour = clock[0]
    last_hour = clock[-1]
    minute_grid = pd.date_range(
        first_hour, last_hour + pd.Timedelta(59, unit="min"), freq="1min", tz="UTC"
    )
    # Include the preceding close only to compute the first requested minute
    # return.  No filling is performed when it is unavailable.
    differencing_grid = minute_grid.insert(0, first_hour - pd.Timedelta(1, unit="min"))
    dense = frame.reindex(differencing_grid)
    log_close = np.log(dense.where(dense > 0.0))
    minute_returns = log_close.diff().reindex(minute_grid)
    cube = minute_returns.to_numpy(dtype=np.float64).reshape(-1, MINUTES_PER_HOUR, len(symbols))
    complete = np.isfinite(cube).all(axis=1)
    hourly_values = np.where(
        complete,
        np.nansum(cube, axis=1),
        np.nan,
    )
    hourly = pd.DataFrame(hourly_values, index=clock, columns=symbols)
    locations = {symbol: location for location, symbol in enumerate(symbols)}
    legs = np.asarray([(locations[s], locations[h]) for s, h in oriented], dtype=np.int64)
    if economic_inputs is not None:
        if not np.array_equal(legs, economic_inputs.pair_leg_indices):
            raise ValueError("economic pair-leg coordinates differ")
        weights = np.asarray(economic_inputs.pair_weights, dtype=np.float64)
        current_decision_available = np.asarray(
            economic_inputs.pair_decision_available, dtype=bool
        )
        target_cashflows = np.asarray(economic_inputs.pair_target_cashflows, dtype=np.float64)
        target_available = np.asarray(economic_inputs.pair_target_available, dtype=bool)
    else:
        weights = _rolling_pair_weights(
            hourly,
            pairs=oriented,
            lookback_hours=int(hedge_lookback_hours),
            minimum_hours=int(minimum_hedge_hours),
            beta_clip=beta_clip,
        )
        current_decision_available = np.isfinite(weights).all(axis=2)
        target_cashflows = np.zeros((len(clock), len(oriented)), dtype=np.float64)
        target_available = np.zeros((len(clock), len(oriented)), dtype=bool)

    endpoints = np.full((len(clock), len(oriented)), np.nan, dtype=np.float32)
    pair_path_available = np.zeros((len(clock), len(oriented)), dtype=bool)
    for pair_number, (signal, hedge) in enumerate(oriented):
        signal_values = hourly[signal].to_numpy(dtype=np.float64)
        hedge_values = hourly[hedge].to_numpy(dtype=np.float64)
        pair_weights = weights[:, pair_number]
        available = (
            np.isfinite(signal_values)
            & np.isfinite(hedge_values)
            & np.isfinite(pair_weights).all(axis=1)
            & current_decision_available[:, pair_number]
        )
        value = pair_weights[:, 0] * signal_values + pair_weights[:, 1] * hedge_values
        endpoints[available, pair_number] = value[available].astype(np.float32)
        pair_path_available[:, pair_number] = available
        if economic_inputs is None:
            target_available[:, pair_number] = available
            target_cashflows[available, pair_number] = value[available]

    feature_available = np.zeros_like(target_available)
    missing_prefix = np.vstack(
        (
            np.zeros((1, len(oriented)), dtype=np.int64),
            np.cumsum(~pair_path_available, axis=0, dtype=np.int64),
        )
    )
    rows = np.arange(ENDPOINT_HISTORY_HOURS, len(clock), dtype=np.int64)
    if len(rows):
        missing_in_history = (
            missing_prefix[rows] - missing_prefix[rows - ENDPOINT_HISTORY_HOURS]
        )
        feature_available[rows] = (
            (missing_in_history == 0) & current_decision_available[rows]
        )
    clean_endpoints = np.where(pair_path_available, endpoints, 0.0).astype(np.float32)
    clean_cube = np.where(np.isfinite(cube), cube, 0.0).transpose(0, 2, 1).astype(np.float32)
    clean_hourly = np.where(np.isfinite(hourly_values), hourly_values, 0.0).astype(np.float32)
    return PairedComponents(
        clock=clock,
        symbols=symbols,
        pair_ids=pair_ids,
        pair_leg_indices=legs,
        coin_minute_returns=clean_cube,
        coin_hourly_returns=clean_hourly,
        pair_weights=weights,
        pair_path_endpoints=clean_endpoints,
        target_cashflows=np.where(target_available, target_cashflows, 0.0),
        feature_available=feature_available,
        target_available=target_available,
    )


def save_components(
    path: Path,
    components: PairedComponents,
    *,
    provenance: Mapping[str, Any],
) -> None:
    """Write a factorized component cache atomically and create-only."""

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "paired_1m_factorized_components_v2",
        "clock_start": components.clock[0].isoformat(),
        "clock_end": components.clock[-1].isoformat(),
        "decision_semantics": "clock t is start of target interval [t,t+1h)",
        "feature_semantics": (
            "path episode t-1 uses w[t-1]; endpoint episodes [t-168h,t); "
            "current w[t]; no target-availability information"
        ),
        "target_semantics": (
            "exact fixed-quantity arithmetic open-to-next-open or terminal-close "
            "pair cashflow, excluding funding[t] and including boundary funding[t+1]"
        ),
        "development_cutoff_exclusive": DEVELOPMENT_END_EXCLUSIVE.isoformat(),
        "symbols": list(components.symbols),
        "pair_ids": list(components.pair_ids),
        "symbol_axis_sha256": axis_sha256(components.symbols),
        "pair_axis_sha256": axis_sha256(components.pair_ids),
        "provenance": dict(provenance),
    }
    temporary = destination.with_name(f".{destination.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        clock_ns=components.clock.asi8,
        symbols=np.asarray(components.symbols),
        pair_ids=np.asarray(components.pair_ids),
        pair_leg_indices=components.pair_leg_indices,
        coin_minute_returns=components.coin_minute_returns,
        coin_hourly_returns=components.coin_hourly_returns,
        pair_weights=components.pair_weights,
        pair_path_endpoints=components.pair_path_endpoints,
        target_cashflows=components.target_cashflows,
        feature_available=components.feature_available,
        target_available=components.target_available,
    )
    temporary.replace(destination)


def load_components(path: Path) -> tuple[PairedComponents, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as stored:
        required = {
            "metadata_json",
            "clock_ns",
            "symbols",
            "pair_ids",
            "pair_leg_indices",
            "coin_minute_returns",
            "coin_hourly_returns",
            "pair_weights",
            "pair_path_endpoints",
            "target_cashflows",
            "feature_available",
            "target_available",
        }
        if set(stored.files) != required:
            raise ValueError("factorized component cache members differ")
        metadata = json.loads(str(stored["metadata_json"].item()))
        if metadata.get("schema") != "paired_1m_factorized_components_v2":
            raise ValueError("factorized component cache schema differs")
        if metadata.get("development_cutoff_exclusive") != DEVELOPMENT_END_EXCLUSIVE.isoformat():
            raise ConfirmationFirewallError("component metadata cutoff differs")
        components = PairedComponents(
            clock=pd.to_datetime(stored["clock_ns"], utc=True),
            symbols=tuple(stored["symbols"].astype(str)),
            pair_ids=tuple(stored["pair_ids"].astype(str)),
            pair_leg_indices=stored["pair_leg_indices"],
            coin_minute_returns=stored["coin_minute_returns"],
            coin_hourly_returns=stored["coin_hourly_returns"],
            pair_weights=stored["pair_weights"],
            pair_path_endpoints=stored["pair_path_endpoints"],
            target_cashflows=stored["target_cashflows"],
            feature_available=stored["feature_available"],
            target_available=stored["target_available"],
        )
    if metadata.get("symbols") != list(components.symbols):
        raise ValueError("component symbol axis differs from metadata")
    if metadata.get("pair_ids") != list(components.pair_ids):
        raise ValueError("component pair axis differs from metadata")
    if metadata.get("symbol_axis_sha256") != axis_sha256(components.symbols):
        raise ValueError("component symbol-axis hash differs from metadata")
    if metadata.get("pair_axis_sha256") != axis_sha256(components.pair_ids):
        raise ValueError("component pair-axis hash differs from metadata")
    return components, metadata


def paired_feature_audit(components: PairedComponents, rows: Sequence[int]) -> dict[str, float | int]:
    """Assert that only the intended within-hour coordinates differ."""

    selected = np.asarray(rows, dtype=np.int64)
    native = components.features(selected, representation="native_1m")
    control = components.features(selected, representation="causal_15m_control")
    if not np.array_equal(native[:, :, PAIR_PATH_DIMENSION:], control[:, :, PAIR_PATH_DIMENSION:]):
        raise AssertionError("common feature tail differs between paired representations")
    native_blocks = native[:, :, :60].reshape(*native.shape[:2], 4, 15).sum(axis=-1)
    control_blocks = control[:, :, :60].reshape(*control.shape[:2], 4, 15).sum(axis=-1)
    if not np.allclose(native_blocks, control_blocks, atol=2e-9, rtol=2e-6):
        raise AssertionError("paired features do not preserve 15m endpoints")
    mask = components.feature_available[selected]
    differences = np.abs(native[:, :, :60] - control[:, :, :60])
    return {
        "rows": int(len(selected)),
        "pairs": int(len(components.pair_ids)),
        "available_cells": int(mask.sum()),
        "maximum_common_tail_difference": 0.0,
        "maximum_15m_endpoint_difference": float(np.max(np.abs(native_blocks - control_blocks))),
        "maximum_within_block_path_difference": float(differences.max(initial=0.0)),
    }


__all__ = [
    "ConfirmationFirewallError",
    "DEVELOPMENT_END_EXCLUSIVE",
    "ENDPOINT_HISTORY_HOURS",
    "HourlyEconomicInputs",
    "PAIR_PATH_DIMENSION",
    "ManifestArchiveRecord",
    "PairedComponents",
    "RAW_DIMENSION",
    "VerifiedAcquisitionManifest",
    "build_components_from_closes",
    "axis_sha256",
    "causal_15m_control",
    "development_months",
    "kline_archive_path",
    "load_acquisition_manifest",
    "load_components",
    "load_development_closes",
    "load_hourly_economic_inputs",
    "paired_feature_audit",
    "read_kline_month",
    "require_development_month",
    "save_components",
]
