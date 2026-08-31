"""Verified reader for Binance's public USD-M futures data archives.

The public archive is useful for reproducible research because every ZIP has a
published SHA-256 sidecar.  This module deliberately implements only the two
datasets needed by the crypto signature experiments: 1-, 5- and 15-minute
USD-M klines plus funding rates.  Existing cache files are immutable; a hash
disagreement is an error rather than an invitation to silently replace the
research input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Literal, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


BINANCE_VISION_BASE_URL = "https://data.binance.vision/data"
_SYMBOL = re.compile(r"^[A-Z0-9]{5,24}$")
_CHECKSUM = re.compile(r"^([0-9a-fA-F]{64})\s+([^/\\]+)\s*$")
ArchiveDataset = Literal["klines", "fundingRate"]
KlineInterval = Literal["1m", "5m", "15m"]
SUPPORTED_KLINE_INTERVALS: tuple[KlineInterval, ...] = ("1m", "5m", "15m")
KLINE_ARCHIVE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def validate_kline_interval(interval: str) -> KlineInterval:
    """Return a supported Binance interval or fail before any I/O occurs."""

    if not isinstance(interval, str) or interval not in SUPPORTED_KLINE_INTERVALS:
        allowed = ", ".join(SUPPORTED_KLINE_INTERVALS)
        raise ValueError(
            f"unsupported kline interval {interval!r}; expected one of {allowed}"
        )
    return cast(KlineInterval, interval)


@dataclass(frozen=True)
class VerifiedArchive:
    """One verified immutable archive in the local cache."""

    dataset: ArchiveDataset
    symbol: str
    month: str
    path: Path
    sha256: str
    source_url: str


def month_labels(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[str, ...]:
    """Return inclusive ``YYYY-MM`` labels intersecting ``[start, end]``."""

    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    if first.tzinfo is not None:
        first = first.tz_convert("UTC").tz_localize(None)
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    if last < first:
        raise ValueError("end must not precede start")
    periods = pd.period_range(first.to_period("M"), last.to_period("M"), freq="M")
    return tuple(str(period) for period in periods)


def archive_relative_path(
    symbol: str,
    month: str,
    *,
    dataset: ArchiveDataset,
    interval: str = "15m",
) -> Path:
    """Return the official relative archive path after strict validation."""

    normalized_interval = validate_kline_interval(interval)
    normalized_symbol = str(symbol).upper()
    if not _SYMBOL.fullmatch(normalized_symbol):
        raise ValueError(f"invalid Binance symbol: {symbol!r}")
    try:
        parsed_month = pd.Period(month, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid month: {month!r}") from exc
    month_label = str(parsed_month)
    if dataset == "klines":
        filename = f"{normalized_symbol}-{normalized_interval}-{month_label}.zip"
        return Path(
            "futures",
            "um",
            "monthly",
            "klines",
            normalized_symbol,
            normalized_interval,
            filename,
        )
    if dataset == "fundingRate":
        filename = f"{normalized_symbol}-fundingRate-{month_label}.zip"
        return Path(
            "futures",
            "um",
            "monthly",
            "fundingRate",
            normalized_symbol,
            filename,
        )
    raise ValueError(f"unsupported archive dataset: {dataset!r}")


def sha256_file(path: Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum(text: str, *, expected_filename: str) -> str:
    """Parse a Binance checksum and reject an unexpected basename."""

    match = _CHECKSUM.fullmatch(text.strip())
    if match is None:
        raise ValueError("malformed Binance CHECKSUM sidecar")
    digest, filename = match.groups()
    if filename != expected_filename:
        raise ValueError(
            f"checksum filename {filename!r} does not match {expected_filename!r}"
        )
    return digest.lower()


def _fetch(url: str, *, timeout_seconds: float) -> bytes:
    request = Request(url, headers={"User-Agent": "pairs-research/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_month_archive(
    cache_directory: Path,
    symbol: str,
    month: str,
    *,
    dataset: ArchiveDataset,
    interval: str = "15m",
    timeout_seconds: float = 60.0,
    allow_missing: bool = False,
) -> VerifiedArchive | None:
    """Download, verify and immutably cache one official monthly archive.

    If both cached files already exist they are verified and reused.  A cached
    hash mismatch raises immediately.  ``allow_missing`` only converts a
    remote HTTP 404 into ``None``; authentication, transport and integrity
    failures remain hard errors.
    """

    relative = archive_relative_path(
        symbol, month, dataset=dataset, interval=interval
    )
    target = Path(cache_directory).expanduser().resolve() / relative
    checksum_target = target.with_name(f"{target.name}.CHECKSUM")
    source_url = f"{BINANCE_VISION_BASE_URL}/{relative.as_posix()}"
    checksum_url = f"{source_url}.CHECKSUM"

    if target.exists() != checksum_target.exists():
        raise ValueError(f"incomplete immutable cache entry: {target}")
    if target.exists():
        expected = parse_checksum(
            checksum_target.read_text(encoding="utf-8"),
            expected_filename=target.name,
        )
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"cached archive checksum mismatch: {target}")
        return VerifiedArchive(
            dataset=dataset,
            symbol=str(symbol).upper(),
            month=str(pd.Period(month, freq="M")),
            path=target,
            sha256=actual,
            source_url=source_url,
        )

    try:
        checksum_payload = _fetch(checksum_url, timeout_seconds=timeout_seconds)
    except HTTPError as exc:
        if allow_missing and exc.code == 404:
            return None
        raise
    checksum_text = checksum_payload.decode("ascii")
    expected = parse_checksum(checksum_text, expected_filename=target.name)
    payload = _fetch(source_url, timeout_seconds=timeout_seconds)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"downloaded archive checksum mismatch: {source_url}")

    # Validate the container before publishing it into the immutable cache.
    try:
        with ZipFile(BytesIO(payload)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1:
                raise ValueError("Binance archive must contain exactly one CSV")
            member = members[0]
            if Path(member.filename).name != member.filename:
                raise ValueError("unsafe path in Binance ZIP archive")
            if not member.filename.endswith(".csv"):
                raise ValueError("Binance archive member must be a CSV")
    except BadZipFile as exc:
        raise ValueError("downloaded Binance archive is not a valid ZIP") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(target, payload)
    _atomic_bytes(checksum_target, checksum_text.encode("ascii"))
    return VerifiedArchive(
        dataset=dataset,
        symbol=str(symbol).upper(),
        month=str(pd.Period(month, freq="M")),
        path=target,
        sha256=actual,
        source_url=source_url,
    )


def read_single_csv(archive: VerifiedArchive) -> pd.DataFrame:
    """Read the sole CSV member of an already verified archive."""

    if sha256_file(archive.path) != archive.sha256:
        raise ValueError(f"archive changed after verification: {archive.path}")
    with ZipFile(archive.path) as container:
        members = [member for member in container.infolist() if not member.is_dir()]
        if len(members) != 1:
            raise ValueError("Binance archive must contain exactly one CSV")
        member = members[0]
        if Path(member.filename).name != member.filename:
            raise ValueError("unsafe path in Binance ZIP archive")
        with container.open(member) as handle:
            frame = pd.read_csv(handle)
        # Official USD-M kline archives through 2021 are headerless, whereas
        # later monthly archives contain the canonical header.  A default
        # ``read_csv`` would silently consume the first historical bar as the
        # column names and concatenation would then produce thousands of NaNs.
        # Detect only the exact official 12-field kline shape and re-read it;
        # every other malformed schema remains a downstream hard error.
        if archive.dataset == "klines" and "open_time" not in frame.columns:
            if len(frame.columns) != len(KLINE_ARCHIVE_COLUMNS):
                raise ValueError(
                    "headerless Binance kline archive has an unexpected field count"
                )
            with container.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    header=None,
                    names=KLINE_ARCHIVE_COLUMNS,
                )
        return frame


def download_archive_range(
    cache_directory: Path,
    symbols: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    datasets: Iterable[ArchiveDataset] = ("klines", "fundingRate"),
    interval: str = "15m",
) -> tuple[VerifiedArchive, ...]:
    """Download a deterministic symbol/month/dataset range."""

    normalized_interval = validate_kline_interval(interval)
    results: list[VerifiedArchive] = []
    for symbol in tuple(str(value).upper() for value in symbols):
        for month in month_labels(start, end):
            for dataset in tuple(datasets):
                item = download_month_archive(
                    cache_directory,
                    symbol,
                    month,
                    dataset=dataset,
                    interval=normalized_interval,
                    allow_missing=False,
                )
                if item is not None:
                    results.append(item)
    return tuple(results)
