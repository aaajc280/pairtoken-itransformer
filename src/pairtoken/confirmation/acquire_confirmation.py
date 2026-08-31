#!/usr/bin/env python3
"""Acquire the exact March-2024--June-2026 Top-24 archive allowlist.

The network-capable operation is delegated to the SHA-256-pinned Binance
Vision downloader from the audited source snapshot.  This wrapper never opens
ZIP bodies.  It verifies published checksums again and emits a create-only,
content-addressed acquisition receipt consumed by ``prepare_confirmation.py``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import pandas as pd

from acquisition_plan import validate_plan
from confirmation_governance import (
    ConfirmationGateError,
    CONTRACT_PATH,
    canonical_json,
    load_contract,
    load_json,
    sha256_file,
    verify_source_freeze,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_AUDITED_ROOT = (
    WORKSPACE / "dependencies" / "pairs_research"
)
DOWNLOADER_SHA256 = "fcb5b20886e9dbcf7fd9a7880ce1e7f3aca4e7653587a3329d53a09f0c40d3de"
SCHEMA = "pairtoken_1m_confirmation_acquisition_v1"
MAX_WORKERS = 16
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Request:
    ordinal: int
    liquidity_rank: int
    symbol: str
    month: str
    dataset: str
    interval: str


@dataclass(frozen=True)
class Record:
    ordinal: int
    liquidity_rank: int
    symbol: str
    month: str
    dataset: str
    interval: str
    availability: str
    cache_disposition: str
    archive_relative_path: str
    archive_sha256: str
    official_checksum_sha256: str
    checksum_relative_path: str
    checksum_sidecar_sha256: str
    source_url: str


def verify_overlap_receipt(path: Path, source_freeze_path: Path) -> dict[str, Any]:
    receipt = load_json(path, label="development overlap receipt")
    required = {
        "schema", "status", "contract_sha256", "source_freeze_sha256",
        "development_start", "development_end_inclusive", "confirmation_rows_parsed",
        "exact_masks", "maximum_absolute_differences", "feature_rows_compared",
        "ledger_hours_compared", "audited_source_hashes",
    }
    if set(receipt) != required or receipt.get("schema") != (
        "pairtoken_1m_confirmation_development_overlap_v1"
    ):
        raise ConfirmationGateError("development overlap receipt schema differs")
    if (
        receipt.get("status") != "passed_exact_parent_parity_before_confirmation_firewall"
        or receipt.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or receipt.get("source_freeze_sha256") != sha256_file(source_freeze_path)
        or receipt.get("confirmation_rows_parsed") is not False
        or receipt.get("development_end_inclusive") != "2025-06-30T23:00:00+00:00"
        or not isinstance(receipt.get("exact_masks"), dict)
        or not all(value is True for value in receipt["exact_masks"].values())
        or not isinstance(receipt.get("maximum_absolute_differences"), dict)
        or any(float(value) != 0.0 for value in receipt["maximum_absolute_differences"].values())
        or int(receipt.get("feature_rows_compared", 0)) < 1
        or int(receipt.get("ledger_hours_compared", 0)) < 2
    ):
        raise ConfirmationGateError("development overlap parity did not pass exactly")
    sources = receipt.get("audited_source_hashes")
    if not isinstance(sources, dict) or not sources or any(
        SHA256_RE.fullmatch(str(value)) is None for value in sources.values()
    ):
        raise ConfirmationGateError("development overlap source binding differs")
    return receipt


def _audited_downloader(root: Path) -> Any:
    source_root = Path(root).resolve() / "src"
    source = source_root / "pairs_research/binance_vision.py"
    if not source.is_file() or sha256_file(source) != DOWNLOADER_SHA256:
        raise ConfirmationGateError("audited Binance Vision downloader differs")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module = importlib.import_module("pairs_research.binance_vision")
    if Path(module.__file__).resolve() != source.resolve():
        raise ConfirmationGateError("Binance Vision import escaped the audited snapshot")
    return module


def requests_from_plan() -> tuple[Request, ...]:
    _plan, rows = validate_plan()
    ranks = {
        symbol: rank
        for rank, symbol in enumerate(load_contract()["data"]["symbols_in_order"], start=1)
    }
    result = tuple(
        Request(
            ordinal=number,
            liquidity_rank=ranks[row["symbol"]],
            symbol=row["symbol"],
            month=row["month"],
            dataset=row["dataset"],
            interval=row["interval"],
        )
        for number, row in enumerate(rows, start=1)
    )
    if len(result) != 1344 or tuple(row.ordinal for row in result) != tuple(range(1, 1345)):
        raise ConfirmationGateError("acquisition request family differs")
    if any(row.month < "2024-03" or row.month > "2026-06" for row in result):
        raise ConfirmationGateError("acquisition request crosses its hard month boundary")
    return result


def _acquire_one(
    request: Request,
    *,
    cache_root: Path,
    vision: Any,
    timeout_seconds: float,
    attempts: int,
) -> Record:
    if request.month > "2026-06":
        raise ConfirmationGateError("July-2026 archive request is forbidden")
    interval = "1m"
    relative = vision.archive_relative_path(
        request.symbol, request.month, dataset=request.dataset, interval=interval
    )
    target = cache_root / relative
    sidecar = target.with_name(f"{target.name}.CHECKSUM")
    existed = target.exists()
    archive = None
    for attempt in range(1, attempts + 1):
        try:
            archive = vision.download_month_archive(
                cache_root,
                request.symbol,
                request.month,
                dataset=request.dataset,
                interval=interval,
                timeout_seconds=timeout_seconds,
                allow_missing=True,
            )
            break
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 30))
    common = dict(
        ordinal=request.ordinal,
        liquidity_rank=request.liquidity_rank,
        symbol=request.symbol,
        month=request.month,
        dataset=request.dataset,
        interval=request.interval,
    )
    source_url = f"{vision.BINANCE_VISION_BASE_URL}/{relative.as_posix()}"
    if archive is None:
        if target.exists() or sidecar.exists():
            raise ConfirmationGateError("missing archive left a partial cache entry")
        return Record(
            **common,
            availability="missing_remote_404",
            cache_disposition="not_available",
            archive_relative_path="",
            archive_sha256="",
            official_checksum_sha256="",
            checksum_relative_path="",
            checksum_sidecar_sha256="",
            source_url=source_url,
        )
    if Path(archive.path).resolve() != target.resolve() or not sidecar.is_file():
        raise ConfirmationGateError("downloader returned an unexpected archive")
    observed = sha256_file(target)
    official = vision.parse_checksum(
        sidecar.read_text(encoding="ascii"), expected_filename=target.name
    )
    if observed != str(archive.sha256) or observed != official:
        raise ConfirmationGateError("archive differs from its official checksum")
    return Record(
        **common,
        availability="available_verified",
        cache_disposition="verified_cached" if existed else "verified_downloaded",
        archive_relative_path=relative.as_posix(),
        archive_sha256=observed,
        official_checksum_sha256=official,
        checksum_relative_path=(relative.parent / f"{relative.name}.CHECKSUM").as_posix(),
        checksum_sidecar_sha256=sha256_file(sidecar),
        source_url=source_url,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    import csv

    output = io.StringIO(newline="")
    if not rows:
        raise ConfirmationGateError("cannot serialize an empty acquisition table")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def publish(
    output: Path,
    *,
    cache_root: Path,
    source_freeze_path: Path,
    audited_root: Path,
    overlap_receipt_path: Path,
    requests: Sequence[Request],
    records: Sequence[Record],
) -> Path:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        request_rows = [asdict(row) for row in requests]
        record_rows = [asdict(row) for row in records]
        available_rows = [row for row in record_rows if row["availability"] == "available_verified"]
        freeze = verify_source_freeze(source_freeze_path)
        manifest = {
            "schema": SCHEMA,
            "status": "complete_exact_allowlist_archives_hash_verified_no_csv_parsed",
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "source_freeze_sha256": sha256_file(source_freeze_path),
            "source_bundle_sha256": freeze["source_bundle_sha256"],
            "downloader_source_sha256": DOWNLOADER_SHA256,
            "wrapper_source_sha256": sha256_file(Path(__file__)),
            "cache_root": str(Path(cache_root).resolve()),
            "first_month": "2024-03",
            "last_month": "2026-06",
            "request_count": 1344,
            "available_verified_count": len(available_rows),
            "missing_remote_404_count": len(records) - len(available_rows),
            "archive_csv_bodies_parsed": False,
            "july_2026_or_later_path_count": 0,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "development_overlap_receipt_sha256": sha256_file(overlap_receipt_path),
        }
        payloads = {
            "requested_inventory.csv": _csv_bytes(request_rows),
            "archive_availability.csv": _csv_bytes(record_rows),
            "archive_manifest.csv": _csv_bytes(available_rows),
            "acquisition_manifest.json": canonical_json(manifest),
            "development_overlap_receipt.json": Path(overlap_receipt_path).read_bytes(),
        }
        for name, payload in payloads.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
        hashes = {name: sha256_file(staging / name) for name in sorted(payloads)}
        hashes_payload = canonical_json({"schema": SCHEMA, "files": hashes})
        (staging / "artifact_hashes.json").write_bytes(hashes_payload)
        (staging / "artifact_hashes.sha256").write_text(
            f"{sha256_file(staging / 'artifact_hashes.json')}  artifact_hashes.json\n",
            encoding="ascii",
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--overlap-receipt", type=Path, required=True)
    parser.add_argument("--audited-root", type=Path, default=DEFAULT_AUDITED_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    verify_source_freeze(args.source_freeze)
    verify_overlap_receipt(args.overlap_receipt, args.source_freeze)
    if not 1 <= args.workers <= MAX_WORKERS or not 1 <= args.attempts <= 20:
        raise ConfirmationGateError("worker or retry budget is invalid")
    requests = requests_from_plan()
    vision = _audited_downloader(args.audited_root)
    cache = args.cache_root.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    def work(row: Request) -> Record:
        return _acquire_one(
            row,
            cache_root=cache,
            vision=vision,
            timeout_seconds=float(args.timeout_seconds),
            attempts=int(args.attempts),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = tuple(executor.map(work, requests))
    if tuple(row.ordinal for row in records) != tuple(row.ordinal for row in requests):
        raise ConfirmationGateError("parallel acquisition changed ordering")
    output = publish(
        args.output,
        cache_root=cache,
        source_freeze_path=args.source_freeze,
        audited_root=args.audited_root,
        overlap_receipt_path=args.overlap_receipt,
        requests=requests,
        records=records,
    )
    print(json.dumps({
        "output": str(output),
        "requested": len(records),
        "available_verified": sum(row.availability == "available_verified" for row in records),
        "latest_month": "2026-06",
        "archive_csv_bodies_parsed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
