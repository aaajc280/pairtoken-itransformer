#!/usr/bin/env python3
"""Validate and expand the static confirmation acquisition plan without I/O.

The script creates an exact requested-inventory CSV.  It performs no network
request, filesystem archive discovery, download, archive open, or data parse.
"""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from confirmation_governance import (
    ConfirmationGateError,
    atomic_create,
    canonical_json,
    load_contract,
    load_json,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "acquisition_plan.json"


def months(first: str, last: str) -> tuple[str, ...]:
    year, month = map(int, first.split("-"))
    end_year, end_month = map(int, last.split("-"))
    values: list[str] = []
    while (year, month) <= (end_year, end_month):
        values.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return tuple(values)


def validate_plan() -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    contract = load_contract()
    plan = load_json(PLAN_PATH, label="acquisition plan")
    if plan.get("schema") != "pairtoken_1m_confirmation_acquisition_plan_v1":
        raise ConfirmationGateError("acquisition-plan schema differs")
    if plan.get("status") != "plan_only_no_archive_was_downloaded_opened_or_parsed":
        raise ConfirmationGateError("acquisition plan is not result blind")
    labels = months(plan["first_archive_month"], plan["last_archive_month"])
    if labels[0] != "2024-03" or labels[-1] != "2026-06" or len(labels) != 28:
        raise ConfirmationGateError("acquisition month range differs")
    if any(label >= "2026-07" for label in labels):
        raise ConfirmationGateError("acquisition plan reaches July 2026")
    symbols = tuple(plan["symbols_in_order"])
    if symbols != tuple(contract["data"]["symbols_in_order"]):
        raise ConfirmationGateError("acquisition and contract symbol axes differ")
    rows = tuple(
        {
            "symbol": symbol,
            "month": month,
            "dataset": dataset,
            "interval": "1m" if dataset == "klines" else "",
        }
        for symbol in symbols
        for month in labels
        for dataset in ("klines", "fundingRate")
    )
    if len(rows) != 1344 or plan.get("dataset_symbol_month_requests") != len(rows):
        raise ConfirmationGateError("acquisition request count differs")
    return plan, rows


def render_inventory(rows: Sequence[Mapping[str, str]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=("symbol", "month", "dataset", "interval"), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    plan, rows = validate_plan()
    payload = render_inventory(rows)
    if args.output is not None:
        atomic_create(args.output, payload)
    print(json.dumps({
        "plan_sha256": sha256_file(PLAN_PATH),
        "requests": len(rows),
        "first_month": rows[0]["month"],
        "last_month": rows[-1]["month"],
        "output_written": str(args.output) if args.output is not None else None,
        "archive_or_network_access": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
