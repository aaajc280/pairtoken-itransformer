#!/usr/bin/env python3
"""Seal the exact 32-artifact N02/C02 confirmation forecast family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from confirmation_governance import seal_all_forecasts, verify_all_forecast_seal


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--forecast-root", type=Path, required=True)
    create.add_argument("--source-freeze", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--forecast-root", type=Path, required=True)
    verify.add_argument("--source-freeze", type=Path, required=True)
    verify.add_argument("--seal", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    if args.command == "create":
        payload = seal_all_forecasts(
            args.forecast_root,
            source_freeze_path=args.source_freeze,
            output=args.output,
        )
    else:
        payload = verify_all_forecast_seal(
            args.forecast_root,
            source_freeze_path=args.source_freeze,
            seal_path=args.seal,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
