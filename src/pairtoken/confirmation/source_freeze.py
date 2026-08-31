#!/usr/bin/env python3
"""Create or verify the result-blind source freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from confirmation_governance import create_source_freeze, verify_source_freeze


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    payload = (
        create_source_freeze(args.output)
        if args.command == "create"
        else verify_source_freeze(args.receipt)
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
