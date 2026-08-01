#!/usr/bin/env python3
"""Summarize a deployment log JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_json)
    if not log_path.exists():
        raise SystemExit(f"Missing log file: {log_path}")

    payload = json.loads(log_path.read_text())
    if isinstance(payload, dict):
        summary = {key: type(value).__name__ for key, value in payload.items()}
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Loaded log payload type: {type(payload).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
