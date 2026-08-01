#!/usr/bin/env python3
"""Summarize history-ablation suite outputs into a compact decision view."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _suite_jsons(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("isolated_suite_*.json") if "/tmp/" not in str(path))


def _scenario_rows(suite_blob: dict) -> list[dict]:
    rows = suite_blob.get("results")
    if isinstance(rows, list):
        return rows
    rows = suite_blob.get("scenarios", [])
    if isinstance(rows, list):
        return rows
    return []


def _metric(row: dict, key: str, default=None):
    if key in row:
        return row[key]
    post = row.get("post_switch_recovery_metrics") or {}
    return post.get(key, default)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    reports = []
    for path in _suite_jsons(args.input_dir.resolve()):
        blob = json.loads(path.read_text())
        grouped = defaultdict(dict)
        for row in _scenario_rows(blob):
            name = row.get("scenario_name") or row.get("scenario")
            if not name:
                continue
            mode = row.get("history_ablation")
            if not mode:
                continue
            base_name = name
            for suffix in ("_normal", "_zero", "_frozen"):
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break
            grouped[base_name][mode] = {
                "score": row.get("score"),
                "post_switch_peak_vel_err": _metric(row, "post_switch_peak_vel_err"),
                "post_switch_peak_yaw_err": _metric(row, "post_switch_peak_yaw_err"),
                "post_switch_peak_base_tilt": _metric(row, "post_switch_peak_base_tilt"),
                "vel_err_recovery_step": _metric(row, "vel_err_recovery_step"),
                "yaw_err_recovery_step": _metric(row, "yaw_err_recovery_step"),
                "base_tilt_recovery_step": _metric(row, "base_tilt_recovery_step"),
            }

        report = {
            "suite_file": str(path),
            "status": blob.get("status"),
            "expected_scenario_count": blob.get("expected_scenario_count"),
            "scenario_count": len(grouped),
            "scenarios": grouped,
        }
        reports.append(report)

    output = {"reports": reports}
    text = json.dumps(output, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
