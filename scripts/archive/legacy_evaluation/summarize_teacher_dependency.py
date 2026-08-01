#!/usr/bin/env python3
"""Summarize teacher dependency audits by checkpoint and terrain family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _checkpoint_iter(name: str) -> int:
    stem = Path(name).stem
    for part in stem.split("_"):
        if part.startswith("model"):
            suffix = part.removeprefix("model")
            if suffix.isdigit():
                return int(suffix)
    if "model_" in stem:
        tail = stem.split("model_", 1)[1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return int(digits)
    return -1


def _extract(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if "results_by_mode" not in payload:
        return None
    return payload


def _winner(results_by_mode: dict, key: str, higher_is_better: bool) -> str:
    best_mode = ""
    best_val = None
    for mode, result in results_by_mode.items():
        val = result["summary_metrics"][key]
        if best_val is None or (val > best_val if higher_is_better else val < best_val):
            best_val = val
            best_mode = mode
    return best_mode


def _terrain_summary(payload: dict) -> dict:
    results = payload["results_by_mode"]
    pairwise = payload.get("pairwise_trace_diffs_vs_normal", {})
    return {
        "terrain_type": payload["terrain_type"],
        "checkpoint": Path(payload["checkpoint"]).name,
        "checkpoint_iter": _checkpoint_iter(payload["checkpoint"]),
        "winner_reward": _winner(results, "reward_step_mean", higher_is_better=True),
        "winner_vel_err": _winner(results, "vel_err_step_mean", higher_is_better=False),
        "winner_yaw_err": _winner(results, "yaw_err_step_mean", higher_is_better=False),
        "winner_tilt": _winner(results, "base_tilt_projected_gravity_xy_mean", higher_is_better=False),
        "normal_reward": results["normal"]["summary_metrics"]["reward_step_mean"],
        "normal_vel_err": results["normal"]["summary_metrics"]["vel_err_step_mean"],
        "normal_yaw_err": results["normal"]["summary_metrics"]["yaw_err_step_mean"],
        "normal_tilt": results["normal"]["summary_metrics"]["base_tilt_projected_gravity_xy_mean"],
        "zero_terrain_action_diff": pairwise.get("zero_terrain", {}).get("mean_action_abs_diff_vs_normal", 0.0),
        "zero_dynamics_action_diff": pairwise.get("zero_dynamics", {}).get("mean_action_abs_diff_vs_normal", 0.0),
        "zero_both_action_diff": pairwise.get("zero_both", {}).get("mean_action_abs_diff_vs_normal", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input_dir.glob("teacher_v3_dependency_audit_*_model_*.json")):
        payload = _extract(path)
        if payload is None:
            continue
        row = _terrain_summary(payload)
        row["source_file"] = path.name
        rows.append(row)

    rows.sort(key=lambda row: (row["checkpoint_iter"], row["terrain_type"]))
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "num_reports": len(rows),
        "rows": rows,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[INFO] Wrote JSON: {args.json_out}")
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
