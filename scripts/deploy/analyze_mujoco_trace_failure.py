#!/usr/bin/env python3
"""Analyze MuJoCo rollout traces for collapse/failure timing.

The runtime traces are large JSON files emitted by ``run_sim2sim.py`` with
``--trace-steps``. This script extracts the first useful failure indicators and
optionally compares a failing trace against a passing reference trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JOINT_NAMES = (
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)

RANGE_KEYS = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json", help="Trace JSON emitted by scripts/deploy/run_sim2sim.py.")
    parser.add_argument(
        "--compare-json",
        default="",
        help="Optional passing/reference trace JSON for side-by-side comparison.",
    )
    parser.add_argument("--height-threshold", type=float, default=0.30)
    parser.add_argument("--tilt-threshold", type=float, default=0.40)
    return parser.parse_args()


def load_trace(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(Path(path).read_text())
    runtime = report["runtime_rehearsal"]
    trace = runtime.get("trace", [])
    if not trace:
        raise ValueError(f"{path} does not contain runtime_rehearsal.trace")
    return report, runtime, trace


def first_step(trace: list[dict[str, Any]], predicate) -> int | None:
    for sample in trace:
        if predicate(sample):
            return int(sample["step"])
    return None


def joint_ranges(runtime: dict[str, Any]) -> list[tuple[float, float]]:
    ranges = runtime["model_diagnostics"]["joint_ranges_rad"]
    return [tuple(ranges[key]) for key in RANGE_KEYS]


def q_target_oob(sample: dict[str, Any], ranges: list[tuple[float, float]]) -> bool:
    return any(q < lo or q > hi for q, (lo, hi) in zip(sample["q_target"], ranges))


def joint_pos_oob(sample: dict[str, Any], ranges: list[tuple[float, float]]) -> bool:
    return any(q < lo or q > hi for q, (lo, hi) in zip(sample["joint_pos"], ranges))


def summarize(path: str | Path, height_threshold: float, tilt_threshold: float) -> dict[str, Any]:
    report, runtime, trace = load_trace(path)
    ranges = joint_ranges(runtime)
    n = len(trace)

    summary: dict[str, Any] = {
        "path": str(path),
        "scenario": report.get("scenario_name", ""),
        "command": report.get("command"),
        "control_dt": runtime["control_dt"],
        "steps": n,
        "height_mean": sum(t["root_height"] for t in trace) / n,
        "height_min": min(t["root_height"] for t in trace),
        "tilt_mean": sum(t["base_tilt_xy_norm"] for t in trace) / n,
        "tilt_max": max(t["base_tilt_xy_norm"] for t in trace),
        "nonfoot_frac": sum(t["contact_audit"]["non_foot_terrain_contact_count"] > 0 for t in trace) / n,
        "ctrl_sat_mean": sum(t["ctrl_saturation_frac"] for t in trace) / n,
        "ctrl_sat_max": max(t["ctrl_saturation_frac"] for t in trace),
        "action_abs_mean": sum(t["action_abs_mean"] for t in trace) / n,
        "action_abs_max": max(t["action_abs_mean"] for t in trace),
        "joint_vel_abs_max": max(t["joint_vel_abs_mean"] for t in trace),
        "vel_err_mean": sum(t["vel_err"] for t in trace) / n,
        "yaw_err_mean": sum(t["yaw_err"] for t in trace) / n,
        "first_height_low": first_step(trace, lambda t: t["root_height"] < height_threshold),
        "first_tilt_high": first_step(trace, lambda t: t["base_tilt_xy_norm"] > tilt_threshold),
        "first_nonfoot": first_step(trace, lambda t: t["contact_audit"]["non_foot_terrain_contact_count"] > 0),
        "first_ctrl_sat": first_step(trace, lambda t: t["ctrl_saturation_frac"] > 0),
        "first_qtarget_oob": first_step(trace, lambda t: q_target_oob(t, ranges)),
        "first_jointpos_oob": first_step(trace, lambda t: joint_pos_oob(t, ranges)),
        "foot_contact_fraction": {
            foot: sum(t["foot_contact"][foot] for t in trace) / n for foot in ("FL", "FR", "RL", "RR")
        },
        "per_joint": {},
    }

    for i, name in enumerate(JOINT_NAMES):
        lo, hi = ranges[i]
        summary["per_joint"][name] = {
            "action_abs_mean": sum(abs(t["action"][i]) for t in trace) / n,
            "action_abs_max": max(abs(t["action"][i]) for t in trace),
            "qtarget_violation_max": max(max(lo - t["q_target"][i], t["q_target"][i] - hi, 0.0) for t in trace),
            "ctrl_abs_mean": sum(abs(t["applied_ctrl"][i]) for t in trace) / n,
        }

    return summary


def format_step(step: int | None, control_dt: float) -> str:
    if step is None:
        return "None"
    return f"{step} ({step * control_dt:.2f}s)"


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Trace: {summary['path']}")
    print(f"Scenario: {summary['scenario']} command={summary['command']}")
    print(
        "Core: "
        f"h_mean={summary['height_mean']:.3f} h_min={summary['height_min']:.3f} "
        f"tilt_mean={summary['tilt_mean']:.3f} tilt_max={summary['tilt_max']:.3f} "
        f"nonfoot_frac={summary['nonfoot_frac']:.3f}"
    )
    print(
        "Control: "
        f"action_abs_mean={summary['action_abs_mean']:.3f} action_abs_max={summary['action_abs_max']:.3f} "
        f"ctrl_sat_mean={summary['ctrl_sat_mean']:.3f} ctrl_sat_max={summary['ctrl_sat_max']:.3f}"
    )
    for key in (
        "first_height_low",
        "first_tilt_high",
        "first_nonfoot",
        "first_ctrl_sat",
        "first_qtarget_oob",
        "first_jointpos_oob",
    ):
        print(f"{key}: {format_step(summary[key], summary['control_dt'])}")
    contacts = " ".join(f"{foot}={frac:.2f}" for foot, frac in summary["foot_contact_fraction"].items())
    print(f"Foot contact fraction: {contacts}")
    print("Top joints by q_target limit violation:")
    rows = sorted(
        summary["per_joint"].items(), key=lambda item: item[1]["qtarget_violation_max"], reverse=True
    )
    for name, stats in rows[:6]:
        print(
            f"  {name:9s} violation={stats['qtarget_violation_max']:.3f} "
            f"action_abs_mean={stats['action_abs_mean']:.3f} action_abs_max={stats['action_abs_max']:.3f}"
        )


def print_compare(target: dict[str, Any], reference: dict[str, Any]) -> None:
    print("\nComparison target - reference:")
    for key in (
        "height_mean",
        "height_min",
        "tilt_mean",
        "tilt_max",
        "nonfoot_frac",
        "action_abs_mean",
        "action_abs_max",
        "ctrl_sat_mean",
        "ctrl_sat_max",
        "vel_err_mean",
        "yaw_err_mean",
    ):
        print(f"{key:20s} target={target[key]:.3f} reference={reference[key]:.3f} delta={target[key]-reference[key]:+.3f}")

    print("\nLargest target-reference action_abs_mean deltas:")
    rows = []
    for name, target_stats in target["per_joint"].items():
        ref_stats = reference["per_joint"][name]
        rows.append((target_stats["action_abs_mean"] - ref_stats["action_abs_mean"], name))
    for delta, name in sorted(rows, reverse=True)[:8]:
        target_stats = target["per_joint"][name]
        ref_stats = reference["per_joint"][name]
        print(
            f"  {name:9s} delta={delta:+.3f} "
            f"target={target_stats['action_abs_mean']:.3f} reference={ref_stats['action_abs_mean']:.3f} "
            f"target_oob={target_stats['qtarget_violation_max']:.3f}"
        )


def main() -> None:
    args = parse_args()
    target = summarize(args.trace_json, args.height_threshold, args.tilt_threshold)
    print_summary(target)
    if args.compare_json:
        reference = summarize(args.compare_json, args.height_threshold, args.tilt_threshold)
        print_compare(target, reference)


if __name__ == "__main__":
    main()
