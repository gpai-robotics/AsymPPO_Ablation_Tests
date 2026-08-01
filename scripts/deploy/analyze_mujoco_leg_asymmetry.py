#!/usr/bin/env python3
"""Analyze fixed and mirrored leg asymmetry from MuJoCo suite rollout JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


JOINT_PAIRS = (
    ("front_hip", "FR_hip_joint", "FL_hip_joint"),
    ("front_thigh", "FR_thigh_joint", "FL_thigh_joint"),
    ("front_calf", "FR_calf_joint", "FL_calf_joint"),
    ("rear_hip", "RL_hip_joint", "RR_hip_joint"),
    ("rear_thigh", "RL_thigh_joint", "RR_thigh_joint"),
    ("rear_calf", "RL_calf_joint", "RR_calf_joint"),
)

MIRRORED_SCENARIOS = (
    ("lateral", "asym_lateral_left", "asym_lateral_right"),
    ("yaw", "asym_yaw_left", "asym_yaw_right"),
    ("push", "asym_push_left", "asym_push_right"),
)

METRICS = (
    "action_abs_mean",
    "q_target_err_abs_mean",
    "ctrl_abs_mean",
    "joint_vel_abs_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1.0e-9:
        return None
    return numerator / denominator


def _load_scenario(scenario_dir: Path) -> dict[str, object]:
    per_joint: dict[str, dict[str, list[float]]] = {}
    gravity_x: list[float] = []
    gravity_y: list[float] = []
    base_height: list[float] = []
    non_foot_contact: list[float] = []
    contact_bodies: dict[str, int] = {}
    rollout_count = 0

    for path in sorted(scenario_dir.glob("rollout_*.json")):
        payload = json.loads(path.read_text())
        runtime = payload.get("runtime_rehearsal") or {}
        summary = runtime.get("summary_metrics") or {}
        joints = summary.get("per_joint") or {}
        if not joints:
            continue
        rollout_count += 1
        gravity_x.append(float(summary.get("projected_gravity_x_mean") or 0.0))
        gravity_y.append(float(summary.get("projected_gravity_y_mean") or 0.0))
        base_height.append(float(summary.get("base_height_mean") or 0.0))
        non_foot_contact.append(float(summary.get("non_foot_terrain_contact_step_fraction") or 0.0))
        for joint_name, metrics in joints.items():
            joint_store = per_joint.setdefault(joint_name, {metric: [] for metric in METRICS})
            for metric in METRICS:
                joint_store[metric].append(float(metrics.get(metric) or 0.0))
        for item in summary.get("top_non_foot_terrain_contact_pairs") or []:
            body = str(item.get("pair", "")).split(":", 1)[0]
            contact_bodies[body] = contact_bodies.get(body, 0) + int(item.get("count") or 0)

    joint_means = {
        joint_name: {metric: _mean(values) for metric, values in metrics.items()}
        for joint_name, metrics in per_joint.items()
    }
    pair_comparisons = {}
    for label, numerator_name, denominator_name in JOINT_PAIRS:
        numerator = joint_means.get(numerator_name, {})
        denominator = joint_means.get(denominator_name, {})
        pair_comparisons[label] = {
            "numerator_joint": numerator_name,
            "denominator_joint": denominator_name,
            **{
                f"{metric}_ratio": _ratio(numerator.get(metric), denominator.get(metric))
                for metric in METRICS
            },
        }

    return {
        "rollout_count": rollout_count,
        "projected_gravity_x_mean": _mean(gravity_x),
        "projected_gravity_y_mean": _mean(gravity_y),
        "base_height_mean": _mean(base_height),
        "non_foot_terrain_contact_step_fraction": _mean(non_foot_contact),
        "joint_means": joint_means,
        "pair_comparisons": pair_comparisons,
        "top_non_foot_contact_bodies": [
            {"body": body, "count": count}
            for body, count in sorted(contact_bodies.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
    }


def main() -> int:
    args = parse_args()
    scenario_root = args.suite_dir / "scenario_runs"
    scenarios = {
        path.name: _load_scenario(path)
        for path in sorted(scenario_root.iterdir())
        if path.is_dir()
    }
    mirrored = {}
    for label, positive_name, negative_name in MIRRORED_SCENARIOS:
        positive = scenarios.get(positive_name)
        negative = scenarios.get(negative_name)
        if positive is None or negative is None:
            continue
        mirrored[label] = {
            "positive_scenario": positive_name,
            "negative_scenario": negative_name,
            "front_thigh_ctrl_ratio_delta": (
                (positive["pair_comparisons"]["front_thigh"]["ctrl_abs_mean_ratio"] or 0.0)
                - (negative["pair_comparisons"]["front_thigh"]["ctrl_abs_mean_ratio"] or 0.0)
            ),
            "front_thigh_error_ratio_delta": (
                (positive["pair_comparisons"]["front_thigh"]["q_target_err_abs_mean_ratio"] or 0.0)
                - (negative["pair_comparisons"]["front_thigh"]["q_target_err_abs_mean_ratio"] or 0.0)
            ),
            "projected_gravity_y_sum": (
                (positive["projected_gravity_y_mean"] or 0.0)
                + (negative["projected_gravity_y_mean"] or 0.0)
            ),
        }

    report = {
        "suite_dir": str(args.suite_dir),
        "interpretation": {
            "ratio": "Values above 1.0 mean the numerator joint works harder or tracks worse than its mirror.",
            "fixed_asymmetry": "A ratio remaining above 1.0 in both mirrored directions suggests fixed policy/model asymmetry.",
            "load_dependent": "A ratio that crosses 1.0 when direction is mirrored suggests expected load-dependent behavior.",
        },
        "scenarios": scenarios,
        "mirrored_pair_deltas": mirrored,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
