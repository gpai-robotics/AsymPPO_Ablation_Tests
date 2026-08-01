#!/usr/bin/env python3
"""Summarize `go2_ctrl` text logs into a compact deployment scorecard."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"

VELOCITY_CMD_RE = re.compile(
    rf"VelocityCmd vx/vy/wz raw=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"target=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"filtered=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"lin_vel=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"imu_wz={FLOAT_RE} blend_alpha={FLOAT_RE}"
)

OBS_BASE_RE = re.compile(
    rf"ObsDiag policy_obs base_ang=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"gravity=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"cmd=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\]"
)

JOINT_ERR_RE = re.compile(
    rf"JointDiag err "
    rf"FL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"FR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"side_abs_err L=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"R=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\]"
)

JOINT_GENERIC_RE = re.compile(
    rf"JointDiag (?P<label>raw_action|rel_cmd|rel_pos) "
    rf"FL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"FR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\]"
)

FSM_RE = re.compile(r"FSM: Change state from (?P<old>\w+) to (?P<new>\w+)")


@dataclass
class WindowStats:
    count: int
    mean_abs: float
    max_abs: float
    p95_abs: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, help="Path to raw go2_ctrl text log.")
    parser.add_argument("--json-out", help="Optional output path for machine-readable JSON summary.")
    return parser.parse_args()


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    frac = rank - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _window_stats(values: list[float]) -> WindowStats:
    if not values:
        return WindowStats(count=0, mean_abs=0.0, max_abs=0.0, p95_abs=0.0)
    abs_values = sorted(abs(value) for value in values)
    return WindowStats(
        count=len(values),
        mean_abs=_mean(abs_values),
        max_abs=abs_values[-1],
        p95_abs=_percentile(abs_values, 0.95),
    )


def _subset_stats(entries: list[dict[str, Any]], value_key: str, predicate: Any) -> dict[str, float]:
    return asdict(_window_stats([entry[value_key] for entry in entries if predicate(entry)]))


def _signed_window(values: list[dict[str, Any]], key: str, threshold: float = 0.05) -> dict[str, Any]:
    positive = [entry[key] for entry in values if entry[key] >= threshold]
    negative = [entry[key] for entry in values if entry[key] <= -threshold]
    near_zero = [entry[key] for entry in values if abs(entry[key]) < threshold]
    return {
        "positive": asdict(_window_stats(positive)),
        "negative": asdict(_window_stats(negative)),
        "near_zero": asdict(_window_stats(near_zero)),
    }


def _extract_floats(match: re.Match[str], start: int = 1) -> list[float]:
    return [float(group) for group in match.groups()[start - 1 :]]


def main() -> int:
    args = _parse_args()
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise SystemExit(f"Missing log file: {log_path}")

    velocity_cmds: list[dict[str, Any]] = []
    obs_samples: list[dict[str, Any]] = []
    joint_errs: list[dict[str, Any]] = []
    joint_generic: dict[str, list[dict[str, Any]]] = {
        "raw_action": [],
        "rel_cmd": [],
        "rel_pos": [],
    }
    fsm_transitions: list[dict[str, str]] = []

    for line in log_path.read_text().splitlines():
        if match := VELOCITY_CMD_RE.search(line):
            values = _extract_floats(match)
            velocity_cmds.append(
                {
                    "raw": values[0:3],
                    "target": values[3:6],
                    "filtered": values[6:9],
                    "lin_vel": values[9:12],
                    "imu_wz": values[12],
                    "blend_alpha": values[13],
                }
            )
            continue

        if match := OBS_BASE_RE.search(line):
            values = _extract_floats(match)
            obs_samples.append(
                {
                    "base_ang": values[0:3],
                    "gravity": values[3:6],
                    "cmd": values[6:9],
                }
            )
            continue

        if match := JOINT_ERR_RE.search(line):
            values = _extract_floats(match)
            joint_errs.append(
                {
                    "FL": values[0:3],
                    "FR": values[3:6],
                    "RL": values[6:9],
                    "RR": values[9:12],
                    "side_L": values[12:15],
                    "side_R": values[15:18],
                }
            )
            continue

        if match := JOINT_GENERIC_RE.search(line):
            label = match.group("label")
            values = [float(group) for group in match.groups()[1:]]
            joint_generic[label].append(
                {
                    "FL": values[0:3],
                    "FR": values[3:6],
                    "RL": values[6:9],
                    "RR": values[9:12],
                }
            )
            continue

        if match := FSM_RE.search(line):
            fsm_transitions.append({"from": match.group("old"), "to": match.group("new")})

    filtered_vx = [entry["filtered"][0] for entry in velocity_cmds]
    filtered_vy = [entry["filtered"][1] for entry in velocity_cmds]
    filtered_wz = [entry["filtered"][2] for entry in velocity_cmds]
    lin_vx = [entry["lin_vel"][0] for entry in velocity_cmds]
    lin_vy = [entry["lin_vel"][1] for entry in velocity_cmds]
    lin_wz = [entry["imu_wz"] for entry in velocity_cmds]
    zero_cmd_threshold_xy = 0.05
    zero_cmd_threshold_wz = 0.05
    drift_samples = [
        {
            "filtered_vx": entry["filtered"][0],
            "filtered_vy": entry["filtered"][1],
            "filtered_wz": entry["filtered"][2],
            "lin_vx": entry["lin_vel"][0],
            "lin_vy": entry["lin_vel"][1],
            "imu_wz": entry["imu_wz"],
            "lin_vel_xy_norm": math.sqrt(entry["lin_vel"][0] ** 2 + entry["lin_vel"][1] ** 2),
        }
        for entry in velocity_cmds
    ]

    joint_side_bias = []
    for entry in joint_errs:
        joint_side_bias.append(
            {
                "x": _mean(entry["side_L"][0:1]) - _mean(entry["side_R"][0:1]),
                "y": _mean(entry["side_L"][1:2]) - _mean(entry["side_R"][1:2]),
                "z": _mean(entry["side_L"][2:3]) - _mean(entry["side_R"][2:3]),
            }
        )

    per_leg_err_norms = []
    per_leg_axis_abs = {leg: [[], [], []] for leg in ("FL", "FR", "RL", "RR")}
    for entry in joint_errs:
        leg_norm_entry = {}
        for leg in ("FL", "FR", "RL", "RR"):
            leg_norm_entry[leg] = _vector_norm(entry[leg])
            for axis in range(3):
                per_leg_axis_abs[leg][axis].append(abs(entry[leg][axis]))
        per_leg_err_norms.append(leg_norm_entry)

    raw_action_norms = []
    for entry in joint_generic["raw_action"]:
        raw_action_norms.append(
            {
                leg: _vector_norm(entry[leg])
                for leg in ("FL", "FR", "RL", "RR")
            }
        )

    rel_cmd_norms = []
    for entry in joint_generic["rel_cmd"]:
        rel_cmd_norms.append(
            {
                leg: _vector_norm(entry[leg])
                for leg in ("FL", "FR", "RL", "RR")
            }
        )

    rel_pos_norms = []
    for entry in joint_generic["rel_pos"]:
        rel_pos_norms.append(
            {
                leg: _vector_norm(entry[leg])
                for leg in ("FL", "FR", "RL", "RR")
            }
        )

    summary = {
        "log_file": str(log_path),
        "counts": {
            "velocity_cmd": len(velocity_cmds),
            "obs_samples": len(obs_samples),
            "joint_err": len(joint_errs),
            "joint_raw_action": len(joint_generic["raw_action"]),
            "joint_rel_cmd": len(joint_generic["rel_cmd"]),
            "joint_rel_pos": len(joint_generic["rel_pos"]),
            "fsm_transitions": len(fsm_transitions),
        },
        "fsm": {
            "transitions": fsm_transitions,
            "entered_velocity": any(t["to"] == "Velocity" for t in fsm_transitions),
            "returned_to_passive": any(t["to"] == "Passive" for t in fsm_transitions),
        },
        "velocity": {
            "filtered_windows": {
                "vx": _signed_window(
                    [{"vx": value} for value in filtered_vx],
                    "vx",
                ),
                "vy": _signed_window(
                    [{"vy": value} for value in filtered_vy],
                    "vy",
                ),
                "wz": _signed_window(
                    [{"wz": value} for value in filtered_wz],
                    "wz",
                ),
            },
            "filtered_max_abs": {
                "vx": max((abs(value) for value in filtered_vx), default=0.0),
                "vy": max((abs(value) for value in filtered_vy), default=0.0),
                "wz": max((abs(value) for value in filtered_wz), default=0.0),
            },
            "lin_vel_max_abs": {
                "vx": max((abs(value) for value in lin_vx), default=0.0),
                "vy": max((abs(value) for value in lin_vy), default=0.0),
                "imu_wz": max((abs(value) for value in lin_wz), default=0.0),
            },
            "zero_command_drift": {
                "measured_when_all_cmd_axes_near_zero": {
                    "lin_vel_vx": _subset_stats(
                        drift_samples,
                        "lin_vx",
                        lambda entry: (
                            abs(entry["filtered_vx"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_vy"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_wz"]) < zero_cmd_threshold_wz
                        ),
                    ),
                    "lin_vel_vy": _subset_stats(
                        drift_samples,
                        "lin_vy",
                        lambda entry: (
                            abs(entry["filtered_vx"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_vy"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_wz"]) < zero_cmd_threshold_wz
                        ),
                    ),
                    "lin_vel_xy_norm": _subset_stats(
                        drift_samples,
                        "lin_vel_xy_norm",
                        lambda entry: (
                            abs(entry["filtered_vx"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_vy"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_wz"]) < zero_cmd_threshold_wz
                        ),
                    ),
                    "imu_wz": _subset_stats(
                        drift_samples,
                        "imu_wz",
                        lambda entry: (
                            abs(entry["filtered_vx"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_vy"]) < zero_cmd_threshold_xy
                            and abs(entry["filtered_wz"]) < zero_cmd_threshold_wz
                        ),
                    ),
                },
                "measured_when_vx_cmd_near_zero": {
                    "lin_vel_vx": _subset_stats(
                        drift_samples,
                        "lin_vx",
                        lambda entry: abs(entry["filtered_vx"]) < zero_cmd_threshold_xy,
                    ),
                },
                "measured_when_vy_cmd_near_zero": {
                    "lin_vel_vy": _subset_stats(
                        drift_samples,
                        "lin_vy",
                        lambda entry: abs(entry["filtered_vy"]) < zero_cmd_threshold_xy,
                    ),
                },
                "measured_when_wz_cmd_near_zero": {
                    "imu_wz": _subset_stats(
                        drift_samples,
                        "imu_wz",
                        lambda entry: abs(entry["filtered_wz"]) < zero_cmd_threshold_wz,
                    ),
                },
                "filtered_cmd_reference": {
                    "filtered_vx_near_zero": asdict(
                        _window_stats([value for value in filtered_vx if abs(value) < zero_cmd_threshold_xy])
                    ),
                    "filtered_vy_near_zero": asdict(
                        _window_stats([value for value in filtered_vy if abs(value) < zero_cmd_threshold_xy])
                    ),
                    "filtered_wz_near_zero": asdict(
                        _window_stats([value for value in filtered_wz if abs(value) < zero_cmd_threshold_wz])
                    ),
                },
            },
        },
        "obs": {
            "base_ang_norm": asdict(_window_stats([_vector_norm(entry["base_ang"]) for entry in obs_samples])),
            "gravity_xy_tilt": asdict(
                _window_stats(
                    [math.sqrt(entry["gravity"][0] ** 2 + entry["gravity"][1] ** 2) for entry in obs_samples]
                )
            ),
            "cmd_windows": {
                "vx": _signed_window([{"vx": entry["cmd"][0]} for entry in obs_samples], "vx"),
                "vy": _signed_window([{"vy": entry["cmd"][1]} for entry in obs_samples], "vy"),
                "wz": _signed_window([{"wz": entry["cmd"][2]} for entry in obs_samples], "wz"),
            },
        },
        "joint_tracking": {
            "joint_err_leg_norm_mean": {
                leg: _mean([entry[leg] for entry in per_leg_err_norms])
                for leg in ("FL", "FR", "RL", "RR")
            },
            "joint_err_leg_norm_max": {
                leg: max((entry[leg] for entry in per_leg_err_norms), default=0.0)
                for leg in ("FL", "FR", "RL", "RR")
            },
            "joint_err_leg_axis_mean_abs": {
                leg: [
                    _mean(per_leg_axis_abs[leg][axis])
                    for axis in range(3)
                ]
                for leg in ("FL", "FR", "RL", "RR")
            },
            "side_abs_err_mean": {
                "left": [
                    _mean([entry["side_L"][axis] for entry in joint_errs])
                    for axis in range(3)
                ],
                "right": [
                    _mean([entry["side_R"][axis] for entry in joint_errs])
                    for axis in range(3)
                ],
            },
            "side_abs_err_max": {
                "left": [
                    max((entry["side_L"][axis] for entry in joint_errs), default=0.0)
                    for axis in range(3)
                ],
                "right": [
                    max((entry["side_R"][axis] for entry in joint_errs), default=0.0)
                    for axis in range(3)
                ],
            },
            "mean_left_minus_right": {
                axis: _mean([entry[axis] for entry in joint_side_bias])
                for axis in ("x", "y", "z")
            },
            "raw_action_leg_norm_mean": {
                leg: _mean([entry[leg] for entry in raw_action_norms])
                for leg in ("FL", "FR", "RL", "RR")
            },
            "rel_cmd_leg_norm_mean": {
                leg: _mean([entry[leg] for entry in rel_cmd_norms])
                for leg in ("FL", "FR", "RL", "RR")
            },
            "rel_pos_leg_norm_mean": {
                leg: _mean([entry[leg] for entry in rel_pos_norms])
                for leg in ("FL", "FR", "RL", "RR")
            },
        },
        "operator_flags": {
            "use_video_review": True,
            "use_surface_notes": True,
            "use_intervention_notes": True,
        },
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
