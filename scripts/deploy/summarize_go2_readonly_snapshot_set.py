#!/usr/bin/env python3
"""Summarize one or more read-only Go2 snapshot JSON files.

This is meant for repeated sitting or standing captures from a single robot.
It helps answer:

- is the robot statically stable across repeated snapshots?
- are foot-force and IMU posture signatures consistent?
- are motor temperatures consistently elevated?
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Snapshot JSON files produced by probe_go2_readonly.py.",
    )
    parser.add_argument(
        "--label",
        default="snapshot_set",
        help="Friendly name for this set in the report.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_low(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot.get("lowstate", {}).get("snapshot") or {}


def get_sport(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot.get("sportmodestate", {}).get("snapshot") or {}


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def fmt_triplet(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{v:+.{precision}f}" for v in values) + "]"


def vec_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def main() -> int:
    args = parse_args()
    snapshots = [load_json(path) for path in args.inputs]
    lows = [get_low(s) for s in snapshots if get_low(s)]
    sports = [get_sport(s) for s in snapshots if get_sport(s)]

    if not lows:
        raise SystemExit("No valid lowstate snapshots found in inputs.")

    joint_q = [[float(x) for x in low.get("joint_q_12", [])] for low in lows]
    joint_dq = [[float(x) for x in low.get("joint_dq_12", [])] for low in lows]
    gyro = [[float(x) for x in low.get("imu_gyro_xyz", [])] for low in lows]
    accel = [[float(x) for x in low.get("imu_accel_xyz", [])] for low in lows]
    foot = [[float(x) for x in low.get("foot_force", [])] for low in lows]
    temp = [[float(x) for x in low.get("temperature_hint", [])] for low in lows]

    joint_q_mean = [mean([row[i] for row in joint_q]) for i in range(len(joint_q[0]))]
    joint_q_std = [stdev([row[i] for row in joint_q]) for i in range(len(joint_q[0]))]
    joint_dq_mean = [mean([row[i] for row in joint_dq]) for i in range(len(joint_dq[0]))]
    joint_dq_std = [stdev([row[i] for row in joint_dq]) for i in range(len(joint_dq[0]))]
    gyro_mean = [mean([row[i] for row in gyro]) for i in range(len(gyro[0]))]
    gyro_std = [stdev([row[i] for row in gyro]) for i in range(len(gyro[0]))]
    accel_mean = [mean([row[i] for row in accel]) for i in range(len(accel[0]))]
    accel_std = [stdev([row[i] for row in accel]) for i in range(len(accel[0]))]
    foot_mean = [mean([row[i] for row in foot]) for i in range(len(foot[0]))]
    foot_std = [stdev([row[i] for row in foot]) for i in range(len(foot[0]))]
    temp_mean = [mean([row[i] for row in temp]) for i in range(len(temp[0]))]
    temp_std = [stdev([row[i] for row in temp]) for i in range(len(temp[0]))]

    foot_total = [sum(row) for row in foot]
    temp_global_mean = [mean(row) for row in temp]
    accel_xy_tilt = [math.sqrt(row[0] * row[0] + row[1] * row[1]) for row in accel]
    joint_q_l2_to_mean = [
        vec_norm([row[i] - joint_q_mean[i] for i in range(len(joint_q_mean))])
        for row in joint_q
    ]

    print(f"Label: {args.label}")
    print(f"Files: {len(args.inputs)}")
    print(f"Lowstate snapshots: {len(lows)}")
    print(f"Sport snapshots: {len(sports)}")

    print("\n[Joint Position Stability]")
    print(f"mean q_12={fmt_triplet(joint_q_mean[:3])} ...")
    print(
        f"snapshot-to-mean L2: mean={mean(joint_q_l2_to_mean):.4f} "
        f"max={max(joint_q_l2_to_mean):.4f}"
    )
    print(
        f"per-joint std: mean={mean(joint_q_std):.4f} "
        f"max={max(joint_q_std):.4f}"
    )

    print("\n[Joint Velocity Stability]")
    print(f"mean dq_12={fmt_triplet(joint_dq_mean[:3])} ...")
    print(
        f"per-joint std: mean={mean(joint_dq_std):.4f} "
        f"max={max(joint_dq_std):.4f}"
    )

    print("\n[IMU]")
    print(f"gyro mean={fmt_triplet(gyro_mean)} std={fmt_triplet(gyro_std, 4)}")
    print(f"accel mean={fmt_triplet(accel_mean)} std={fmt_triplet(accel_std, 4)}")
    print(
        f"accel xy tilt: mean={mean(accel_xy_tilt):.4f} "
        f"std={stdev(accel_xy_tilt):.4f} max={max(accel_xy_tilt):.4f}"
    )

    print("\n[Foot Force]")
    print(f"foot mean={fmt_triplet(foot_mean, 1)} std={fmt_triplet(foot_std, 2)}")
    print(
        f"total load: mean={mean(foot_total):.1f} "
        f"std={stdev(foot_total):.2f} max={max(foot_total):.1f}"
    )

    print("\n[Motor Temperature]")
    print(f"joint mean={fmt_triplet(temp_mean, 1)} std={fmt_triplet(temp_std, 2)}")
    print(
        f"global mean temp: mean={mean(temp_global_mean):.2f} "
        f"std={stdev(temp_global_mean):.2f} max={max(temp_global_mean):.2f}"
    )

    if sports:
        body_height = [float(s.get("body_height", 0.0)) for s in sports]
        yaw_speed = [float(s.get("yaw_speed", 0.0)) for s in sports]
        print("\n[Sport Mode]")
        print(
            f"body_height mean={mean(body_height):.4f} "
            f"std={stdev(body_height):.4f}"
        )
        print(
            f"yaw_speed mean={mean(yaw_speed):+.4f} "
            f"std={stdev(yaw_speed):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
