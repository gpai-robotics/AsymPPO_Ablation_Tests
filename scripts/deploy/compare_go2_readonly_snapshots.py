#!/usr/bin/env python3
"""Compare two read-only Go2 probe snapshots.

This is intended to be used with JSON outputs produced by:
  scripts/deploy/probe_go2_readonly.py --json-out <file>

It focuses on robot-side signals that can differ across units even when the
software stack is nominally the same.
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
    parser.add_argument("--a", required=True, help="JSON snapshot for robot A.")
    parser.add_argument("--b", required=True, help="JSON snapshot for robot B.")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_low(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot.get("lowstate", {}).get("snapshot") or {}


def get_sport(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot.get("sportmodestate", {}).get("snapshot") or {}


def vector_delta(a: list[float], b: list[float]) -> list[float]:
    n = min(len(a), len(b))
    return [float(a[i]) - float(b[i]) for i in range(n)]


def norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def fmt_list(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{v:+.{precision}f}" for v in values) + "]"


def section(title: str) -> None:
    print(f"\n[{title}]")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


def append_flag(flags: list[str], metric: float, warn: float, critical: float, message: str) -> None:
    if metric >= critical:
        flags.append(f"high: {message}")
    elif metric >= warn:
        flags.append(f"warn: {message}")


def main() -> int:
    args = parse_args()
    a = load_json(args.a)
    b = load_json(args.b)

    low_a = get_low(a)
    low_b = get_low(b)
    sport_a = get_sport(a)
    sport_b = get_sport(b)

    print("Robot A:", args.a)
    print("Robot B:", args.b)

    section("Metadata")
    print(f"A iface={a.get('iface')} low_hz={a.get('lowstate', {}).get('hz_estimate')}")
    print(f"B iface={b.get('iface')} low_hz={b.get('lowstate', {}).get('hz_estimate')}")

    if low_a and low_b:
        q_a = [float(x) for x in low_a.get("joint_q_12", [])]
        q_b = [float(x) for x in low_b.get("joint_q_12", [])]
        dq_a = [float(x) for x in low_a.get("joint_dq_12", [])]
        dq_b = [float(x) for x in low_b.get("joint_dq_12", [])]
        gyro_a = [float(x) for x in low_a.get("imu_gyro_xyz", [])]
        gyro_b = [float(x) for x in low_b.get("imu_gyro_xyz", [])]
        accel_a = [float(x) for x in low_a.get("imu_accel_xyz", [])]
        accel_b = [float(x) for x in low_b.get("imu_accel_xyz", [])]
        foot_a = [int(x) for x in low_a.get("foot_force", [])]
        foot_b = [int(x) for x in low_b.get("foot_force", [])]
        temp_a = [int(x) for x in low_a.get("temperature_hint", [])]
        temp_b = [int(x) for x in low_b.get("temperature_hint", [])]

        dq = vector_delta(q_a, q_b)
        ddq = vector_delta(dq_a, dq_b)
        dgyro = vector_delta(gyro_a, gyro_b)
        daccel = vector_delta(accel_a, accel_b)
        dfoot = [fa - fb for fa, fb in zip(foot_a, foot_b)]
        dtemp = [ta - tb for ta, tb in zip(temp_a, temp_b)]
        abs_dfoot = [abs(x) for x in dfoot]
        abs_dtemp = [abs(x) for x in dtemp]

        section("Joint Position")
        print(f"A q_12={fmt_list(q_a)}")
        print(f"B q_12={fmt_list(q_b)}")
        print(f"A-B={fmt_list(dq)}")
        print(f"|A-B|_2={norm(dq):.3f}")

        section("Joint Velocity")
        print(f"A dq_12={fmt_list(dq_a)}")
        print(f"B dq_12={fmt_list(dq_b)}")
        print(f"A-B={fmt_list(ddq)}")
        print(f"|A-B|_2={norm(ddq):.3f}")

        section("IMU")
        print(f"A gyro={fmt_list(gyro_a)} accel={fmt_list(accel_a)}")
        print(f"B gyro={fmt_list(gyro_b)} accel={fmt_list(accel_b)}")
        print(f"gyro A-B={fmt_list(dgyro)} |A-B|_2={norm(dgyro):.3f}")
        print(f"accel A-B={fmt_list(daccel)} |A-B|_2={norm(daccel):.3f}")

        section("Foot Force")
        print(f"A foot_force={foot_a}")
        print(f"B foot_force={foot_b}")
        print(f"A-B={dfoot}")
        print(
            "totals "
            f"A={sum(foot_a)} B={sum(foot_b)} "
            f"delta={sum(foot_a) - sum(foot_b):+d} "
            f"max_abs_per_foot={max(abs_dfoot, default=0)}"
        )

        section("Motor Temperature")
        print(f"A temp_12={temp_a}")
        print(f"B temp_12={temp_b}")
        print(f"A-B={dtemp}")
        print(
            f"mean_temp A={mean(temp_a):.1f} B={mean(temp_b):.1f} "
            f"delta={mean(temp_a) - mean(temp_b):+.1f} "
            f"max_abs_joint_delta={max(abs_dtemp, default=0)}"
        )

        section("Remote")
        remote_a = low_a.get("remote", {})
        remote_b = low_b.get("remote", {})
        print(f"A buttons={remote_a.get('active_buttons', [])} sticks={[remote_a.get('lx'), remote_a.get('ly'), remote_a.get('rx'), remote_a.get('ry')]}")
        print(f"B buttons={remote_b.get('active_buttons', [])} sticks={[remote_b.get('lx'), remote_b.get('ly'), remote_b.get('rx'), remote_b.get('ry')]}")

    if sport_a or sport_b:
        section("Sport Mode")
        print(f"A sport={json.dumps(sport_a, indent=2)}")
        print(f"B sport={json.dumps(sport_b, indent=2)}")

    if low_a and low_b:
        q_delta_norm = norm(vector_delta(q_a, q_b))
        dq_delta_norm = norm(vector_delta(dq_a, dq_b))
        gyro_delta_norm = norm(vector_delta(gyro_a, gyro_b))
        accel_delta_norm = norm(vector_delta(accel_a, accel_b))
        foot_total_delta = abs(sum(foot_a) - sum(foot_b))
        foot_max_delta = max(abs_dfoot, default=0)
        temp_mean_delta = abs(mean(temp_a) - mean(temp_b))
        temp_max_delta = max(abs_dtemp, default=0)

        flags: list[str] = []
        append_flag(
            flags,
            q_delta_norm,
            warn=0.10,
            critical=0.18,
            message=f"joint pose differs more than expected (L2={q_delta_norm:.3f} rad)",
        )
        append_flag(
            flags,
            dq_delta_norm,
            warn=0.10,
            critical=0.20,
            message=f"rest joint velocity mismatch is elevated (L2={dq_delta_norm:.3f} rad/s)",
        )
        append_flag(
            flags,
            gyro_delta_norm,
            warn=0.03,
            critical=0.08,
            message=f"IMU gyro mismatch is elevated (L2={gyro_delta_norm:.3f})",
        )
        append_flag(
            flags,
            accel_delta_norm,
            warn=0.30,
            critical=0.60,
            message=f"IMU accel/posture mismatch is elevated (L2={accel_delta_norm:.3f})",
        )
        append_flag(
            flags,
            foot_total_delta,
            warn=40.0,
            critical=100.0,
            message=f"total foot-force differs strongly ({foot_total_delta:.0f})",
        )
        append_flag(
            flags,
            float(foot_max_delta),
            warn=20.0,
            critical=40.0,
            message=f"per-foot force delta is large (max={foot_max_delta})",
        )
        append_flag(
            flags,
            temp_mean_delta,
            warn=2.0,
            critical=4.0,
            message=f"mean motor temperature differs ({temp_mean_delta:.1f} C)",
        )
        append_flag(
            flags,
            float(temp_max_delta),
            warn=4.0,
            critical=7.0,
            message=f"single-joint motor temperature delta is large (max={temp_max_delta} C)",
        )

        section("Compatibility Flags")
        if flags:
            for item in flags:
                print(item)
        else:
            print("ok: no large robot-side static-state mismatches flagged")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
