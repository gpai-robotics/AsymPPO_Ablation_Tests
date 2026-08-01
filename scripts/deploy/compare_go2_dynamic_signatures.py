#!/usr/bin/env python3
"""Compare two Go2 dynamic read-only captures.

Use this after capturing both robots while the same controller/policy sequence
is running. It aligns LowState samples with the nearest LowCmd sample, computes
joint tracking and command statistics in deployed policy joint order, and
highlights which joints/legs differ most.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from go2_monitor_schema import POLICY_JOINT_NAMES, POLICY_TO_SDK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-series", required=True, help="Robot A *_series.jsonl capture.")
    parser.add_argument("--a-lowcmd", required=True, help="Robot A *_lowcmd_stream.jsonl capture.")
    parser.add_argument("--b-series", required=True, help="Robot B *_series.jsonl capture.")
    parser.add_argument("--b-lowcmd", required=True, help="Robot B *_lowcmd_stream.jsonl capture.")
    parser.add_argument(
        "--nearest-lowcmd-s",
        type=float,
        default=0.05,
        help="Maximum time delta for pairing a LowState sample with a LowCmd sample.",
    )
    parser.add_argument(
        "--engaged-qdes-threshold",
        type=float,
        default=0.5,
        help="Use samples with ||q_des|| above this threshold for engaged-policy stats.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sdk_to_policy(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (12,):
        return np.full(12, np.nan, dtype=np.float64)
    return array[POLICY_TO_SDK]


def vec_from_snapshot(snapshot: dict[str, Any], key: str) -> np.ndarray:
    return sdk_to_policy([float(x) for x in snapshot.get(key, [])])


def nearest_command(lowcmd_rows: list[dict[str, Any]], wall_time: float, cursor: int) -> tuple[dict[str, Any] | None, int, float]:
    if not lowcmd_rows:
        return None, cursor, math.inf
    cursor = max(0, min(cursor, len(lowcmd_rows) - 1))
    while cursor + 1 < len(lowcmd_rows):
        cur_dt = abs(float(lowcmd_rows[cursor].get("wall_time", 0.0)) - wall_time)
        next_dt = abs(float(lowcmd_rows[cursor + 1].get("wall_time", 0.0)) - wall_time)
        if next_dt > cur_dt:
            break
        cursor += 1
    dt = abs(float(lowcmd_rows[cursor].get("wall_time", 0.0)) - wall_time)
    return lowcmd_rows[cursor], cursor, dt


@dataclass
class DynamicData:
    label: str
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    temp: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    foot_force: np.ndarray
    q_des: np.ndarray
    dq_des: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    lowcmd_dt: np.ndarray
    lowstate_hz: float | None
    lowcmd_hz: float | None


def topic_hz(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    t0 = float(rows[0].get("wall_time", 0.0))
    t1 = float(rows[-1].get("wall_time", 0.0))
    if t1 <= t0:
        return None
    return float(len(rows) - 1) / (t1 - t0)


def load_dynamic(label: str, series_path: str, lowcmd_path: str, max_dt_s: float) -> DynamicData:
    series_rows = load_jsonl(series_path)
    lowcmd_rows = load_jsonl(lowcmd_path)

    t_rows: list[float] = []
    q_rows: list[np.ndarray] = []
    dq_rows: list[np.ndarray] = []
    tau_rows: list[np.ndarray] = []
    temp_rows: list[np.ndarray] = []
    gyro_rows: list[np.ndarray] = []
    accel_rows: list[np.ndarray] = []
    foot_rows: list[np.ndarray] = []
    qdes_rows: list[np.ndarray] = []
    dqdes_rows: list[np.ndarray] = []
    tauff_rows: list[np.ndarray] = []
    kp_rows: list[np.ndarray] = []
    kd_rows: list[np.ndarray] = []
    lowcmd_dt_rows: list[float] = []

    cursor = 0
    t0: float | None = None
    for row in series_rows:
        low = ((row.get("lowstate") or {}).get("snapshot") or {})
        if not low:
            continue
        wall_time = float(row.get("wall_time", 0.0))
        cmd_row, cursor, dt = nearest_command(lowcmd_rows, wall_time, cursor)
        if cmd_row is None or dt > max_dt_s:
            continue
        cmd = ((cmd_row.get("lowcmd") or {}).get("snapshot") or {})
        if not cmd:
            continue
        if t0 is None:
            t0 = wall_time
        t_rows.append(wall_time - t0)
        q_rows.append(vec_from_snapshot(low, "joint_q_12"))
        dq_rows.append(vec_from_snapshot(low, "joint_dq_12"))
        tau_rows.append(vec_from_snapshot(low, "joint_tau_est_12"))
        temp_rows.append(vec_from_snapshot(low, "temperature_hint"))
        gyro_rows.append(np.asarray(low.get("imu_gyro_xyz", [np.nan, np.nan, np.nan]), dtype=np.float64))
        accel_rows.append(np.asarray(low.get("imu_accel_xyz", [np.nan, np.nan, np.nan]), dtype=np.float64))
        foot_rows.append(np.asarray(low.get("foot_force", [np.nan] * 4), dtype=np.float64))
        qdes_rows.append(vec_from_snapshot(cmd, "joint_q_des_12"))
        dqdes_rows.append(vec_from_snapshot(cmd, "joint_dq_des_12"))
        tauff_rows.append(vec_from_snapshot(cmd, "joint_tau_ff_12"))
        kp_rows.append(vec_from_snapshot(cmd, "joint_kp_12"))
        kd_rows.append(vec_from_snapshot(cmd, "joint_kd_12"))
        lowcmd_dt_rows.append(dt)

    if not t_rows:
        raise SystemExit(
            f"No aligned samples for {label}. Check that series and lowcmd streams overlap in time."
        )

    return DynamicData(
        label=label,
        t=np.asarray(t_rows, dtype=np.float64),
        q=np.vstack(q_rows),
        dq=np.vstack(dq_rows),
        tau_est=np.vstack(tau_rows),
        temp=np.vstack(temp_rows),
        gyro=np.vstack(gyro_rows),
        accel=np.vstack(accel_rows),
        foot_force=np.vstack(foot_rows),
        q_des=np.vstack(qdes_rows),
        dq_des=np.vstack(dqdes_rows),
        tau_ff=np.vstack(tauff_rows),
        kp=np.vstack(kp_rows),
        kd=np.vstack(kd_rows),
        lowcmd_dt=np.asarray(lowcmd_dt_rows, dtype=np.float64),
        lowstate_hz=topic_hz(series_rows),
        lowcmd_hz=topic_hz(lowcmd_rows),
    )


def mean_abs(array: np.ndarray) -> np.ndarray:
    return np.nanmean(np.abs(array), axis=0)


def peak_abs(array: np.ndarray) -> np.ndarray:
    return np.nanmax(np.abs(array), axis=0)


def fmt(value: float | None, precision: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{precision}f}"


def top_joints(values: np.ndarray, count: int = 6) -> list[tuple[str, float]]:
    order = np.argsort(values)[::-1][:count]
    return [(POLICY_JOINT_NAMES[int(idx)], float(values[int(idx)])) for idx in order]


def print_top(title: str, values: np.ndarray, count: int = 6) -> None:
    print(title)
    for name, value in top_joints(values, count=count):
        print(f"  {name:<9} {value:.4f}")


def engaged_mask(data: DynamicData, threshold: float) -> np.ndarray:
    qdes_norm = np.linalg.norm(data.q_des, axis=1)
    mask = qdes_norm >= threshold
    if not np.any(mask):
        return np.ones_like(qdes_norm, dtype=bool)
    return mask


def describe_robot(data: DynamicData, threshold: float) -> dict[str, Any]:
    mask = engaged_mask(data, threshold)
    q_err = data.q_des - data.q
    q_err_engaged = q_err[mask]
    tau_engaged = data.tau_est[mask]
    dq_engaged = data.dq[mask]

    q_err_mean = mean_abs(q_err_engaged)
    q_err_peak = peak_abs(q_err_engaged)
    tau_mean = mean_abs(tau_engaged)
    dq_mean = mean_abs(dq_engaged)

    return {
        "samples": int(data.t.shape[0]),
        "duration_s": float(data.t[-1] - data.t[0]) if data.t.shape[0] > 1 else 0.0,
        "engaged_samples": int(np.sum(mask)),
        "lowstate_hz": data.lowstate_hz,
        "lowcmd_hz": data.lowcmd_hz,
        "lowcmd_pair_dt_mean_ms": float(np.nanmean(data.lowcmd_dt) * 1000.0),
        "q_err_mean_abs": q_err_mean,
        "q_err_peak_abs": q_err_peak,
        "tau_est_mean_abs": tau_mean,
        "dq_mean_abs": dq_mean,
        "q_des_mean_abs": mean_abs(data.q_des[mask]),
        "kp_min": float(np.nanmin(data.kp)),
        "kp_max": float(np.nanmax(data.kp)),
        "kd_min": float(np.nanmin(data.kd)),
        "kd_max": float(np.nanmax(data.kd)),
        "foot_force_mean": np.nanmean(data.foot_force[mask], axis=0),
        "gyro_abs_mean": mean_abs(data.gyro[mask]),
        "temp_max": np.nanmax(data.temp, axis=0),
    }


def print_robot_summary(data: DynamicData, summary: dict[str, Any]) -> None:
    print(f"\n[{data.label}]")
    print(
        f"samples={summary['samples']} engaged={summary['engaged_samples']} "
        f"duration={summary['duration_s']:.2f}s "
        f"lowstate_hz={fmt(summary['lowstate_hz'], 1)} lowcmd_hz={fmt(summary['lowcmd_hz'], 1)} "
        f"pair_dt_mean={summary['lowcmd_pair_dt_mean_ms']:.1f}ms"
    )
    print(
        f"kp=[{summary['kp_min']:.2f}, {summary['kp_max']:.2f}] "
        f"kd=[{summary['kd_min']:.2f}, {summary['kd_max']:.2f}] "
        f"foot_force_mean={np.round(summary['foot_force_mean'], 1).tolist()} "
        f"gyro_abs_mean={np.round(summary['gyro_abs_mean'], 3).tolist()}"
    )
    print_top("Top q_err mean_abs joints:", summary["q_err_mean_abs"])
    print_top("Top q_err peak_abs joints:", summary["q_err_peak_abs"])
    print_top("Top tau_est mean_abs joints:", summary["tau_est_mean_abs"])
    print_top("Top dq mean_abs joints:", summary["dq_mean_abs"])


def compare_summaries(a: dict[str, Any], b: dict[str, Any]) -> None:
    print("\n[A-B Dynamic Delta]")
    scalar_keys = [
        "lowstate_hz",
        "lowcmd_hz",
        "lowcmd_pair_dt_mean_ms",
        "kp_min",
        "kp_max",
        "kd_min",
        "kd_max",
    ]
    for key in scalar_keys:
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            continue
        print(f"{key:<24} A={fmt(av)} B={fmt(bv)} delta={fmt(float(av) - float(bv))}")

    for key, label in [
        ("q_err_mean_abs", "q_err mean_abs delta"),
        ("q_err_peak_abs", "q_err peak_abs delta"),
        ("tau_est_mean_abs", "tau_est mean_abs delta"),
        ("dq_mean_abs", "dq mean_abs delta"),
        ("q_des_mean_abs", "q_des mean_abs delta"),
        ("temp_max", "max temperature delta"),
    ]:
        delta = np.asarray(a[key], dtype=np.float64) - np.asarray(b[key], dtype=np.float64)
        print_top(label + ":", np.abs(delta), count=6)

    foot_delta = np.asarray(a["foot_force_mean"]) - np.asarray(b["foot_force_mean"])
    print(f"foot_force_mean A-B={np.round(foot_delta, 1).tolist()}")


def main() -> int:
    args = parse_args()
    a = load_dynamic("Robot A", args.a_series, args.a_lowcmd, args.nearest_lowcmd_s)
    b = load_dynamic("Robot B", args.b_series, args.b_lowcmd, args.nearest_lowcmd_s)

    summary_a = describe_robot(a, args.engaged_qdes_threshold)
    summary_b = describe_robot(b, args.engaged_qdes_threshold)

    print("Robot A series:", args.a_series)
    print("Robot A lowcmd:", args.a_lowcmd)
    print("Robot B series:", args.b_series)
    print("Robot B lowcmd:", args.b_lowcmd)
    print_robot_summary(a, summary_a)
    print_robot_summary(b, summary_b)
    compare_summaries(summary_a, summary_b)

    print("\n[Interpretation]")
    print("- If q_des/kp/kd match but q_err/tau/dq diverge, suspect hardware response/calibration/friction/load.")
    print("- If q_des or gains differ, suspect controller config, policy bundle, mode, or runtime path mismatch.")
    print("- If divergence appears before Velocity/policy takeover, investigate FSM stance or robot startup state first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
