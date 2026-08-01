#!/usr/bin/env python3
"""Summarize a saved Go2 realtime monitor JSONL artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from go2_monitor_schema import POLICY_JOINT_NAMES, normalize_payload_joint_order

JOINT_NAMES = POLICY_JOINT_NAMES
SIDE_INDEX = {
    "left": np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int32),
    "right": np.asarray([1, 3, 5, 7, 9, 11], dtype=np.int32),
}
LEG_INDEX = {
    "FL": np.asarray([0, 4, 8], dtype=np.int32),
    "FR": np.asarray([1, 5, 9], dtype=np.int32),
    "RL": np.asarray([2, 6, 10], dtype=np.int32),
    "RR": np.asarray([3, 7, 11], dtype=np.int32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to *_monitor.jsonl")
    parser.add_argument(
        "--near-zero-cmd-threshold",
        type=float,
        default=0.05,
        help="Threshold for treating remote command axes as near zero.",
    )
    parser.add_argument(
        "--engaged-qdes-threshold",
        type=float,
        default=0.25,
        help="Threshold on ||q_des|| for considering the controller engaged.",
    )
    return parser.parse_args()


def _safe_stats(arr: np.ndarray) -> dict[str, float | int | None]:
    if arr.size == 0:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _load(path: Path) -> dict[str, np.ndarray]:
    wall_time = []
    remote = []
    dds_low = []
    dds_sport = []
    dds_lowcmd = []
    q = []
    q_des = []
    q_err = []
    joint_vel = []
    tau_est = []
    temperature = []
    foot_force = []
    imu_gyro = []
    sport_vel = []
    sport_yaw = []
    legacy_remap_applied = False

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload, remapped = normalize_payload_joint_order(json.loads(line))
            legacy_remap_applied = legacy_remap_applied or remapped
            latest = payload["latest"]
            wall_time.append(float(payload["wall_time"]))
            remote.append(
                [
                    float(payload["remote_cmd"]["vx"]),
                    float(payload["remote_cmd"]["vy"]),
                    float(payload["remote_cmd"]["wz"]),
                ]
            )
            dds_low.append(float(payload["dds_hz"]["low"]))
            dds_sport.append(float(payload["dds_hz"]["sport"]))
            dds_lowcmd.append(float(payload["dds_hz"]["lowcmd"]))
            q.append(latest["q"])
            q_des.append(latest["q_des"])
            q_err.append(latest["q_err"])
            joint_vel.append(latest["joint_vel"])
            tau_est.append(latest["tau_est"])
            temperature.append(latest["temperature"])
            foot_force.append(latest["foot_force"])
            imu_gyro.append(latest["imu_gyro"])
            sport_vel.append(latest["sport_vel"])
            sport_yaw.append([float(latest["sport_yaw"])])

    return {
        "wall_time": np.asarray(wall_time, dtype=np.float64),
        "remote": np.asarray(remote, dtype=np.float32),
        "dds_low": np.asarray(dds_low, dtype=np.float32),
        "dds_sport": np.asarray(dds_sport, dtype=np.float32),
        "dds_lowcmd": np.asarray(dds_lowcmd, dtype=np.float32),
        "q": np.asarray(q, dtype=np.float32),
        "q_des": np.asarray(q_des, dtype=np.float32),
        "q_err": np.asarray(q_err, dtype=np.float32),
        "joint_vel": np.asarray(joint_vel, dtype=np.float32),
        "tau_est": np.asarray(tau_est, dtype=np.float32),
        "temperature": np.asarray(temperature, dtype=np.float32),
        "foot_force": np.asarray(foot_force, dtype=np.float32),
        "imu_gyro": np.asarray(imu_gyro, dtype=np.float32),
        "sport_vel": np.asarray(sport_vel, dtype=np.float32),
        "sport_yaw": np.asarray(sport_yaw, dtype=np.float32),
        "legacy_remap_applied": np.asarray([legacy_remap_applied], dtype=np.bool_),
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _phase_stats(
    mask: np.ndarray,
    q_err_abs: np.ndarray,
    tau_abs: np.ndarray,
    temp: np.ndarray,
    gyro_norm: np.ndarray,
    sport_vel_norm: np.ndarray,
    q_err_joint_peak: np.ndarray,
) -> dict[str, object]:
    if not np.any(mask):
        return {
            "count": 0,
            "q_err_mean": None,
            "tau_mean": None,
            "temp_mean": None,
            "gyro_mean": None,
            "sport_vel_mean": None,
            "top_joint": None,
            "top_joint_q_err_mean": None,
            "top_joint_q_err_peak": None,
        }

    q_phase = q_err_abs[mask]
    tau_phase = tau_abs[mask]
    temp_phase = temp[mask]
    gyro_phase = gyro_norm[mask]
    sport_phase = sport_vel_norm[mask]
    q_joint_mean = np.mean(q_phase, axis=0)
    top_joint = int(np.argmax(q_joint_mean))
    return {
        "count": int(np.sum(mask)),
        "q_err_mean": float(np.mean(q_phase)),
        "tau_mean": float(np.mean(tau_phase)),
        "temp_mean": float(np.mean(temp_phase)),
        "gyro_mean": float(np.mean(gyro_phase)),
        "sport_vel_mean": float(np.mean(sport_phase)),
        "top_joint": JOINT_NAMES[top_joint],
        "top_joint_q_err_mean": float(q_joint_mean[top_joint]),
        "top_joint_q_err_peak": float(np.max(q_phase[:, top_joint])),
    }


def main() -> int:
    args = parse_args()
    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    data = _load(path)
    n = int(data["wall_time"].size)
    if n == 0:
        raise SystemExit(f"No samples found in {path}")

    duration_s = float(data["wall_time"][-1] - data["wall_time"][0]) if n > 1 else 0.0
    stream_hz = float((n - 1) / duration_s) if duration_s > 0.0 else None

    q_err_abs = np.abs(data["q_err"])
    joint_vel_abs = np.abs(data["joint_vel"])
    tau_abs = np.abs(data["tau_est"])
    temp = data["temperature"]
    gyro_norm = np.linalg.norm(data["imu_gyro"], axis=1)
    sport_vel_norm = np.linalg.norm(data["sport_vel"], axis=1)
    foot_force = data["foot_force"]
    q_des_norm = np.linalg.norm(data["q_des"], axis=1)

    near_zero_mask = np.all(np.abs(data["remote"]) <= args.near_zero_cmd_threshold, axis=1)
    engaged_mask = q_des_norm >= args.engaged_qdes_threshold
    pre_engagement_mask = ~engaged_mask
    active_cmd_mask = ~near_zero_mask
    engaged_neutral_mask = engaged_mask & near_zero_mask
    engaged_active_mask = engaged_mask & active_cmd_mask

    side_mean_abs_err = {
        side: float(np.mean(q_err_abs[:, idx])) for side, idx in SIDE_INDEX.items()
    }
    side_mean_abs_tau = {
        side: float(np.mean(tau_abs[:, idx])) for side, idx in SIDE_INDEX.items()
    }
    side_mean_temp = {
        side: float(np.mean(temp[:, idx])) for side, idx in SIDE_INDEX.items()
    }

    hottest_joint_idx = int(np.argmax(np.max(temp, axis=0)))
    hottest_joint_peak = float(np.max(temp[:, hottest_joint_idx]))
    hottest_joint_mean = float(np.mean(temp[:, hottest_joint_idx]))

    q_err_joint_mean = np.mean(q_err_abs, axis=0)
    q_err_joint_peak = np.max(q_err_abs, axis=0)
    tau_joint_mean = np.mean(tau_abs, axis=0)
    temp_joint_mean = np.mean(temp, axis=0)
    temp_joint_peak = np.max(temp, axis=0)

    print(f"jsonl: {path}")
    if bool(data["legacy_remap_applied"][0]):
        print("joint_order: legacy SDK-order capture remapped to policy order")
    else:
        print("joint_order: policy")
    print(f"samples: {n}")
    print(f"duration_s: {duration_s:.3f}")
    print(f"stream_hz_estimate: {_format_float(stream_hz, 2)}")
    print(
        "dds_hz_mean:"
        f" low={_format_float(float(np.mean(data['dds_low'])), 1)}"
        f" sport={_format_float(float(np.mean(data['dds_sport'])), 1)}"
        f" lowcmd={_format_float(float(np.mean(data['dds_lowcmd'])), 1)}"
    )

    print()
    print("Phases:")
    print(
        f"  pre_engagement_samples={int(np.sum(pre_engagement_mask))}"
        f" engaged_samples={int(np.sum(engaged_mask))}"
        f" engaged_qdes_threshold={args.engaged_qdes_threshold:.3f}"
    )

    print()
    print("Zero-command windows:")
    print(
        f"  samples={int(np.sum(near_zero_mask))} / {n}"
        f" threshold={args.near_zero_cmd_threshold:.3f}"
    )
    if np.any(near_zero_mask):
        zero_vxy = np.linalg.norm(data["sport_vel"][near_zero_mask, :2], axis=1)
        zero_yaw = np.abs(data["sport_yaw"][near_zero_mask, 0])
        print(
            "  drift_vxy:"
            f" mean={_format_float(float(np.mean(zero_vxy)))}"
            f" p95={_format_float(float(np.percentile(zero_vxy, 95)))}"
            f" max={_format_float(float(np.max(zero_vxy)))}"
        )
        print(
            "  drift_yaw:"
            f" mean={_format_float(float(np.mean(zero_yaw)))}"
            f" p95={_format_float(float(np.percentile(zero_yaw, 95)))}"
            f" max={_format_float(float(np.max(zero_yaw)))}"
        )
    else:
        print("  no near-zero remote command samples")

    print()
    print("Whole-run motion:")
    print(
        "  gyro_norm:"
        f" mean={_format_float(float(np.mean(gyro_norm)))}"
        f" p95={_format_float(float(np.percentile(gyro_norm, 95)))}"
        f" max={_format_float(float(np.max(gyro_norm)))}"
    )
    print(
        "  sport_vel_norm:"
        f" mean={_format_float(float(np.mean(sport_vel_norm)))}"
        f" p95={_format_float(float(np.percentile(sport_vel_norm, 95)))}"
        f" max={_format_float(float(np.max(sport_vel_norm)))}"
    )

    print()
    print("Phase drilldown:")
    for name, mask in (
        ("pre_engagement", pre_engagement_mask),
        ("engaged_all", engaged_mask),
        ("engaged_neutral_cmd", engaged_neutral_mask),
        ("engaged_active_cmd", engaged_active_mask),
    ):
        stats = _phase_stats(
            mask=mask,
            q_err_abs=q_err_abs,
            tau_abs=tau_abs,
            temp=temp,
            gyro_norm=gyro_norm,
            sport_vel_norm=sport_vel_norm,
            q_err_joint_peak=q_err_joint_peak,
        )
        print(
            f"  {name}:"
            f" samples={stats['count']}"
            f" q_err_mean={_format_float(stats['q_err_mean'])}"
            f" tau_mean={_format_float(stats['tau_mean'])}"
            f" temp_mean={_format_float(stats['temp_mean'], 1)}"
            f" gyro_mean={_format_float(stats['gyro_mean'])}"
            f" sport_vel_mean={_format_float(stats['sport_vel_mean'])}"
        )
        if stats["top_joint"] is not None:
            print(
                f"    top_q_err_joint={stats['top_joint']}"
                f" mean_abs={_format_float(stats['top_joint_q_err_mean'])}"
                f" peak_abs={_format_float(stats['top_joint_q_err_peak'])}"
            )

    print()
    print("Side balance:")
    print(
        "  q_err_mean_abs:"
        f" left={_format_float(side_mean_abs_err['left'])}"
        f" right={_format_float(side_mean_abs_err['right'])}"
        f" delta={_format_float(side_mean_abs_err['left'] - side_mean_abs_err['right'])}"
    )
    print(
        "  tau_mean_abs:"
        f" left={_format_float(side_mean_abs_tau['left'])}"
        f" right={_format_float(side_mean_abs_tau['right'])}"
        f" delta={_format_float(side_mean_abs_tau['left'] - side_mean_abs_tau['right'])}"
    )
    print(
        "  temp_mean:"
        f" left={_format_float(side_mean_temp['left'], 1)}"
        f" right={_format_float(side_mean_temp['right'], 1)}"
        f" delta={_format_float(side_mean_temp['left'] - side_mean_temp['right'], 1)}"
    )

    print()
    print("Per-leg q_err mean_abs:")
    for leg, idx in LEG_INDEX.items():
        print(f"  {leg}: {_format_float(float(np.mean(q_err_abs[:, idx])))}")

    print()
    print("Hottest joint:")
    print(
        f"  {JOINT_NAMES[hottest_joint_idx]}:"
        f" mean_temp={_format_float(hottest_joint_mean, 1)}"
        f" peak_temp={_format_float(hottest_joint_peak, 1)}"
    )

    print()
    print("Top joints by mean_abs q_err:")
    top_q = np.argsort(q_err_joint_mean)[::-1][:5]
    for rank, idx in enumerate(top_q, start=1):
        print(
            f"  {rank}. {JOINT_NAMES[idx]}:"
            f" mean_abs={_format_float(float(q_err_joint_mean[idx]))}"
            f" peak_abs={_format_float(float(q_err_joint_peak[idx]))}"
        )

    print()
    print("Top joints by mean_abs tau_est:")
    top_tau = np.argsort(tau_joint_mean)[::-1][:5]
    for rank, idx in enumerate(top_tau, start=1):
        print(
            f"  {rank}. {JOINT_NAMES[idx]}:"
            f" mean_abs={_format_float(float(tau_joint_mean[idx]))}"
            f" peak_abs={_format_float(float(np.max(tau_abs[:, idx])))}"
        )

    print()
    print("Top joints by peak temperature:")
    top_temp = np.argsort(temp_joint_peak)[::-1][:5]
    for rank, idx in enumerate(top_temp, start=1):
        print(
            f"  {rank}. {JOINT_NAMES[idx]}:"
            f" mean={_format_float(float(temp_joint_mean[idx]), 1)}"
            f" peak={_format_float(float(temp_joint_peak[idx]), 1)}"
        )

    print()
    print("Foot-force summary:")
    for foot_idx in range(foot_force.shape[1]):
        stats = _safe_stats(foot_force[:, foot_idx])
        print(
            f"  foot_{foot_idx}:"
            f" mean={_format_float(stats['mean'], 1)}"
            f" p95={_format_float(stats['p95'], 1)}"
            f" max={_format_float(stats['max'], 1)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
