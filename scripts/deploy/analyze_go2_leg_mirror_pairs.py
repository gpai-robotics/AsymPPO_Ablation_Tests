#!/usr/bin/env python3
"""Focused mirrored-leg analysis for Go2 realtime monitor JSONLs.

This is meant to test hypotheses like:
- front-right leg is under-lifting / dragging
- rear-left hip is compensating harder than its mirror
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from go2_monitor_schema import normalize_payload_joint_order


JOINT_INDEX = {
    "FL_hip": 0,
    "FR_hip": 1,
    "RL_hip": 2,
    "RR_hip": 3,
    "FL_thigh": 4,
    "FR_thigh": 5,
    "RL_thigh": 6,
    "RR_thigh": 7,
    "FL_calf": 8,
    "FR_calf": 9,
    "RL_calf": 10,
    "RR_calf": 11,
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


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _load(path: Path) -> dict[str, np.ndarray]:
    wall_time = []
    remote = []
    q = []
    q_des = []
    q_err = []
    joint_vel = []
    tau_est = []
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
            q.append(latest["q"])
            q_des.append(latest["q_des"])
            q_err.append(latest["q_err"])
            joint_vel.append(latest["joint_vel"])
            tau_est.append(latest["tau_est"])

    return {
        "wall_time": np.asarray(wall_time, dtype=np.float64),
        "remote": np.asarray(remote, dtype=np.float32),
        "q": np.asarray(q, dtype=np.float32),
        "q_des": np.asarray(q_des, dtype=np.float32),
        "q_err": np.asarray(q_err, dtype=np.float32),
        "joint_vel": np.asarray(joint_vel, dtype=np.float32),
        "tau_est": np.asarray(tau_est, dtype=np.float32),
        "legacy_remap_applied": np.asarray([legacy_remap_applied], dtype=np.bool_),
    }


def _mask(data: dict[str, np.ndarray], near_zero_cmd_threshold: float, engaged_qdes_threshold: float) -> dict[str, np.ndarray]:
    remote = data["remote"]
    q_des_norm = np.linalg.norm(data["q_des"], axis=1)
    near_zero = np.all(np.abs(remote) <= near_zero_cmd_threshold, axis=1)
    engaged = q_des_norm >= engaged_qdes_threshold
    active = engaged & (~near_zero)
    return {"engaged": engaged, "active": active, "neutral": engaged & near_zero}


def _joint_stats(mask: np.ndarray, data: dict[str, np.ndarray], joint_name: str) -> dict[str, float | None]:
    idx = JOINT_INDEX[joint_name]
    if not np.any(mask):
        return {
            "q_err_mean_abs": None,
            "q_err_peak_abs": None,
            "tau_mean_abs": None,
            "tau_peak_abs": None,
            "dq_mean_abs": None,
            "q_range": None,
            "q_des_range": None,
        }
    q = data["q"][mask, idx]
    q_des = data["q_des"][mask, idx]
    q_err = np.abs(data["q_err"][mask, idx])
    tau = np.abs(data["tau_est"][mask, idx])
    dq = np.abs(data["joint_vel"][mask, idx])
    return {
        "q_err_mean_abs": float(np.mean(q_err)),
        "q_err_peak_abs": float(np.max(q_err)),
        "tau_mean_abs": float(np.mean(tau)),
        "tau_peak_abs": float(np.max(tau)),
        "dq_mean_abs": float(np.mean(dq)),
        "q_range": float(np.max(q) - np.min(q)),
        "q_des_range": float(np.max(q_des) - np.min(q_des)),
    }


def _leg_stats(mask: np.ndarray, data: dict[str, np.ndarray], leg_name: str) -> dict[str, float | None]:
    idx = LEG_INDEX[leg_name]
    if not np.any(mask):
        return {
            "q_err_mean_abs": None,
            "tau_mean_abs": None,
            "dq_mean_abs": None,
            "q_range_norm_mean": None,
            "q_des_range_norm_mean": None,
        }
    q = data["q"][mask][:, idx]
    q_des = data["q_des"][mask][:, idx]
    q_err = np.abs(data["q_err"][mask][:, idx])
    tau = np.abs(data["tau_est"][mask][:, idx])
    dq = np.abs(data["joint_vel"][mask][:, idx])
    q_range = np.max(q, axis=0) - np.min(q, axis=0)
    q_des_range = np.max(q_des, axis=0) - np.min(q_des, axis=0)
    return {
        "q_err_mean_abs": float(np.mean(q_err)),
        "tau_mean_abs": float(np.mean(tau)),
        "dq_mean_abs": float(np.mean(dq)),
        "q_range_norm_mean": float(np.mean(q_range)),
        "q_des_range_norm_mean": float(np.mean(q_des_range)),
    }


def _print_pair(title: str, left_name: str, right_name: str, left: dict[str, float | None], right: dict[str, float | None]) -> None:
    print(f"  {title}:")
    for key in left.keys():
        lv = left[key]
        rv = right[key]
        delta = None if lv is None or rv is None else lv - rv
        print(
            f"    {key}:"
            f" {left_name}={_format_float(lv)}"
            f" {right_name}={_format_float(rv)}"
            f" delta={_format_float(delta)}"
        )


def main() -> int:
    args = parse_args()
    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    data = _load(path)
    if data["wall_time"].size == 0:
        raise SystemExit(f"No samples found in {path}")

    masks = _mask(data, args.near_zero_cmd_threshold, args.engaged_qdes_threshold)

    print(f"jsonl: {path}")
    if bool(data["legacy_remap_applied"][0]):
        print("joint_order: legacy SDK-order capture remapped to policy order")
    else:
        print("joint_order: policy")
    print(f"samples: {data['wall_time'].size}")
    print(
        f"engaged_samples: {int(np.sum(masks['engaged']))} "
        f"active_samples: {int(np.sum(masks['active']))} "
        f"neutral_samples: {int(np.sum(masks['neutral']))}"
    )

    print()
    print("Leg mirror comparison (engaged_active_cmd):")
    _print_pair(
        "front pair",
        "FL",
        "FR",
        _leg_stats(masks["active"], data, "FL"),
        _leg_stats(masks["active"], data, "FR"),
    )
    _print_pair(
        "rear pair",
        "RL",
        "RR",
        _leg_stats(masks["active"], data, "RL"),
        _leg_stats(masks["active"], data, "RR"),
    )

    print()
    print("Joint focus (engaged_active_cmd):")
    _print_pair(
        "FR_thigh vs FL_thigh",
        "FL_thigh",
        "FR_thigh",
        _joint_stats(masks["active"], data, "FL_thigh"),
        _joint_stats(masks["active"], data, "FR_thigh"),
    )
    _print_pair(
        "FR_calf vs FL_calf",
        "FL_calf",
        "FR_calf",
        _joint_stats(masks["active"], data, "FL_calf"),
        _joint_stats(masks["active"], data, "FR_calf"),
    )
    _print_pair(
        "RL_hip vs RR_hip",
        "RR_hip",
        "RL_hip",
        _joint_stats(masks["active"], data, "RR_hip"),
        _joint_stats(masks["active"], data, "RL_hip"),
    )

    print()
    print("Joint focus (engaged_neutral_cmd):")
    _print_pair(
        "FR_thigh vs FL_thigh",
        "FL_thigh",
        "FR_thigh",
        _joint_stats(masks["neutral"], data, "FL_thigh"),
        _joint_stats(masks["neutral"], data, "FR_thigh"),
    )
    _print_pair(
        "FR_calf vs FL_calf",
        "FL_calf",
        "FR_calf",
        _joint_stats(masks["neutral"], data, "FL_calf"),
        _joint_stats(masks["neutral"], data, "FR_calf"),
    )
    _print_pair(
        "RL_hip vs RR_hip",
        "RR_hip",
        "RL_hip",
        _joint_stats(masks["neutral"], data, "RR_hip"),
        _joint_stats(masks["neutral"], data, "RL_hip"),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
