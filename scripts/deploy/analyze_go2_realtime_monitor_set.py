#!/usr/bin/env python3
"""Compare hotspot patterns across multiple Go2 realtime monitor JSONLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from go2_monitor_schema import POLICY_JOINT_NAMES, normalize_payload_joint_order

JOINT_NAMES = POLICY_JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        action="append",
        default=[],
        help="Path to a monitor jsonl. Repeatable.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob for monitor jsonl files. Repeatable.",
    )
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


def _resolve_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.jsonl:
        path = Path(item)
        if path.exists():
            paths.append(path)
    for pattern in args.glob:
        paths.extend(sorted(Path().glob(pattern)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _load_monitor_jsonl(path: Path) -> dict[str, np.ndarray]:
    wall_time = []
    remote = []
    q_des = []
    q_err = []
    tau_est = []
    temperature = []
    imu_gyro = []
    sport_vel = []
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
            q_des.append(latest["q_des"])
            q_err.append(latest["q_err"])
            tau_est.append(latest["tau_est"])
            temperature.append(latest["temperature"])
            imu_gyro.append(latest["imu_gyro"])
            sport_vel.append(latest["sport_vel"])

    return {
        "wall_time": np.asarray(wall_time, dtype=np.float64),
        "remote": np.asarray(remote, dtype=np.float32),
        "q_des": np.asarray(q_des, dtype=np.float32),
        "q_err": np.asarray(q_err, dtype=np.float32),
        "tau_est": np.asarray(tau_est, dtype=np.float32),
        "temperature": np.asarray(temperature, dtype=np.float32),
        "imu_gyro": np.asarray(imu_gyro, dtype=np.float32),
        "sport_vel": np.asarray(sport_vel, dtype=np.float32),
        "legacy_remap_applied": np.asarray([legacy_remap_applied], dtype=np.bool_),
    }


def _summarize_run(path: Path, data: dict[str, np.ndarray], near_zero_cmd_threshold: float, engaged_qdes_threshold: float) -> dict[str, object]:
    n = int(data["wall_time"].size)
    if n == 0:
        raise ValueError(f"No samples in {path}")
    duration_s = float(data["wall_time"][-1] - data["wall_time"][0]) if n > 1 else 0.0
    q_err_abs = np.abs(data["q_err"])
    tau_abs = np.abs(data["tau_est"])
    temp = data["temperature"]
    gyro_norm = np.linalg.norm(data["imu_gyro"], axis=1)
    sport_vel_norm = np.linalg.norm(data["sport_vel"], axis=1)
    q_des_norm = np.linalg.norm(data["q_des"], axis=1)
    near_zero_mask = np.all(np.abs(data["remote"]) <= near_zero_cmd_threshold, axis=1)
    engaged_mask = q_des_norm >= engaged_qdes_threshold
    engaged_active_mask = engaged_mask & (~near_zero_mask)

    def phase_joint_mean(mask: np.ndarray, arr: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.full((arr.shape[1],), np.nan, dtype=np.float32)
        return np.mean(arr[mask], axis=0)

    q_mean_engaged = phase_joint_mean(engaged_mask, q_err_abs)
    tau_mean_engaged = phase_joint_mean(engaged_mask, tau_abs)
    q_mean_active = phase_joint_mean(engaged_active_mask, q_err_abs)
    tau_mean_active = phase_joint_mean(engaged_active_mask, tau_abs)

    def top_joint(mean_values: np.ndarray) -> tuple[str | None, float | None]:
        finite = np.isfinite(mean_values)
        if not np.any(finite):
            return None, None
        idx = int(np.nanargmax(mean_values))
        return JOINT_NAMES[idx], float(mean_values[idx])

    top_q_engaged, top_q_engaged_val = top_joint(q_mean_engaged)
    top_tau_engaged, top_tau_engaged_val = top_joint(tau_mean_engaged)
    top_q_active, top_q_active_val = top_joint(q_mean_active)
    top_tau_active, top_tau_active_val = top_joint(tau_mean_active)
    hottest_idx = int(np.argmax(np.max(temp, axis=0)))

    return {
        "path": path,
        "samples": n,
        "duration_s": duration_s,
        "engaged_samples": int(np.sum(engaged_mask)),
        "engaged_active_samples": int(np.sum(engaged_active_mask)),
        "gyro_mean_engaged": float(np.mean(gyro_norm[engaged_mask])) if np.any(engaged_mask) else None,
        "sport_vel_mean_engaged": float(np.mean(sport_vel_norm[engaged_mask])) if np.any(engaged_mask) else None,
        "top_q_engaged": top_q_engaged,
        "top_q_engaged_val": top_q_engaged_val,
        "top_tau_engaged": top_tau_engaged,
        "top_tau_engaged_val": top_tau_engaged_val,
        "top_q_active": top_q_active,
        "top_q_active_val": top_q_active_val,
        "top_tau_active": top_tau_active,
        "top_tau_active_val": top_tau_active_val,
        "hottest_joint": JOINT_NAMES[hottest_idx],
        "hottest_temp": float(np.max(temp[:, hottest_idx])),
        "q_mean_engaged": q_mean_engaged,
        "tau_mean_engaged": tau_mean_engaged,
        "q_mean_active": q_mean_active,
        "tau_mean_active": tau_mean_active,
        "legacy_remap_applied": bool(data["legacy_remap_applied"][0]),
    }


def main() -> int:
    args = parse_args()
    paths = _resolve_paths(args)
    if not paths:
        raise SystemExit("No monitor jsonl files found. Use --jsonl or --glob.")

    runs = []
    for path in paths:
        data = _load_monitor_jsonl(path)
        runs.append(
            _summarize_run(
                path=path,
                data=data,
                near_zero_cmd_threshold=args.near_zero_cmd_threshold,
                engaged_qdes_threshold=args.engaged_qdes_threshold,
            )
        )

    print(f"runs: {len(runs)}")
    print()
    print("Per-run hotspots:")
    for run in runs:
        print(f"  {run['path']}")
        if run["legacy_remap_applied"]:
            print("    joint_order=legacy_sdk_remapped_to_policy")
        else:
            print("    joint_order=policy")
        print(
            f"    samples={run['samples']} duration_s={_format_float(run['duration_s'], 2)}"
            f" engaged={run['engaged_samples']} active={run['engaged_active_samples']}"
        )
        print(
            f"    engaged_q_err={run['top_q_engaged']} ({_format_float(run['top_q_engaged_val'])})"
            f" engaged_tau={run['top_tau_engaged']} ({_format_float(run['top_tau_engaged_val'])})"
        )
        print(
            f"    active_q_err={run['top_q_active']} ({_format_float(run['top_q_active_val'])})"
            f" active_tau={run['top_tau_active']} ({_format_float(run['top_tau_active_val'])})"
        )
        print(
            f"    hottest_joint={run['hottest_joint']} peak_temp={_format_float(run['hottest_temp'], 1)}"
            f" gyro_mean_engaged={_format_float(run['gyro_mean_engaged'])}"
            f" sport_vel_mean_engaged={_format_float(run['sport_vel_mean_engaged'])}"
        )

    print()
    print("Cross-run hotspot frequency:")
    for label, key in (
        ("engaged_q_err", "top_q_engaged"),
        ("engaged_tau", "top_tau_engaged"),
        ("active_q_err", "top_q_active"),
        ("active_tau", "top_tau_active"),
        ("hottest_temp", "hottest_joint"),
    ):
        counts: dict[str, int] = {}
        for run in runs:
            name = run[key]
            if name is None:
                continue
            counts[str(name)] = counts.get(str(name), 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        print(f"  {label}:")
        for joint_name, count in ranked[:5]:
            print(f"    {joint_name}: {count}/{len(runs)} runs")

    print()
    print("Cross-run average joint rankings:")
    for title, key in (
        ("engaged q_err mean_abs", "q_mean_engaged"),
        ("engaged tau mean_abs", "tau_mean_engaged"),
        ("active q_err mean_abs", "q_mean_active"),
        ("active tau mean_abs", "tau_mean_active"),
    ):
        stack = np.asarray([run[key] for run in runs], dtype=np.float32)
        joint_means = np.nanmean(stack, axis=0)
        order = np.argsort(np.nan_to_num(joint_means, nan=-np.inf))[::-1]
        print(f"  {title}:")
        for rank, idx in enumerate(order[:5], start=1):
            print(f"    {rank}. {JOINT_NAMES[int(idx)]}: {_format_float(float(joint_means[int(idx)]))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
