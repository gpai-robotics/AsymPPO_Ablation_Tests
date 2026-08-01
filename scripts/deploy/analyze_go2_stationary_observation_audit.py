#!/usr/bin/env python3
"""Analyze a read-only Go2 LowState capture against a deployment bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPTS = REPO_ROOT / "scripts" / "deploy"
if str(DEPLOY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DEPLOY_SCRIPTS))

from history_layout import flatten_policy_history, resolve_history_layout


# Index a Unitree hardware-order array with this map to produce policy order.
GO2_HW_INDEX_FOR_POLICY = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=np.int64)
# Index a policy-order array with this map to produce Unitree hardware order.
GO2_POLICY_INDEX_FOR_HW = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10], dtype=np.int64)
if not (
    np.array_equal(GO2_POLICY_INDEX_FOR_HW[GO2_HW_INDEX_FOR_POLICY], np.arange(12))
    and np.array_equal(GO2_HW_INDEX_FOR_POLICY[GO2_POLICY_INDEX_FOR_HW], np.arange(12))
):
    raise RuntimeError("Go2 hardware/policy joint maps are not inverse permutations.")
POLICY_JOINT_NAMES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowstate-jsonl", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    return parser.parse_args()


def find_artifact(bundle_dir: Path, suffix: str) -> Path:
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    for name in manifest["exported_artifacts"]:
        if name.endswith(suffix):
            path = bundle_dir / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing bundle artifact ending in {suffix!r}.")


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion_wxyz
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            snapshot = row.get("lowstate", {}).get("snapshot")
            if snapshot is not None:
                samples.append(
                    {
                        "monotonic_ns": int(row["monotonic_ns"]),
                        "snapshot": snapshot,
                    }
                )
    if len(samples) < 2:
        raise SystemExit(f"Capture has fewer than two valid LowState samples: {path}")
    return samples


def select_control_samples(samples: list[dict[str, Any]], control_dt_s: float) -> list[dict[str, Any]]:
    selected = [samples[0]]
    next_ns = samples[0]["monotonic_ns"] + int(control_dt_s * 1e9)
    for sample in samples[1:]:
        if sample["monotonic_ns"] >= next_ns:
            selected.append(sample)
            next_ns += int(control_dt_s * 1e9)
    return selected


def main() -> int:
    args = parse_args()
    capture_path = Path(args.lowstate_jsonl).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()
    output_path = Path(args.json_out).resolve()

    deploy_cfg = json.loads(find_artifact(bundle_dir, ".deploy_config.json").read_text())
    policy_path = find_artifact(bundle_dir, ".torchscript.pt")
    observations_cfg = deploy_cfg["observations"]
    policy_order = observations_cfg["policy_order"]
    policy_dim = int(observations_cfg["policy_dim"])
    history_length = int(observations_cfg["policy_history_length"])
    history_layout = resolve_history_layout(observations_cfg)
    default_joint_pos = np.asarray(deploy_cfg["robot"]["default_joint_pos"], dtype=np.float32)

    raw_samples = load_samples(capture_path)
    raw_times_s = np.asarray([sample["monotonic_ns"] for sample in raw_samples], dtype=np.float64) * 1e-9
    raw_intervals = np.diff(raw_times_s)
    control_dt = 1.0 / args.control_rate_hz
    control_samples = select_control_samples(raw_samples, control_dt)
    if len(control_samples) <= int(args.warmup_s * args.control_rate_hz):
        raise SystemExit("Capture is too short after the requested warmup.")

    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    history = np.zeros((history_length, policy_dim), dtype=np.float32)
    previous_action = np.zeros(12, dtype=np.float32)

    observations = []
    actions = []
    q_policy_rows = []
    dq_policy_rows = []
    gyro_rows = []
    gravity_rows = []
    quaternion_norms = []
    foot_force_rows = []
    temperature_rows = []
    remote_rows = []

    for sample_idx, sample in enumerate(control_samples):
        snapshot = sample["snapshot"]
        q_hw = np.asarray(snapshot["joint_q_12"], dtype=np.float32)
        dq_hw = np.asarray(snapshot["joint_dq_12"], dtype=np.float32)
        q_policy = q_hw[GO2_HW_INDEX_FOR_POLICY]
        dq_policy = dq_hw[GO2_HW_INDEX_FOR_POLICY]
        quaternion = np.asarray(snapshot["imu_quaternion_wxyz"], dtype=np.float32)
        gyro = np.asarray(snapshot["imu_gyro_xyz"], dtype=np.float32)
        gravity = projected_gravity(quaternion)
        command = np.zeros(3, dtype=np.float32)

        terms = {
            "base_ang_vel": gyro,
            "projected_gravity": gravity,
            "velocity_commands": command,
            "joint_pos_rel": q_policy - default_joint_pos,
            "joint_vel_rel": dq_policy,
            "last_action": previous_action,
        }
        obs = np.concatenate(
            [np.asarray(terms[str(term["name"])], dtype=np.float32) for term in policy_order]
        )
        if obs.shape != (policy_dim,):
            raise RuntimeError(f"Constructed observation has shape {obs.shape}, expected {(policy_dim,)}.")

        if sample_idx == 0:
            history[:] = obs
        else:
            history[:-1] = history[1:]
            history[-1] = obs
        history_flat = flatten_policy_history(history, policy_order, layout=history_layout)
        with torch.inference_mode():
            action = (
                policy(
                    torch.from_numpy(obs).reshape(1, -1),
                    torch.from_numpy(history_flat).reshape(1, -1),
                )
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
        previous_action = action

        observations.append(obs)
        actions.append(action)
        q_policy_rows.append(q_policy)
        dq_policy_rows.append(dq_policy)
        gyro_rows.append(gyro)
        gravity_rows.append(gravity)
        quaternion_norms.append(float(np.linalg.norm(quaternion)))
        foot_force_rows.append(snapshot["foot_force"])
        temperature_rows.append(snapshot["temperature_hint"])
        remote = snapshot["remote"]
        remote_rows.append([remote["lx"], remote["ly"], remote["rx"], remote["ry"]])

    warmup = int(round(args.warmup_s * args.control_rate_hz))
    measured_slice = slice(warmup, None)
    obs_array = np.asarray(observations)[measured_slice]
    action_array = np.asarray(actions)[measured_slice]
    q_policy = np.asarray(q_policy_rows)[measured_slice]
    dq_policy = np.asarray(dq_policy_rows)[measured_slice]
    gyro = np.asarray(gyro_rows)[measured_slice]
    gravity = np.asarray(gravity_rows)[measured_slice]
    quaternion_norm = np.asarray(quaternion_norms)[measured_slice]
    foot_force = np.asarray(foot_force_rows)[measured_slice]
    temperature = np.asarray(temperature_rows)[measured_slice]
    remote = np.asarray(remote_rows)[measured_slice]

    q_rel = q_policy - default_joint_pos
    joint_summary = {}
    for idx, name in enumerate(POLICY_JOINT_NAMES):
        joint_summary[name] = {
            "q_mean_rad": float(np.mean(q_policy[:, idx])),
            "q_std_rad": float(np.std(q_policy[:, idx])),
            "q_rel_mean_rad": float(np.mean(q_rel[:, idx])),
            "q_rel_abs_mean_rad": float(np.mean(np.abs(q_rel[:, idx]))),
            "dq_abs_mean_rad_s": float(np.mean(np.abs(dq_policy[:, idx]))),
            "dq_abs_max_rad_s": float(np.max(np.abs(dq_policy[:, idx]))),
            "shadow_action_abs_mean": float(np.mean(np.abs(action_array[:, idx]))),
            "shadow_action_abs_max": float(np.max(np.abs(action_array[:, idx]))),
        }

    raw_hz = 1.0 / float(np.mean(raw_intervals))
    control_times = np.asarray(
        [sample["monotonic_ns"] for sample in control_samples], dtype=np.float64
    ) * 1e-9
    control_intervals = np.diff(control_times)
    gravity_norm = np.linalg.norm(gravity, axis=1)
    remote_abs_max = float(np.max(np.abs(remote)))
    q_rel_abs_max = float(np.max(np.abs(q_rel)))
    action_abs_max = float(np.max(np.abs(action_array)))

    checks = [
        {
            "name": "raw_lowstate_rate",
            "ok": 450.0 <= raw_hz <= 550.0,
            "detail": f"{raw_hz:.2f} Hz expected approximately 500 Hz",
        },
        {
            "name": "control_sample_count",
            "ok": len(obs_array) >= int(5.0 * args.control_rate_hz),
            "detail": f"{len(obs_array)} measured 50 Hz samples",
        },
        {
            "name": "finite_observation",
            "ok": bool(np.isfinite(obs_array).all()),
            "detail": f"shape={list(obs_array.shape)}",
        },
        {
            "name": "finite_shadow_action",
            "ok": bool(np.isfinite(action_array).all()),
            "detail": f"max_abs={action_abs_max:.4f}",
        },
        {
            "name": "stable_pose_proximity",
            "ok": q_rel_abs_max <= 0.45,
            "detail": f"max_abs_joint_offset={q_rel_abs_max:.4f} rad, limit=0.4500 rad",
        },
        {
            "name": "bounded_shadow_action",
            "ok": action_abs_max <= 4.0,
            "detail": f"max_abs={action_abs_max:.4f}, review_limit=4.0000",
        },
        {
            "name": "quaternion_norm",
            "ok": float(np.max(np.abs(quaternion_norm - 1.0))) <= 0.05,
            "detail": f"mean={float(np.mean(quaternion_norm)):.5f}",
        },
        {
            "name": "projected_gravity_norm",
            "ok": float(np.max(np.abs(gravity_norm - 1.0))) <= 0.08,
            "detail": f"mean={float(np.mean(gravity_norm)):.5f}",
        },
        {
            "name": "stationary_joint_velocity",
            "ok": float(np.percentile(np.abs(dq_policy), 95)) <= 0.5,
            "detail": f"p95_abs={float(np.percentile(np.abs(dq_policy), 95)):.4f} rad/s",
        },
        {
            "name": "stationary_gyro",
            "ok": float(np.percentile(np.abs(gyro), 95)) <= 0.25,
            "detail": f"p95_abs={float(np.percentile(np.abs(gyro), 95)):.4f} rad/s",
        },
        {
            "name": "remote_neutral",
            "ok": remote_abs_max <= 0.1,
            "detail": f"axis_abs_max={remote_abs_max:.4f}",
        },
    ]
    report = {
        "status": "pass" if all(check["ok"] for check in checks) else "review",
        "safety": {
            "read_only": True,
            "lowcmd_publisher_created": False,
            "policy_actions_transmitted": False,
            "shadow_policy_inference_only": True,
        },
        "capture": {
            "path": str(capture_path),
            "raw_sample_count": len(raw_samples),
            "raw_rate_hz": raw_hz,
            "raw_interval_ms": scalar_stats(raw_intervals * 1000.0),
            "control_sample_count": len(control_samples),
            "control_interval_ms": scalar_stats(control_intervals * 1000.0),
            "warmup_s": args.warmup_s,
            "measured_sample_count": len(obs_array),
        },
        "bundle": {
            "path": str(bundle_dir),
            "policy_path": str(policy_path),
            "policy_dim": policy_dim,
            "history_length": history_length,
            "history_dim": policy_dim * history_length,
            "history_layout": history_layout,
            "hardware_indices_gathered_into_policy_order": GO2_HW_INDEX_FOR_POLICY.tolist(),
            "policy_indices_gathered_into_hardware_order": GO2_POLICY_INDEX_FOR_HW.tolist(),
        },
        "observations": {
            "base_ang_vel": stats(gyro),
            "projected_gravity": stats(gravity),
            "projected_gravity_norm": scalar_stats(gravity_norm),
            "quaternion_norm": scalar_stats(quaternion_norm),
            "joint_pos_rel": stats(q_rel),
            "joint_vel": stats(dq_policy),
            "remote_axes": stats(remote),
            "foot_force": stats(foot_force),
            "temperature": stats(temperature),
            "full_observation_norm": scalar_stats(np.linalg.norm(obs_array, axis=1)),
        },
        "shadow_policy": {
            "action": stats(action_array),
            "action_abs_max": action_abs_max,
            "action_delta_abs_mean": float(np.mean(np.abs(np.diff(action_array, axis=0)))),
        },
        "joints": joint_summary,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"status: {report['status']}")
    print(f"raw_lowstate_hz: {raw_hz:.2f}")
    print(f"measured_control_samples: {len(obs_array)}")
    print(f"gyro_p95_abs: {np.percentile(np.abs(gyro), 95):.4f} rad/s")
    print(f"joint_dq_p95_abs: {np.percentile(np.abs(dq_policy), 95):.4f} rad/s")
    print(f"joint_offset_abs_max: {q_rel_abs_max:.4f} rad")
    print(f"gravity_mean: {np.mean(gravity, axis=0).round(5).tolist()}")
    print(f"shadow_action_abs_max: {action_abs_max:.4f}")
    print(f"report: {output_path}")
    for check in checks:
        print(f"  [{'PASS' if check['ok'] else 'REVIEW'}] {check['name']}: {check['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
