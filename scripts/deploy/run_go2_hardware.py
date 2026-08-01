#!/usr/bin/env python3
"""Bundle-driven low-level deployment runner for Unitree Go2 hardware.

This is the first repo-native hardware bring-up shell for frozen deployment
bundles. It intentionally mirrors the safe operational flow of the old
`sim2real_unitree_sdk2py` bring-up scripts while reading the actual exported
bundle contract from this repo:

- mode switch into low-level DDS
- wait for robot state
- zero torque gate
- move to default stance
- hold default until operator start
- run the TorchScript bundle at 50 Hz

The script is deliberately conservative. It prioritizes correctness of:

- observation order
- history update semantics
- action scaling
- default joint pose
- operator-controlled start/stop

Current intentional limitations for first bring-up:

- base linear velocity is still zero-filled in the observation
- forward-only command gating is the safe default
- stance-only validation can be used before any policy walking
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK2PY_ROOT = REPO_ROOT / "reference_repos" / "sim2real_unitree_sdk2py"
MODE_SWITCH_SCRIPT = SDK2PY_ROOT / "example" / "go2" / "low_level" / "debug" / "mode_switch.py"


GO2_HW_TO_POLICY = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
GO2_POLICY_TO_HW = np.zeros_like(GO2_HW_TO_POLICY)
GO2_POLICY_TO_HW[GO2_HW_TO_POLICY] = np.arange(12, dtype=np.int64)

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self) -> None:
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.button = [0] * 16

    def set(self, data: bytes) -> None:
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


def projected_gravity_from_quat(quaternion_wxyz: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    qw, qx, qy, qz = quaternion_wxyz
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


def _find_exported_artifact(bundle_dir: Path, suffix: str) -> Path:
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(suffix):
            artifact_path = bundle_dir / artifact
            if artifact_path.exists():
                return artifact_path
    raise SystemExit(f"Could not find artifact ending with {suffix!r} in {bundle_dir}")


def _ensure_sdk_import_path() -> None:
    sdk_path = str(SDK2PY_ROOT)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)


def _prepare_low_level_mode(net_if: str) -> None:
    if not MODE_SWITCH_SCRIPT.exists():
        raise SystemExit(f"Mode switch script not found: {MODE_SWITCH_SCRIPT}")
    subprocess.run([sys.executable, str(MODE_SWITCH_SCRIPT), net_if], check=True)


def create_zero_cmd(cmd: Any) -> None:
    for motor in cmd.motor_cmd:
        motor.q = 0.0
        motor.qd = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0


def init_cmd_go(cmd: Any) -> None:
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for motor in cmd.motor_cmd:
        motor.mode = 0x0A
        motor.q = POS_STOP_F
        motor.qd = VEL_STOP_F
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0


@dataclass
class HardwareContract:
    policy_obs_dim: int
    policy_history_length: int
    action_dim: int
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    command_default: np.ndarray


class Go2HardwareRunner:
    def __init__(
        self,
        bundle_dir: Path,
        net_if: str,
        dry_run: bool = False,
        stance_only: bool = False,
        forward_only: bool = True,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.net_if = net_if
        self.dry_run = dry_run
        self.stance_only = stance_only
        self.forward_only = forward_only
        self.remote = RemoteController()
        self.low_state = None
        self._warned_zero_lin_vel = False

        manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
        if manifest.get("policy_kind") != "blind_history_policy":
            raise SystemExit(
                "This first hardware runner currently supports only "
                "`blind_history_policy` bundles."
            )

        deploy_cfg = json.loads(_find_exported_artifact(bundle_dir, ".deploy_config.json").read_text())
        metadata = json.loads(_find_exported_artifact(bundle_dir, ".export_metadata.json").read_text())
        policy_path = _find_exported_artifact(bundle_dir, ".torchscript.pt")

        tensor_contract = metadata["tensor_contract"]
        self.contract = HardwareContract(
            policy_obs_dim=int(tensor_contract["policy_obs_dim"]),
            policy_history_length=int(deploy_cfg["observations"]["policy_history_length"]),
            action_dim=int(tensor_contract["action_dim"]),
            default_joint_pos=np.asarray(deploy_cfg["robot"]["default_joint_pos"], dtype=np.float32),
            joint_stiffness=np.asarray(deploy_cfg["robot"]["joint_stiffness"], dtype=np.float32),
            joint_damping=np.asarray(deploy_cfg["robot"]["joint_damping"], dtype=np.float32),
            action_scale=np.asarray(deploy_cfg["actions"]["scale"], dtype=np.float32),
            action_offset=np.asarray(deploy_cfg["actions"]["offset"], dtype=np.float32),
            command_default=np.asarray(deploy_cfg["commands"]["base_velocity"]["default"], dtype=np.float32),
        )
        self.control_dt = float(deploy_cfg["control"]["step_dt"])
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()
        self.history = np.zeros((self.contract.policy_history_length, self.contract.policy_obs_dim), dtype=np.float32)
        self.last_action = np.zeros(self.contract.action_dim, dtype=np.float32)

        if dry_run:
            self._dds_ready = False
            return

        _ensure_sdk_import_path()
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__LowCmd_ as LowCmdGo,
            unitree_go_msg_dds__LowState_ as LowStateGo,
        )

        self.ChannelFactoryInitialize = ChannelFactoryInitialize
        self.ChannelPublisher = ChannelPublisher
        self.ChannelSubscriber = ChannelSubscriber
        self.CRC = CRC
        self.LowCmdGo = LowCmdGo
        self.LowStateGo = LowStateGo

        _prepare_low_level_mode(net_if)
        ChannelFactoryInitialize(0, net_if)
        self.low_cmd = LowCmdGo()
        self.low_state = LowStateGo()
        self.pub = ChannelPublisher("rt/lowcmd", type(self.low_cmd))
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", type(self.low_state))
        self.sub.Init(self._on_lowstate, 10)
        init_cmd_go(self.low_cmd)
        self.crc = CRC()
        self._dds_ready = True

    def _on_lowstate(self, msg: Any) -> None:
        self.low_state = msg
        self.remote.set(self.low_state.wireless_remote)

    def _send_cmd(self) -> None:
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def _wait_for_state(self) -> None:
        while self.low_state is None or getattr(self.low_state, "tick", 0) == 0:
            time.sleep(self.control_dt)

    def _policy_order_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q_hw = np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)
        dq_hw = np.array([self.low_state.motor_state[i].dq for i in range(self.contract.action_dim)], dtype=np.float32)
        return q_hw[GO2_HW_TO_POLICY], dq_hw[GO2_HW_TO_POLICY]

    def _policy_obs(self, command: np.ndarray) -> np.ndarray:
        q_policy, dq_policy = self._policy_order_joint_state()
        quat = self.low_state.imu_state.quaternion
        gyro = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        gravity = projected_gravity_from_quat(quat)
        if not self._warned_zero_lin_vel:
            print(
                "[WARN] base_lin_vel is currently zero-filled in hardware observation. "
                "Treat first walking as a conservative bring-up, not final sim2real validation."
            )
            self._warned_zero_lin_vel = True
        obs = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),  # TODO: add real base lin vel estimate on robot
                gyro.astype(np.float32),
                gravity,
                command.astype(np.float32),
                (q_policy - self.contract.default_joint_pos).astype(np.float32),
                dq_policy.astype(np.float32),
                self.last_action.astype(np.float32),
            ],
            axis=0,
        )
        if obs.shape[0] != self.contract.policy_obs_dim:
            raise RuntimeError(f"Expected {self.contract.policy_obs_dim}D obs, got {obs.shape[0]}")
        return obs

    def _append_history(self, obs: np.ndarray) -> None:
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs

    def _run_policy(self, obs: np.ndarray) -> np.ndarray:
        policy_obs = torch.from_numpy(obs).unsqueeze(0)
        history = torch.from_numpy(self.history.reshape(1, -1))
        with torch.no_grad():
            action = self.policy(policy_obs, history).cpu().numpy().squeeze(0)
        return action.astype(np.float32)

    def _target_hw(self, action: np.ndarray) -> np.ndarray:
        q_target_policy = self.contract.action_offset + self.contract.action_scale * action
        return q_target_policy[GO2_POLICY_TO_HW]

    def zero_torque_state(self) -> None:
        print("[STATE] Zero Torque. Press START on the wireless remote to continue.")
        while self.remote.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self._send_cmd()
            time.sleep(self.control_dt)

    def move_to_default(self) -> None:
        print("[STATE] Moving to default stance.")
        steps = int(round(2.0 / self.control_dt))
        q_hw = np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)
        default_hw = self.contract.default_joint_pos[GO2_POLICY_TO_HW]
        for step in range(steps):
            alpha = float(step + 1) / float(steps)
            q_target = (1.0 - alpha) * q_hw + alpha * default_hw
            for i in range(self.contract.action_dim):
                motor = self.low_cmd.motor_cmd[i]
                motor.q = float(q_target[i])
                motor.kp = float(self.contract.joint_stiffness[GO2_HW_TO_POLICY[i]])
                motor.kd = float(self.contract.joint_damping[GO2_HW_TO_POLICY[i]])
                motor.tau = 0.0
            self._send_cmd()
            time.sleep(self.control_dt)

    def hold_default(self) -> None:
        print("[STATE] Holding default stance. Press A to start policy. Press SELECT to abort.")
        default_hw = self.contract.default_joint_pos[GO2_POLICY_TO_HW]
        while self.remote.button[KeyMap.A] != 1:
            if self.remote.button[KeyMap.select] == 1:
                raise KeyboardInterrupt("Operator aborted during hold_default")
            for i in range(self.contract.action_dim):
                motor = self.low_cmd.motor_cmd[i]
                motor.q = float(default_hw[i])
                motor.kp = float(self.contract.joint_stiffness[GO2_HW_TO_POLICY[i]])
                motor.kd = float(self.contract.joint_damping[GO2_HW_TO_POLICY[i]])
                motor.tau = 0.0
            self._send_cmd()
            time.sleep(self.control_dt)

    def run_policy_loop(self, max_steps: int) -> None:
        if self.forward_only:
            print("[STATE] Running policy in forward-only safety mode. Press SELECT to stop.")
        else:
            print("[STATE] Running policy with full remote command mapping. Press SELECT to stop.")
        command = self.contract.command_default.copy()
        for step_idx in range(max_steps):
            if self.remote.button[KeyMap.select] == 1:
                print("[INFO] Operator stop received.")
                break
            if self.forward_only:
                command[:] = np.array([self.remote.ly, 0.0, 0.0], dtype=np.float32)
            else:
                command[:] = np.array([self.remote.ly, -self.remote.lx, -self.remote.rx], dtype=np.float32)
            obs = self._policy_obs(command)
            self._append_history(obs)
            action = self._run_policy(obs)
            target_hw = self._target_hw(action)
            for i in range(self.contract.action_dim):
                motor = self.low_cmd.motor_cmd[i]
                motor.q = float(target_hw[i])
                motor.kp = float(self.contract.joint_stiffness[GO2_HW_TO_POLICY[i]])
                motor.kd = float(self.contract.joint_damping[GO2_HW_TO_POLICY[i]])
                motor.tau = 0.0
            self.last_action = action.copy()
            self._send_cmd()
            time.sleep(self.control_dt)
            if step_idx % 100 == 0:
                print(
                    f"[INFO] step={step_idx} cmd=({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}) "
                    f"action_abs_mean={float(np.mean(np.abs(action))):.3f}"
                )

    def run(self, max_steps: int) -> None:
        print(f"[INFO] Bundle: {self.bundle_dir}")
        print(f"[INFO] policy_obs_dim={self.contract.policy_obs_dim} history_len={self.contract.policy_history_length}")
        print(f"[INFO] action_dim={self.contract.action_dim} control_dt={self.control_dt:.3f}")
        print(f"[INFO] stance_only={self.stance_only} forward_only={self.forward_only}")
        if self.dry_run:
            print("[INFO] Dry run only. Bundle contract resolved successfully.")
            return
        self._wait_for_state()
        self.zero_torque_state()
        self.move_to_default()
        self.hold_default()
        if self.stance_only:
            print("[INFO] Stance-only bring-up completed. Exiting before policy walk.")
            return
        self.run_policy_loop(max_steps=max_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--net-if", default="eno1")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true", help="Resolve bundle contract without touching Unitree DDS.")
    parser.add_argument(
        "--stance-only",
        action="store_true",
        help="Stop after move-to-default and hold_default. Useful for first hardware-shell validation.",
    )
    parser.add_argument(
        "--allow-lateral-yaw",
        action="store_true",
        help="Disable the safe forward-only command clamp and allow lateral/yaw remote commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Go2HardwareRunner(
        bundle_dir=Path(args.bundle_dir),
        net_if=args.net_if,
        dry_run=args.dry_run,
        stance_only=args.stance_only,
        forward_only=not args.allow_lateral_yaw,
    )
    runner.run(max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
