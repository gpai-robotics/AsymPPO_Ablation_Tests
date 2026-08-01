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

Recent safety additions:

- history is primed from the live standing observation before walking starts
- operator stop and exceptions trigger a short posture-hold stop routine
- remote commands pass through a small deadband to reduce joystick noise
- seated startup is supported through a staged sit-to-stand sequence
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

from history_layout import flatten_policy_history, resolve_history_layout


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK2PY_ROOT = REPO_ROOT / "reference_repos" / "sim2real_unitree_sdk2py"
MODE_SWITCH_SCRIPT = SDK2PY_ROOT / "example" / "go2" / "low_level" / "debug" / "mode_switch.py"


# Index a policy-order array with this map to produce Unitree hardware order.
GO2_POLICY_INDEX_FOR_HW = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10], dtype=np.int64)
# Index a Unitree hardware-order array with this map to produce policy order.
GO2_HW_INDEX_FOR_POLICY = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=np.int64)
if not (
    np.array_equal(GO2_POLICY_INDEX_FOR_HW[GO2_HW_INDEX_FOR_POLICY], np.arange(12))
    and np.array_equal(GO2_HW_INDEX_FOR_POLICY[GO2_POLICY_INDEX_FOR_HW], np.arange(12))
):
    raise RuntimeError("Go2 hardware/policy joint maps are not inverse permutations.")

GO2_HW_JOINT_NAMES = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
]

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
        motor.mode = 0x01
        motor.q = 0.0
        motor.qd = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0


def create_damping_cmd(cmd: Any) -> None:
    for motor in cmd.motor_cmd:
        motor.mode = 0x01
        motor.q = 0.0
        motor.qd = 0.0
        motor.kp = 0.0
        motor.kd = 8.0
        motor.tau = 0.0


def init_cmd_go(cmd: Any) -> None:
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for motor in cmd.motor_cmd:
        motor.mode = 0x01
        motor.q = POS_STOP_F
        motor.qd = VEL_STOP_F
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0


@dataclass
class HardwareContract:
    policy_obs_dim: int
    policy_history_length: int
    history_layout: str
    action_dim: int
    policy_order: list[dict[str, Any]]
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
        mapping_only: bool = False,
        stance_only: bool = False,
        forward_only: bool = True,
        command_deadband: float = 0.05,
        startup_posture: str = "seated",
        exit_mode: str = "damping",
    ) -> None:
        self.bundle_dir = bundle_dir
        self.net_if = net_if
        self.dry_run = dry_run
        self.mapping_only = mapping_only
        self.stance_only = stance_only
        self.forward_only = forward_only
        self.command_deadband = float(command_deadband)
        self.startup_posture = startup_posture
        self.exit_mode = exit_mode
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
            history_layout=resolve_history_layout(deploy_cfg["observations"]),
            action_dim=int(tensor_contract["action_dim"]),
            policy_order=list(deploy_cfg["observations"].get("policy_order", [])),
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
        self.policy_offsets = self._build_policy_offsets()

        if dry_run:
            self._dds_ready = False
            return

        _ensure_sdk_import_path()
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.utils.crc import CRC
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_ as LowCmdGo,
                unitree_go_msg_dds__LowState_ as LowStateGo,
            )
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown module"
            raise SystemExit(
                "Missing Unitree DDS Python dependency while starting hardware runner.\n"
                f"Missing module: {missing}\n\n"
                "This runner depends on the `sim2real_unitree_sdk2py` stack and its Python "
                "DDS dependency chain, especially `cyclonedds==0.10.2`.\n\n"
                "Recommended fix:\n"
                "1. Create or activate a dedicated hardware/SDK environment.\n"
                "2. Install the local SDK repo in editable mode:\n"
                "   cd reference_repos/sim2real_unitree_sdk2py && pip install -e .\n"
                "3. Verify DDS import:\n"
                "   python - <<'PY'\n"
                "import cyclonedds\n"
                "print(cyclonedds.__version__)\n"
                "PY\n\n"
                "If `cyclonedds` still fails to install, follow the CycloneDDS build notes in:\n"
                "reference_repos/sim2real_unitree_sdk2py/README.md\n"
            ) from exc

        self.ChannelFactoryInitialize = ChannelFactoryInitialize
        self.ChannelPublisher = ChannelPublisher
        self.ChannelSubscriber = ChannelSubscriber
        self.MotionSwitcherClient = MotionSwitcherClient
        self.SportClient = SportClient
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
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
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

    def _check_abort_button(self) -> None:
        if self.remote.button[KeyMap.select] == 1:
            raise KeyboardInterrupt("Operator aborted with SELECT.")

    def _policy_order_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q_hw = np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)
        dq_hw = np.array([self.low_state.motor_state[i].dq for i in range(self.contract.action_dim)], dtype=np.float32)
        return q_hw[GO2_HW_INDEX_FOR_POLICY], dq_hw[GO2_HW_INDEX_FOR_POLICY]

    def _build_policy_offsets(self) -> dict[str, tuple[int, int]]:
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for term in self.contract.policy_order:
            name = str(term["name"])
            dim = int(term["dim"])
            offsets[name] = (cursor, cursor + dim)
            cursor += dim
        if cursor != self.contract.policy_obs_dim:
            raise RuntimeError(
                f"Deploy config policy_order sums to {cursor}, expected {self.contract.policy_obs_dim}."
            )
        return offsets

    def _policy_obs(self, command: np.ndarray) -> np.ndarray:
        q_policy, dq_policy = self._policy_order_joint_state()
        quat = self.low_state.imu_state.quaternion
        gyro = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        gravity = projected_gravity_from_quat(quat)
        if "base_lin_vel" in self.policy_offsets and not self._warned_zero_lin_vel:
            print(
                "[WARN] base_lin_vel is currently zero-filled in hardware observation. "
                "Treat first walking as a conservative bring-up, not final sim2real validation."
            )
            self._warned_zero_lin_vel = True
        term_values = {
            "base_lin_vel": np.zeros(3, dtype=np.float32),  # TODO: add real base lin vel estimate for C1-style contracts
            "base_ang_vel": gyro.astype(np.float32),
            "projected_gravity": gravity,
            "velocity_commands": command.astype(np.float32),
            "joint_pos_rel": (q_policy - self.contract.default_joint_pos).astype(np.float32),
            "joint_vel_rel": dq_policy.astype(np.float32),
            "last_action": self.last_action.astype(np.float32),
        }
        obs_parts = []
        for term in self.contract.policy_order:
            name = str(term["name"])
            if name not in term_values:
                raise RuntimeError(f"Unsupported policy observation term in deploy config: {name}")
            value = np.asarray(term_values[name], dtype=np.float32)
            expected_dim = int(term["dim"])
            if value.shape != (expected_dim,):
                raise RuntimeError(f"Observation term {name} expected shape {(expected_dim,)}, got {value.shape}.")
            obs_parts.append(value)
        obs = np.concatenate(obs_parts, axis=0)
        if obs.shape[0] != self.contract.policy_obs_dim:
            raise RuntimeError(f"Expected {self.contract.policy_obs_dim}D obs, got {obs.shape[0]}")
        return obs

    def _append_history(self, obs: np.ndarray) -> None:
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs

    def _prime_history(self, obs: np.ndarray) -> None:
        self.history[:] = obs[None, :]

    def _apply_deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self.command_deadband else value

    def _run_policy(self, obs: np.ndarray) -> np.ndarray:
        policy_obs = torch.from_numpy(obs).unsqueeze(0)
        history_flat = flatten_policy_history(
            self.history,
            self.contract.policy_order,
            layout=self.contract.history_layout,
        )
        history = torch.from_numpy(history_flat.reshape(1, -1))
        with torch.no_grad():
            action = self.policy(policy_obs, history).cpu().numpy().squeeze(0)
        return action.astype(np.float32)

    def _target_hw(self, action: np.ndarray) -> np.ndarray:
        q_target_policy = self.contract.action_offset + self.contract.action_scale * action
        return q_target_policy[GO2_POLICY_INDEX_FOR_HW]

    def zero_torque_state(self) -> None:
        print("[STATE] Zero Torque. Press START on the wireless remote to continue. Press SELECT to abort.")
        while self.remote.button[KeyMap.start] != 1:
            self._check_abort_button()
            create_zero_cmd(self.low_cmd)
            self._send_cmd()
            time.sleep(self.control_dt)

    def _command_pose_once(self, q_target_hw: np.ndarray) -> None:
        for i in range(self.contract.action_dim):
            motor = self.low_cmd.motor_cmd[i]
            motor.mode = 0x01
            motor.q = float(q_target_hw[i])
            motor.qd = 0.0
            motor.kp = float(self.contract.joint_stiffness[GO2_POLICY_INDEX_FOR_HW[i]])
            motor.kd = float(self.contract.joint_damping[GO2_POLICY_INDEX_FOR_HW[i]])
            motor.tau = 0.0
        self._send_cmd()

    def _interpolate_pose(self, q_start_hw: np.ndarray, q_end_hw: np.ndarray, duration_s: float, label: str) -> None:
        print(f"[STATE] {label}")
        steps = max(1, int(round(duration_s / self.control_dt)))
        for step in range(steps):
            self._check_abort_button()
            alpha = float(step + 1) / float(steps)
            q_target = (1.0 - alpha) * q_start_hw + alpha * q_end_hw
            self._command_pose_once(q_target)
            time.sleep(self.control_dt)

    def _hold_pose(self, q_target_hw: np.ndarray, duration_s: float, label: str) -> None:
        print(f"[STATE] {label}")
        steps = max(1, int(round(duration_s / self.control_dt)))
        for _ in range(steps):
            self._check_abort_button()
            self._command_pose_once(q_target_hw)
            time.sleep(self.control_dt)

    def _default_hw(self) -> np.ndarray:
        return self.contract.default_joint_pos[GO2_POLICY_INDEX_FOR_HW]

    def _current_hw_joint_pos(self) -> np.ndarray:
        return np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)

    def _print_stance_tracking(self, q_target_hw: np.ndarray, *, prefix: str) -> None:
        q_hw = self._current_hw_joint_pos()
        err_hw = q_target_hw - q_hw
        foot_force = [getattr(force, "__float__", lambda: force)() for force in getattr(self.low_state, "foot_force", [])]
        print(
            f"[STANCE] {prefix} max_abs_err={float(np.max(np.abs(err_hw))):.3f} "
            f"rear_err={{RR_thigh:{err_hw[7]:+.3f}, RR_calf:{err_hw[8]:+.3f}, "
            f"RL_thigh:{err_hw[10]:+.3f}, RL_calf:{err_hw[11]:+.3f}}} "
            f"foot_force={foot_force}"
        )
        for idx in (7, 8, 10, 11):
            print(
                f"[STANCE]   {GO2_HW_JOINT_NAMES[idx]:9s} "
                f"actual={q_hw[idx]:+.3f} target={q_target_hw[idx]:+.3f} err={err_hw[idx]:+.3f}"
            )

    def _seated_startup_sequence(self) -> None:
        q_start_hw = np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)
        target_pos_1 = np.array(
            [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65],
            dtype=np.float32,
        )
        target_pos_2 = np.array(
            [0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3],
            dtype=np.float32,
        )
        target_pos_3 = np.array(
            [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65, -0.5, 1.36, -2.65, 0.5, 1.36, -2.65],
            dtype=np.float32,
        )
        self._interpolate_pose(q_start_hw, target_pos_1, duration_s=1.5, label="Seated startup: phase 1/4")
        self._interpolate_pose(target_pos_1, target_pos_2, duration_s=1.5, label="Seated startup: phase 2/4")
        self._hold_pose(target_pos_2, duration_s=1.5, label="Seated startup: phase 3/4 hold")
        self._interpolate_pose(target_pos_2, target_pos_3, duration_s=1.8, label="Seated startup: phase 4/4")
        self._interpolate_pose(target_pos_3, self._default_hw(), duration_s=2.0, label="Seated startup: settle into bundle default stance")
        self._print_stance_tracking(self._default_hw(), prefix="after seated startup")
        print("[STATE] Seated startup sequence completed.")

    def move_to_default(self) -> None:
        if self.startup_posture == "seated":
            self._seated_startup_sequence()
            return
        q_hw = np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)
        self._interpolate_pose(
            q_hw,
            self._default_hw(),
            duration_s=2.5,
            label="Standing startup: direct ramp into bundle default stance",
        )
        print("[STATE] Default stance reached ✅")

    def hold_default(self) -> None:
        print("[STATE] Holding default stance. Press A to start policy. Press SELECT to abort.")
        default_hw = self._default_hw()
        while self.remote.button[KeyMap.A] != 1:
            self._check_abort_button()
            self._command_pose_once(default_hw)
            time.sleep(self.control_dt)

    def hold_stance_only(self) -> None:
        print("[STATE] Stance-only hold active. Robot will keep receiving default stance targets.")
        print("[STATE] Press SELECT when you want to end the stance-only test.")
        default_hw = self._default_hw()
        last_print_t = 0.0
        while True:
            self._check_abort_button()
            self._command_pose_once(default_hw)
            now = time.monotonic()
            if now - last_print_t >= 1.0:
                self._print_stance_tracking(default_hw, prefix="hold")
                last_print_t = now
            time.sleep(self.control_dt)

    def _burst_send(self, make_cmd_fn: Any, repeats: int, sleep_s: float, label: str) -> None:
        print(f"[STATE] {label}")
        for _ in range(repeats):
            make_cmd_fn(self.low_cmd)
            self._send_cmd()
            time.sleep(sleep_s)

    def _release_low_level_mode(self) -> None:
        try:
            self.msc.ReleaseMode()
            print("[STATE] Requested low-level mode release.")
        except Exception as exc:
            print(f"[WARN] Could not release low-level mode cleanly: {exc}")

    def _safe_stop(self) -> None:
        if self.dry_run or not getattr(self, "_dds_ready", False):
            return
        if self.low_state is None or getattr(self.low_state, "tick", 0) == 0:
            return
        if self.exit_mode == "hold":
            q_hw = np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)
            self._hold_pose(
                q_target_hw=q_hw,
                duration_s=0.5,
                label="Safe stop: holding current posture briefly before exit.",
            )
            return
        if self.exit_mode == "damping":
            self._burst_send(
                create_damping_cmd,
                repeats=200,
                sleep_s=0.002,
                label="Safe stop: engaging damping mode.",
            )
            self._release_low_level_mode()
            return
        if self.exit_mode == "zero-torque":
            self._burst_send(
                create_zero_cmd,
                repeats=200,
                sleep_s=0.002,
                label="Safe stop: sending zero-torque commands.",
            )
            self._release_low_level_mode()
            return
        print(f"[WARN] Unknown exit_mode={self.exit_mode!r}; falling back to damping.")
        self._burst_send(
            create_damping_cmd,
            repeats=200,
            sleep_s=0.002,
            label="Safe stop: engaging damping mode.",
        )
        self._release_low_level_mode()

    def _print_mapping_audit(self) -> None:
        source_by_term = {
            "base_lin_vel": "PLACEHOLDER zeros on hardware today",
            "base_ang_vel": "IMU gyroscope from LowState",
            "projected_gravity": "quaternion -> gravity projection",
            "velocity_commands": "mapped remote command",
            "joint_pos_rel": "low-level motor q minus bundle default pose",
            "joint_vel_rel": "low-level motor dq",
            "last_action": "previous deployed policy action",
        }
        obs_lines = []
        for term in self.contract.policy_order:
            name = str(term["name"])
            dim = int(term["dim"])
            source = source_by_term.get(name, "unknown source")
            obs_lines.append(f"\n  - {name} ({dim}): {source}")

        print("[AUDIT] Hardware mapping contract")
        print(
            "[AUDIT] Remote -> command: "
            "forward uses left-stick Y; omni mode uses left-stick Y -> vx, "
            "-left-stick X -> vy, -right-stick X -> yaw_rate."
        )
        print(
            "[AUDIT] Safety command mode: "
            f"{'forward-only clamp' if self.forward_only else 'full omni mapping'} "
            f"with deadband={self.command_deadband:.2f}."
        )
        print("[AUDIT] Observation sources:" + "".join(obs_lines))
        print(
            "[AUDIT] Joint order remap:"
            f"\n  - hardware indices gathered into policy order: {GO2_HW_INDEX_FOR_POLICY.tolist()}"
            f"\n  - policy indices gathered into hardware order: {GO2_POLICY_INDEX_FOR_HW.tolist()}"
        )
        print(
            "[AUDIT] Action semantics:"
            f"\n  - type: joint position targets"
            f"\n  - target = offset + scale * action"
            f"\n  - offset(default pose): {self.contract.action_offset.tolist()}"
            f"\n  - scale: {self.contract.action_scale.tolist()}"
        )
        print(
            "[AUDIT] Control contract:"
            f"\n  - policy_obs_dim={self.contract.policy_obs_dim}"
            f"\n  - history_length={self.contract.policy_history_length}"
            f"\n  - history_layout={self.contract.history_layout}"
            f"\n  - action_dim={self.contract.action_dim}"
            f"\n  - step_dt={self.control_dt:.3f}s"
            f"\n  - default command={self.contract.command_default.tolist()}"
        )
        if "base_lin_vel" in self.policy_offsets:
            print(
                "[AUDIT] Safety note: this bundle requires base_lin_vel, but hardware currently zero-fills it. "
                "Treat first walking as conservative bring-up, not final sim2real validation."
            )
        else:
            print("[AUDIT] Safety note: this bundle does not require base_lin_vel in the actor observation.")

    def run_policy_loop(self, max_steps: int) -> None:
        if self.forward_only:
            print("[STATE] Running policy in forward-only safety mode. Press SELECT to stop.")
        else:
            print("[STATE] Running policy with full remote command mapping. Press SELECT to stop.")
        command = self.contract.command_default.copy()
        start_obs = self._policy_obs(command)
        self._prime_history(start_obs)
        for step_idx in range(max_steps):
            if self.remote.button[KeyMap.select] == 1:
                print("[INFO] Operator stop received.")
                break
            if self.forward_only:
                command[:] = np.array([self._apply_deadband(self.remote.ly), 0.0, 0.0], dtype=np.float32)
            else:
                command[:] = np.array(
                    [
                        self._apply_deadband(self.remote.ly),
                        -self._apply_deadband(self.remote.lx),
                        -self._apply_deadband(self.remote.rx),
                    ],
                    dtype=np.float32,
                )
            obs = self._policy_obs(command)
            self._append_history(obs)
            action = self._run_policy(obs)
            target_hw = self._target_hw(action)
            for i in range(self.contract.action_dim):
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 0x01
                motor.q = float(target_hw[i])
                motor.qd = 0.0
                motor.kp = float(self.contract.joint_stiffness[GO2_POLICY_INDEX_FOR_HW[i]])
                motor.kd = float(self.contract.joint_damping[GO2_POLICY_INDEX_FOR_HW[i]])
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
        print(
            f"[INFO] startup_posture={self.startup_posture} stance_only={self.stance_only} forward_only={self.forward_only} "
            f"command_deadband={self.command_deadband:.2f} exit_mode={self.exit_mode}"
        )
        self._print_mapping_audit()
        if self.mapping_only:
            print("[INFO] Mapping-only audit completed. Exiting before DDS or robot interaction.")
            return
        if self.dry_run:
            print("[INFO] Dry run only. Bundle contract resolved successfully.")
            return
        try:
            self._wait_for_state()
            self.zero_torque_state()
            self.move_to_default()
            if self.stance_only:
                self.hold_stance_only()
                return
            self.hold_default()
            self.run_policy_loop(max_steps=max_steps)
        except KeyboardInterrupt as exc:
            print(f"[INFO] Operator stop: {exc}")
        finally:
            self._safe_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--net-if", default="eno1")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true", help="Resolve bundle contract without touching Unitree DDS.")
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Print the hardware mapping / observation audit from the bundle contract and exit without DDS.",
    )
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
    parser.add_argument(
        "--command-deadband",
        type=float,
        default=0.05,
        help="Suppress small joystick noise around zero before mapping remote commands into policy commands.",
    )
    parser.add_argument(
        "--startup-posture",
        choices=("seated", "standing"),
        default="seated",
        help="Choose the startup assumption for the robot before low-level policy bring-up. `seated` is the safer default.",
    )
    parser.add_argument(
        "--exit-mode",
        choices=("hold", "damping", "zero-torque"),
        default="damping",
        help=(
            "How to leave the robot on abort/exit. "
            "`hold` keeps the current pose briefly, "
            "`damping` sends damping commands then releases low-level mode, "
            "`zero-torque` sends zero-torque commands then releases low-level mode."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Go2HardwareRunner(
        bundle_dir=Path(args.bundle_dir),
        net_if=args.net_if,
        dry_run=(args.dry_run or args.mapping_only),
        mapping_only=args.mapping_only,
        stance_only=args.stance_only,
        forward_only=not args.allow_lateral_yaw,
        command_deadband=args.command_deadband,
        startup_posture=args.startup_posture,
        exit_mode=args.exit_mode,
    )
    runner.run(max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
