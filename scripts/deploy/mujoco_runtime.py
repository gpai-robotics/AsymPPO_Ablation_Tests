#!/usr/bin/env python3
"""Repo-owned MuJoCo runtime bridge for deployable Go2 policies.

This module deliberately stays small and explicit:

- load the exported TorchScript deployment artifact
- load the canonical Go2 MuJoCo menagerie scene
- reconstruct the deployable observation contract:
  - policy
  - policy_history
- run a simple PD position-target loop using the same IsaacLab action semantics

It does not depend on the Unitree SDK2 bridge. That repo remains a runtime
reference only; this file is our own deployment-side bridge surface.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


GO2_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

GO2_ACTUATOR_NAMES = [
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

GO2_FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]


@dataclass
class BridgeConfig:
    model_path: Path
    policy_artifact_path: Path
    deploy_config_path: Path | None = None
    control_dt: float = 0.02
    physics_dt: float = 0.005
    history_length: int = 20
    action_scale: float = 0.25
    kp: float = 50.0
    kd: float = 3.5
    command_x: float = 0.5
    command_y: float = 0.0
    command_yaw: float = 0.0
    trace_steps: int = 25
    viewer: bool = False
    viewer_dt: float = 0.02
    real_time_factor: float = 1.0
    latent_clamp_max_abs: float = 0.0
    policy_kind: str = "blind_adaptive_student"
    ground_friction: float = 0.0
    foot_friction: float = 0.0
    base_mass_scale: float = 1.0
    motor_strength_scale: float = 1.0
    joint_damping_scale: float = 1.0
    passive_joint_damping_scale: float = 1.0
    passive_joint_frictionloss_scale: float = 1.0
    actuator_model: str = "simple_pd"
    dc_motor_velocity_limit: float = 30.0
    teleop_keyboard: bool = False
    teleop_step_x: float = 0.1
    teleop_step_y: float = 0.05
    teleop_step_yaw: float = 0.15
    teleop_limit_x: float = 1.0
    teleop_limit_y: float = 0.4
    teleop_limit_yaw: float = 1.0
    scenario_name: str = ""
    command_schedule: list[dict[str, Any]] = field(default_factory=list)
    wrench_schedule: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    reset_pos_xy_jitter: float = 0.0
    reset_yaw_jitter_deg: float = 0.0
    reset_joint_pos_jitter: float = 0.0
    reset_joint_vel_jitter: float = 0.0
    history_ablation: str = "normal"


class Go2MujocoDeployBridge:
    """Minimal MuJoCo-side execution bridge for deployable Go2 bundles."""

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "MuJoCo Python package is not available. Install it before running "
                "the runtime bridge."
            ) from exc

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "PyTorch is not available. Run the runtime bridge in an environment "
                "that can load the exported TorchScript policy."
            ) from exc

        self.mujoco = mujoco
        self.torch = torch
        self.model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.physics_dt
        self.policy = torch.jit.load(str(cfg.policy_artifact_path), map_location="cpu")
        self.policy.eval()
        self.deploy_cfg = self._load_deploy_config(cfg.deploy_config_path)

        self.joint_names = self.deploy_cfg["robot"]["joint_names"]
        self.actuator_names = self.deploy_cfg["robot"]["actuator_names"]
        self.joint_qpos_indices = np.array([self._joint_qpos_index(name) for name in self.joint_names], dtype=np.int32)
        self.joint_dof_indices = np.array([self._joint_dof_index(name) for name in self.joint_names], dtype=np.int32)
        self.actuator_indices = np.array([self._actuator_index(name) for name in self.actuator_names], dtype=np.int32)
        self.base_body_id = self._base_body_id()
        self.foot_geom_ids = {name: self._geom_index(name) for name in GO2_FOOT_GEOM_NAMES}
        self.floor_geom_ids = self._ground_geom_ids()

        self.default_joint_pos = np.asarray(self.deploy_cfg["robot"]["default_joint_pos"], dtype=np.float32)
        self.base_init_pos = np.asarray(
            self.deploy_cfg["robot"].get("base_init_pos", [0.0, 0.0, 0.4]),
            dtype=np.float32,
        )
        self.base_init_quat = np.asarray(
            self.deploy_cfg["robot"].get("base_init_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        self.joint_stiffness = np.asarray(self.deploy_cfg["robot"]["joint_stiffness"], dtype=np.float32)
        self.joint_damping = np.asarray(self.deploy_cfg["robot"]["joint_damping"], dtype=np.float32)
        self.action_scale = np.asarray(self.deploy_cfg["actions"]["scale"], dtype=np.float32)
        self.action_offset = np.asarray(self.deploy_cfg["actions"]["offset"], dtype=np.float32)
        self.actuator_mode = self._infer_actuator_mode()
        self.actuator_ctrlrange = np.asarray(self.model.actuator_ctrlrange[self.actuator_indices], dtype=np.float64)
        self.actuator_ctrl_abs_limit = np.maximum(
            np.abs(self.actuator_ctrlrange[:, 0]),
            np.abs(self.actuator_ctrlrange[:, 1]),
        ).astype(np.float32)
        self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
        default_command = self.deploy_cfg.get("commands", {}).get("base_velocity", {}).get("default", [0.5, 0.0, 0.0])
        self.command = np.array([cfg.command_x, cfg.command_y, cfg.command_yaw], dtype=np.float32)
        if np.allclose(self.command, np.array([0.5, 0.0, 0.0], dtype=np.float32)):
            self.command = np.asarray(default_command, dtype=np.float32)
        self.default_command = self.command.copy()
        self.command_schedule = sorted(
            [dict(item) for item in self.cfg.command_schedule],
            key=lambda item: int(item.get("step", 0)),
        )
        self.wrench_schedule = sorted(
            [dict(item) for item in self.cfg.wrench_schedule],
            key=lambda item: int(item.get("start_step", item.get("step", 0))),
        )
        self._next_command_schedule_idx = 0
        self.event_log: list[dict[str, Any]] = []
        self.rng = np.random.default_rng(int(self.cfg.seed))
        self.policy_history_length = int(self.deploy_cfg["observations"].get("policy_history_length", 0))
        self.policy_dim = int(self.deploy_cfg["observations"]["policy_dim"])
        self.policy_order = list(self.deploy_cfg["observations"].get("policy_order", []))
        if not self.policy_order:
            self.policy_order = [
                {"name": "base_lin_vel", "dim": 3},
                {"name": "base_ang_vel", "dim": 3},
                {"name": "projected_gravity", "dim": 3},
                {"name": "velocity_commands", "dim": 3},
                {"name": "joint_pos_rel", "dim": 12},
                {"name": "joint_vel_rel", "dim": 12},
                {"name": "last_action", "dim": 12},
            ]
        self.policy_slices = self._build_policy_slices()
        self.history = np.zeros(
            (
                self.policy_history_length,
                self.policy_dim,
            ),
            dtype=np.float32,
        )
        self.frozen_history = np.zeros_like(self.history)
        self._apply_runtime_overrides()
        self._seed_history_with_current_obs()

    def _clip_command(self) -> None:
        self.command[0] = float(np.clip(self.command[0], -self.cfg.teleop_limit_x, self.cfg.teleop_limit_x))
        self.command[1] = float(np.clip(self.command[1], -self.cfg.teleop_limit_y, self.cfg.teleop_limit_y))
        self.command[2] = float(np.clip(self.command[2], -self.cfg.teleop_limit_yaw, self.cfg.teleop_limit_yaw))

    def _print_teleop_help(self) -> None:
        print(
            "[INFO] Keyboard teleop enabled: "
            "W/S=forward, J/L=lateral, A/D=yaw, SPACE=zero lateral+yaw, R=reset default command, X=halt"
        )
        print(
            "[INFO] Initial teleop command: "
            f"vx={self.command[0]:+.2f}, vy={self.command[1]:+.2f}, yaw={self.command[2]:+.2f}"
        )

    def _maybe_print_command_update(self) -> None:
        print(
            "[INFO] Teleop command updated: "
            f"vx={self.command[0]:+.2f}, vy={self.command[1]:+.2f}, yaw={self.command[2]:+.2f}"
        )

    def _viewer_key_callback(self, keycode: int) -> None:
        glfw = self.mujoco.glfw.glfw
        updated = False

        if keycode in (glfw.KEY_W, glfw.KEY_UP):
            self.command[0] += self.cfg.teleop_step_x
            updated = True
        elif keycode in (glfw.KEY_S, glfw.KEY_DOWN):
            self.command[0] -= self.cfg.teleop_step_x
            updated = True
        elif keycode == glfw.KEY_J:
            self.command[1] += self.cfg.teleop_step_y
            updated = True
        elif keycode == glfw.KEY_L:
            self.command[1] -= self.cfg.teleop_step_y
            updated = True
        elif keycode in (glfw.KEY_A, glfw.KEY_LEFT):
            self.command[2] += self.cfg.teleop_step_yaw
            updated = True
        elif keycode in (glfw.KEY_D, glfw.KEY_RIGHT):
            self.command[2] -= self.cfg.teleop_step_yaw
            updated = True
        elif keycode == glfw.KEY_SPACE:
            self.command[1:] = 0.0
            updated = True
        elif keycode == glfw.KEY_R:
            self.command[:] = self.default_command
            updated = True
        elif keycode == glfw.KEY_X:
            self.command[:] = 0.0
            updated = True

        if updated:
            self._clip_command()
            self._maybe_print_command_update()

    def _apply_scheduled_command_updates(self, step_idx: int) -> None:
        while self._next_command_schedule_idx < len(self.command_schedule):
            item = self.command_schedule[self._next_command_schedule_idx]
            if int(item.get("step", 0)) != step_idx:
                break
            command = np.asarray(item.get("command", self.command), dtype=np.float32)
            if command.shape != (3,):
                raise RuntimeError(f"Scheduled command must be 3D, got shape {command.shape}")
            self.command[:] = command
            self._clip_command()
            self.event_log.append(
                {
                    "step": step_idx,
                    "event_type": "command_switch",
                    "label": str(item.get("label", "")),
                    "command": self.command.tolist(),
                }
            )
            self._next_command_schedule_idx += 1

    def _scheduled_wrench_for_step(self, step_idx: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        wrench = np.zeros(6, dtype=np.float32)
        started_events: list[dict[str, Any]] = []
        for item in self.wrench_schedule:
            start_step = int(item.get("start_step", item.get("step", 0)))
            duration_steps = max(1, int(item.get("duration_steps", 1)))
            if start_step <= step_idx < start_step + duration_steps:
                force = np.asarray(item.get("force_world", [0.0, 0.0, 0.0]), dtype=np.float32)
                torque = np.asarray(item.get("torque_world", [0.0, 0.0, 0.0]), dtype=np.float32)
                if force.shape != (3,) or torque.shape != (3,):
                    raise RuntimeError("Scheduled wrench events must define 3D force_world and torque_world.")
                wrench[:3] += force
                wrench[3:] += torque
                if step_idx == start_step:
                    started_events.append(
                        {
                            "step": step_idx,
                            "event_type": "wrench_start",
                            "label": str(item.get("label", "")),
                            "duration_steps": duration_steps,
                            "force_world": force.tolist(),
                            "torque_world": torque.tolist(),
                        }
                    )
        return wrench, started_events

    def _history_for_policy(self) -> np.ndarray:
        if self.cfg.history_ablation == "normal":
            return self.history
        if self.cfg.history_ablation == "zero":
            return np.zeros_like(self.history)
        if self.cfg.history_ablation == "frozen":
            return self.frozen_history
        raise RuntimeError(f"Unsupported history ablation mode: {self.cfg.history_ablation}")

    def _policy_forward_debug(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return action plus optional latent/actor-observation debug tensors.

        The exported TorchScript module is traced from a small wrapper that
        computes:

            latent = adaptation_module(policy_history)
            actor_obs = cat([policy_obs, latent])
            action = actor(actor_obs)

        We try to reproduce those intermediate tensors here for debugging. If
        the TorchScript module does not expose the expected submodules, we still
        return the action and leave the debug tensors as ``None``.
        """
        policy_obs = self.torch.from_numpy(obs).unsqueeze(0)
        history_source = self._history_for_policy()
        policy_history = self.torch.from_numpy(history_source.reshape(1, -1))
        latent_np: np.ndarray | None = None
        actor_obs_np: np.ndarray | None = None

        with self.torch.inference_mode():
            latent = None
            action = None
            adaptation_module = getattr(self.policy, "adaptation_module", None)
            actor_module = getattr(self.policy, "actor", None)
            if self.cfg.policy_kind == "blind_fixed_policy":
                action = self.policy(policy_obs)
            elif self.cfg.policy_kind == "blind_history_policy":
                action = self.policy(policy_obs, policy_history)
            elif adaptation_module is not None:
                latent = adaptation_module(policy_history)
                if self.cfg.latent_clamp_max_abs > 0.0:
                    latent = self.torch.clamp(
                        latent,
                        min=-self.cfg.latent_clamp_max_abs,
                        max=self.cfg.latent_clamp_max_abs,
                    )
                latent_np = latent.squeeze(0).cpu().numpy().astype(np.float32)
            if latent is not None:
                actor_obs = self.torch.cat([policy_obs, latent], dim=-1)
                actor_obs_np = actor_obs.squeeze(0).cpu().numpy().astype(np.float32)
                if actor_module is not None:
                    action = actor_module(actor_obs)
                else:
                    action = self.policy(policy_obs, policy_history)
            elif action is None:
                action = self.policy(policy_obs, policy_history)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if action_np.shape != (len(self.joint_names),):
            raise RuntimeError(f"Expected 12-dim action, got shape {action_np.shape}.")
        return action_np, latent_np, actor_obs_np

    def _load_deploy_config(self, deploy_config_path: Path | None) -> dict[str, Any]:
        if deploy_config_path is not None and deploy_config_path.exists():
            return json.loads(deploy_config_path.read_text())
        return {
            "robot": {
                "joint_names": GO2_JOINT_NAMES,
                "actuator_names": GO2_ACTUATOR_NAMES,
                "default_joint_pos": self.model.qpos0[:12].tolist(),
                "joint_stiffness": [self.cfg.kp] * 12,
                "joint_damping": [self.cfg.kd] * 12,
            },
            "actions": {
                "scale": [self.cfg.action_scale] * 12,
                "offset": self.model.qpos0[:12].tolist(),
            },
            "observations": {
                "policy_dim": 48,
                "policy_history_length": self.cfg.history_length,
            },
            "commands": {"base_velocity": {"default": [self.cfg.command_x, self.cfg.command_y, self.cfg.command_yaw]}},
        }

    def _build_policy_slices(self) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        offset = 0
        for term in self.policy_order:
            name = str(term["name"])
            dim = int(term["dim"])
            slices[name] = slice(offset, offset + dim)
            offset += dim
        if offset != self.policy_dim:
            raise RuntimeError(
                "Deploy observation contract is inconsistent: "
                f"policy_order dims sum to {offset}, policy_dim is {self.policy_dim}."
            )
        return slices

    def _joint_qpos_index(self, joint_name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Could not find MuJoCo joint: {joint_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dof_index(self, joint_name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Could not find MuJoCo joint: {joint_name}")
        return int(self.model.jnt_dofadr[joint_id])

    def _actuator_index(self, actuator_name: str) -> int:
        actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise RuntimeError(f"Could not find MuJoCo actuator: {actuator_name}")
        return actuator_id

    def _geom_index(self, geom_name: str) -> int:
        geom_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise RuntimeError(f"Could not find MuJoCo geom: {geom_name}")
        return geom_id

    def _base_body_id(self) -> int:
        for name in ("base", "base_link"):
            body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        raise RuntimeError("Could not find a base body named 'base' or 'base_link'.")

    def _ground_geom_ids(self) -> list[int]:
        geom_ids: list[int] = []
        for name in ("floor", "ground", "terrain"):
            geom_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                geom_ids.append(int(geom_id))
        if geom_ids:
            return geom_ids
        for geom_id in range(int(self.model.ngeom)):
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(self.mujoco.mjtGeom.mjGEOM_PLANE):
                geom_ids.append(geom_id)
        return geom_ids

    def _apply_runtime_overrides(self) -> None:
        if self.cfg.ground_friction > 0.0:
            for geom_id in self.floor_geom_ids:
                self.model.geom_friction[geom_id, 0] = float(self.cfg.ground_friction)
        if self.cfg.foot_friction > 0.0:
            for geom_id in self.foot_geom_ids.values():
                self.model.geom_friction[geom_id, 0] = float(self.cfg.foot_friction)
        if abs(self.cfg.base_mass_scale - 1.0) > 1.0e-6:
            self.model.body_mass[self.base_body_id] *= float(self.cfg.base_mass_scale)
            self.model.body_inertia[self.base_body_id] *= float(self.cfg.base_mass_scale)
        if abs(self.cfg.motor_strength_scale - 1.0) > 1.0e-6:
            self.action_scale = self.action_scale * float(self.cfg.motor_strength_scale)
        if abs(self.cfg.joint_damping_scale - 1.0) > 1.0e-6:
            self.joint_damping = self.joint_damping * float(self.cfg.joint_damping_scale)
        if abs(self.cfg.passive_joint_damping_scale - 1.0) > 1.0e-6:
            self.model.dof_damping[self.joint_dof_indices] *= float(self.cfg.passive_joint_damping_scale)
        if abs(self.cfg.passive_joint_frictionloss_scale - 1.0) > 1.0e-6:
            self.model.dof_frictionloss[self.joint_dof_indices] *= float(self.cfg.passive_joint_frictionloss_scale)

    def _model_diagnostics(self) -> dict[str, Any]:
        joint_damping = np.asarray(self.model.dof_damping[self.joint_dof_indices], dtype=np.float32)
        joint_frictionloss = np.asarray(self.model.dof_frictionloss[self.joint_dof_indices], dtype=np.float32)
        return {
            "joint_damping": joint_damping.tolist(),
            "joint_damping_mean": float(np.mean(joint_damping)),
            "joint_frictionloss": joint_frictionloss.tolist(),
            "joint_frictionloss_mean": float(np.mean(joint_frictionloss)),
            "actuator_ctrl_abs_limit": self.actuator_ctrl_abs_limit.tolist(),
            "actuator_ctrl_abs_limit_mean": float(np.mean(self.actuator_ctrl_abs_limit)),
            "actuator_ctrl_abs_limit_max": float(np.max(self.actuator_ctrl_abs_limit)),
            "actuator_model": self.cfg.actuator_model,
            "dc_motor_velocity_limit": float(self.cfg.dc_motor_velocity_limit),
        }

    def _configure_viewer_camera(self, viewer) -> None:
        cam = viewer.cam
        cam.type = self.mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.base_body_id
        cam.distance = 2.2
        cam.azimuth = 135.0
        cam.elevation = -20.0
        root_pos, _ = self._root_pose_world()
        cam.lookat[:] = root_pos

    def _update_viewer_camera(self, viewer) -> None:
        root_pos, _ = self._root_pose_world()
        viewer.cam.lookat[:] = root_pos

    def _infer_actuator_mode(self) -> str:
        biastype = np.asarray(self.model.actuator_biastype[self.actuator_indices], dtype=np.int32)
        ctrlrange = np.asarray(self.model.actuator_ctrlrange[self.actuator_indices], dtype=np.float32)
        if np.all(biastype == 1) and np.all(np.abs(ctrlrange[:, 1] - ctrlrange[:, 0]) < 10.0):
            return "position_target"
        return "torque_pd"

    def _body_velocity_local(self) -> tuple[np.ndarray, np.ndarray]:
        xmat = np.asarray(self.data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        cvel = np.asarray(self.data.cvel[self.base_body_id], dtype=np.float64)
        world_ang = cvel[:3]
        world_lin = cvel[3:]
        local_lin = xmat.T @ world_lin
        local_ang = xmat.T @ world_ang
        return local_lin.astype(np.float32), local_ang.astype(np.float32)

    def _projected_gravity(self) -> np.ndarray:
        world_gravity = np.asarray(self.model.opt.gravity, dtype=np.float64)
        gravity_norm = np.linalg.norm(world_gravity)
        if gravity_norm < 1.0e-9:
            world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            world_gravity = world_gravity / gravity_norm
        xmat = np.asarray(self.data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        return (xmat.T @ world_gravity).astype(np.float32)

    def _root_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.data.xpos[self.base_body_id], dtype=np.float32),
            np.asarray(self.data.xquat[self.base_body_id], dtype=np.float32),
        )

    def _world_linear_velocity(self) -> np.ndarray:
        cvel = np.asarray(self.data.cvel[self.base_body_id], dtype=np.float64)
        # MuJoCo spatial velocities use the ordering rot:lin in world frame.
        return cvel[3:].astype(np.float32)

    def _heading_forward_world(self) -> np.ndarray:
        xmat = np.asarray(self.data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        forward = xmat[:, 0].copy()
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1.0e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (forward / norm).astype(np.float32)

    def _foot_world_state(self) -> tuple[dict[str, list[float]], dict[str, bool]]:
        foot_pos_world = {
            name: np.asarray(self.data.geom_xpos[geom_id], dtype=np.float32).tolist()
            for name, geom_id in self.foot_geom_ids.items()
        }
        foot_contact = {name: False for name in self.foot_geom_ids}
        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            for name, geom_id in self.foot_geom_ids.items():
                if int(contact.geom1) == geom_id or int(contact.geom2) == geom_id:
                    foot_contact[name] = True
        return foot_pos_world, foot_contact

    def _policy_obs(self) -> np.ndarray:
        joint_pos = np.asarray(self.data.qpos[self.joint_qpos_indices], dtype=np.float32)
        joint_vel = np.asarray(self.data.qvel[self.joint_dof_indices], dtype=np.float32)
        base_lin_vel, base_ang_vel = self._body_velocity_local()
        term_values = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": self._projected_gravity(),
            "velocity_commands": self.command,
            "joint_pos_rel": joint_pos - self.default_joint_pos,
            "joint_vel_rel": joint_vel,
            "last_action": self.last_action,
        }
        values: list[np.ndarray] = []
        for term in self.policy_order:
            name = str(term["name"])
            if name not in term_values:
                raise RuntimeError(f"Unsupported policy observation term in deploy config: {name}")
            value = np.asarray(term_values[name], dtype=np.float32)
            expected_dim = int(term["dim"])
            if value.shape != (expected_dim,):
                raise RuntimeError(
                    f"Observation term '{name}' expected shape ({expected_dim},), got {value.shape}."
                )
            values.append(value)
        obs = np.concatenate(values, dtype=np.float32)
        if obs.shape != (self.policy_dim,):
            raise RuntimeError(f"Expected {self.policy_dim}-dim policy observation, got shape {obs.shape}.")
        return obs

    def _obs_term_or_empty(self, obs: np.ndarray, name: str) -> list[float]:
        term_slice = self.policy_slices.get(name)
        if term_slice is None:
            return []
        return obs[term_slice].tolist()

    def _obs_term_array(self, obs: np.ndarray, name: str) -> np.ndarray | None:
        term_slice = self.policy_slices.get(name)
        if term_slice is None:
            return None
        return np.asarray(obs[term_slice], dtype=np.float32)

    def _seed_history_with_current_obs(self) -> None:
        self.mujoco.mj_forward(self.model, self.data)
        obs = self._policy_obs()
        if self.policy_history_length > 0:
            self.history[:] = obs[None, :]
            self.frozen_history[:] = self.history

    def _initialize_default_pose(self) -> None:
        # Start the robot from the exported IsaacLab default joint posture so the
        # first policy/history observations match the deployment contract better.
        base_init_pos = self.base_init_pos.astype(np.float64).copy()
        if self.cfg.reset_pos_xy_jitter > 0.0:
            base_init_pos[0:2] += self.rng.uniform(
                low=-self.cfg.reset_pos_xy_jitter,
                high=self.cfg.reset_pos_xy_jitter,
                size=2,
            ).astype(np.float64)

        base_init_quat = self.base_init_quat.astype(np.float64).copy()
        if self.cfg.reset_yaw_jitter_deg > 0.0:
            yaw = np.deg2rad(
                self.rng.uniform(-self.cfg.reset_yaw_jitter_deg, self.cfg.reset_yaw_jitter_deg)
            )
            cy = float(np.cos(0.5 * yaw))
            sy = float(np.sin(0.5 * yaw))
            yaw_quat = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
            w1, x1, y1, z1 = yaw_quat
            w2, x2, y2, z2 = base_init_quat
            base_init_quat = np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dtype=np.float64,
            )
            base_init_quat = base_init_quat / max(np.linalg.norm(base_init_quat), 1.0e-9)

        if self.model.nq >= 7 and self.model.jnt_type[0] == self.mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[0:3] = base_init_pos
            self.data.qpos[3:7] = base_init_quat
        if self.model.nv >= 6 and self.model.jnt_type[0] == self.mujoco.mjtJoint.mjJNT_FREE:
            self.data.qvel[0:6] = 0.0
        joint_qpos = self.default_joint_pos.astype(np.float64).copy()
        if self.cfg.reset_joint_pos_jitter > 0.0:
            joint_qpos += self.rng.uniform(
                low=-self.cfg.reset_joint_pos_jitter,
                high=self.cfg.reset_joint_pos_jitter,
                size=joint_qpos.shape,
            ).astype(np.float64)
        self.data.qpos[self.joint_qpos_indices] = joint_qpos
        joint_qvel = np.zeros(len(self.joint_dof_indices), dtype=np.float64)
        if self.cfg.reset_joint_vel_jitter > 0.0:
            joint_qvel += self.rng.uniform(
                low=-self.cfg.reset_joint_vel_jitter,
                high=self.cfg.reset_joint_vel_jitter,
                size=joint_qvel.shape,
            ).astype(np.float64)
        self.data.qvel[self.joint_dof_indices] = joint_qvel
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _append_history(self, obs: np.ndarray) -> None:
        if self.policy_history_length <= 0:
            return
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        action_np, _, _ = self._policy_forward_debug(obs)
        return action_np

    def _clip_torque_like_isaac_dc_motor(self, torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        vel_limit = max(float(self.cfg.dc_motor_velocity_limit), 1.0e-6)
        clipped_vel = np.clip(joint_vel, -2.0 * vel_limit, 2.0 * vel_limit)
        sat = self.actuator_ctrl_abs_limit.astype(np.float32)
        max_effort = np.minimum(sat * (1.0 - clipped_vel / vel_limit), sat)
        min_effort = np.maximum(sat * (-1.0 - clipped_vel / vel_limit), -sat)
        return np.clip(torque, min_effort, max_effort).astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint_pos = np.asarray(self.data.qpos[self.joint_qpos_indices], dtype=np.float32)
        joint_vel = np.asarray(self.data.qvel[self.joint_dof_indices], dtype=np.float32)
        q_target = self.action_offset + self.action_scale * action
        ctrl = np.asarray(self.data.ctrl, dtype=np.float64)
        ctrl[:] = 0.0
        if self.actuator_mode == "position_target":
            ctrl_values = np.clip(q_target.astype(np.float64), self.actuator_ctrlrange[:, 0], self.actuator_ctrlrange[:, 1])
        else:
            torque = self.joint_stiffness * (q_target - joint_pos) + self.joint_damping * (0.0 - joint_vel)
            if self.cfg.actuator_model == "isaac_dc_motor":
                ctrl_values = self._clip_torque_like_isaac_dc_motor(torque.astype(np.float32), joint_vel).astype(
                    np.float64
                )
            else:
                ctrl_values = np.clip(
                    torque.astype(np.float64),
                    self.actuator_ctrlrange[:, 0],
                    self.actuator_ctrlrange[:, 1],
                )
        ctrl[self.actuator_indices] = ctrl_values
        self.last_action = action.copy()
        applied_ctrl = np.asarray(ctrl_values, dtype=np.float32).copy()
        return q_target.astype(np.float32), applied_ctrl

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self._initialize_default_pose()
        self.last_action[:] = 0.0
        self.command[:] = self.default_command
        self._next_command_schedule_idx = 0
        self.event_log = []
        self._seed_history_with_current_obs()

    def _run_loop(self, num_steps: int, viewer=None) -> dict[str, Any]:
        reward_proxy = []
        vel_err = []
        yaw_err = []
        base_height = []
        base_tilt = []
        action_abs = []
        action_delta = []
        joint_vel_abs = []
        q_target_joint_err = []
        ctrl_abs = []
        ctrl_saturation_frac = []
        latent_norm = []
        latent_max_abs = []
        actor_obs_norm = []
        foot_contact_counts = {name: 0 for name in self.foot_geom_ids}
        foot_height_means = {name: [] for name in self.foot_geom_ids}
        trace = []
        first_event_step: int | None = None
        next_viewer_sync_time = time.perf_counter()
        substeps = max(1, int(round(self.cfg.control_dt / self.cfg.physics_dt)))
        wall_step_start = time.perf_counter()
        prev_action = None

        for step_idx in range(num_steps):
            prev_event_count = len(self.event_log)
            self._apply_scheduled_command_updates(step_idx)
            if len(self.event_log) > prev_event_count and first_event_step is None:
                first_event_step = step_idx
            scheduled_wrench, started_events = self._scheduled_wrench_for_step(step_idx)
            if started_events:
                self.event_log.extend(started_events)
                if first_event_step is None:
                    first_event_step = step_idx
            obs = self._policy_obs()
            self._append_history(obs)
            action, latent, actor_obs = self._policy_forward_debug(obs)
            q_target, applied_ctrl = self._apply_action(action)

            for _ in range(substeps):
                self.data.xfrc_applied[self.base_body_id, :] = scheduled_wrench.astype(np.float64)
                self.mujoco.mj_step(self.model, self.data)
                if viewer is not None and not viewer.is_running():
                    num_steps = step_idx + 1
                    break
            self.data.xfrc_applied[self.base_body_id, :] = 0.0

            if viewer is not None:
                now = time.perf_counter()
                if now >= next_viewer_sync_time:
                    self._update_viewer_camera(viewer)
                    viewer.sync()
                    next_viewer_sync_time = now + self.cfg.viewer_dt
                if not viewer.is_running():
                    break
                expected_elapsed = ((step_idx + 1) * self.cfg.control_dt) / max(self.cfg.real_time_factor, 1.0e-6)
                actual_elapsed = time.perf_counter() - wall_step_start
                sleep_time = expected_elapsed - actual_elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            next_obs = self._policy_obs()
            local_lin_vel, local_ang_vel = self._body_velocity_local()
            base_lin_vel_obs = self._obs_term_array(next_obs, "base_lin_vel")
            base_ang_vel_obs = self._obs_term_array(next_obs, "base_ang_vel")
            gravity_obs = self._obs_term_array(next_obs, "projected_gravity")
            planar_vel = (base_lin_vel_obs[:2] if base_lin_vel_obs is not None else local_lin_vel[:2])
            yaw_vel = float(base_ang_vel_obs[2] if base_ang_vel_obs is not None else local_ang_vel[2])
            gravity_proj = gravity_obs if gravity_obs is not None else self._projected_gravity()
            joint_pos = np.asarray(self.data.qpos[self.joint_qpos_indices], dtype=np.float32)
            joint_vel = np.asarray(self.data.qvel[self.joint_dof_indices], dtype=np.float32)
            root_height = float(self.data.xpos[self.base_body_id, 2])
            vel_err_step = float(np.linalg.norm(planar_vel - self.command[:2]))
            yaw_err_step = float(abs(yaw_vel - self.command[2]))
            base_tilt_step = float(np.linalg.norm(gravity_proj[:2]))

            vel_err.append(vel_err_step)
            yaw_err.append(yaw_err_step)
            base_height.append(root_height)
            base_tilt.append(base_tilt_step)
            action_abs.append(float(np.mean(np.abs(action))))
            joint_vel_abs.append(float(np.mean(np.abs(joint_vel))))
            q_target_joint_err.append(float(np.mean(np.abs(q_target - joint_pos))))
            ctrl_abs.append(float(np.mean(np.abs(applied_ctrl))))
            ctrl_saturation_frac.append(
                float(np.mean(np.abs(applied_ctrl) >= (self.actuator_ctrl_abs_limit - 1.0e-4)))
            )
            if prev_action is None:
                action_delta.append(0.0)
            else:
                action_delta.append(float(np.mean(np.abs(action - prev_action))))
            prev_action = action.copy()
            reward_proxy.append(float(np.linalg.norm(planar_vel)))
            if latent is not None:
                latent_norm.append(float(np.linalg.norm(latent)))
                latent_max_abs.append(float(np.max(np.abs(latent))))
            if actor_obs is not None:
                actor_obs_norm.append(float(np.linalg.norm(actor_obs)))

            foot_pos_world, foot_contact = self._foot_world_state()
            for name, pos in foot_pos_world.items():
                foot_height_means[name].append(float(pos[2]))
                if foot_contact[name]:
                    foot_contact_counts[name] += 1

            if self.cfg.trace_steps < 0 or step_idx < self.cfg.trace_steps:
                root_pos_w, root_quat_w = self._root_pose_world()
                world_lin_vel = self._world_linear_velocity()
                heading_forward_w = self._heading_forward_world()
                heading_speed = float(np.dot(world_lin_vel[:2], heading_forward_w[:2]))
                trace.append(
                    {
                        "step": step_idx,
                        "command": self.command.tolist(),
                        "scheduled_wrench_world": scheduled_wrench.tolist(),
                        "root_pos_world": root_pos_w.tolist(),
                        "root_quat_world_wxyz": root_quat_w.tolist(),
                        "world_lin_vel": world_lin_vel.tolist(),
                        "heading_forward_world_xy": heading_forward_w[:2].tolist(),
                        "heading_speed_world_xy": heading_speed,
                        "base_lin_vel_local": self._obs_term_or_empty(next_obs, "base_lin_vel"),
                        "base_ang_vel_local": self._obs_term_or_empty(next_obs, "base_ang_vel"),
                        "projected_gravity": self._obs_term_or_empty(next_obs, "projected_gravity"),
                        "root_height": root_height,
                        "base_tilt_xy_norm": base_tilt_step,
                        "vel_err": vel_err_step,
                        "yaw_err": yaw_err_step,
                        "joint_pos": joint_pos.tolist(),
                        "joint_pos_rel": (joint_pos - self.default_joint_pos).tolist(),
                        "joint_vel": joint_vel.tolist(),
                        "last_action_used_in_obs": self._obs_term_or_empty(next_obs, "last_action"),
                        "policy_obs_head": obs[:12].tolist(),
                        "history_tail_head": self.history[-1, :12].tolist() if self.history.shape[0] > 0 else [],
                        "foot_pos_world": foot_pos_world,
                        "foot_contact": foot_contact,
                        "action": action.tolist(),
                        "action_abs_mean": float(np.mean(np.abs(action))),
                        "action_delta_mean": float(action_delta[-1]),
                        "latent": latent.tolist() if latent is not None else None,
                        "latent_norm": float(np.linalg.norm(latent)) if latent is not None else None,
                        "latent_max_abs": float(np.max(np.abs(latent))) if latent is not None else None,
                        "actor_obs_norm": float(np.linalg.norm(actor_obs)) if actor_obs is not None else None,
                        "q_target": q_target.tolist(),
                        "q_target_joint_err_mean": float(q_target_joint_err[-1]),
                        "applied_ctrl": applied_ctrl.tolist(),
                        "ctrl_abs_mean": float(ctrl_abs[-1]),
                        "ctrl_saturation_frac": float(ctrl_saturation_frac[-1]),
                        "joint_vel_abs_mean": float(joint_vel_abs[-1]),
                    }
                )

        def _mean_or_zero(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        post_event_slice = slice(first_event_step, None) if first_event_step is not None else slice(0, 0)
        post_event_summary = {
            "reward_proxy_mean": _mean_or_zero(reward_proxy[post_event_slice]),
            "vel_err_step_mean": _mean_or_zero(vel_err[post_event_slice]),
            "yaw_err_step_mean": _mean_or_zero(yaw_err[post_event_slice]),
            "base_height_mean": _mean_or_zero(base_height[post_event_slice]),
            "base_tilt_projected_gravity_xy_mean": _mean_or_zero(base_tilt[post_event_slice]),
        } if first_event_step is not None else {}

        return {
            "status": "completed_runtime_rehearsal",
            "num_steps": num_steps,
            "control_dt": self.cfg.control_dt,
            "physics_dt": self.cfg.physics_dt,
            "substeps_per_control": substeps,
            "command": self.command.tolist(),
            "default_joint_pos": self.default_joint_pos.tolist(),
            "actuator_mode": self.actuator_mode,
            "runtime_overrides": {
                "ground_friction": float(self.cfg.ground_friction),
                "foot_friction": float(self.cfg.foot_friction),
                "base_mass_scale": float(self.cfg.base_mass_scale),
                "motor_strength_scale": float(self.cfg.motor_strength_scale),
                "joint_damping_scale": float(self.cfg.joint_damping_scale),
                "passive_joint_damping_scale": float(self.cfg.passive_joint_damping_scale),
                "passive_joint_frictionloss_scale": float(self.cfg.passive_joint_frictionloss_scale),
                "actuator_model": self.cfg.actuator_model,
                "dc_motor_velocity_limit": float(self.cfg.dc_motor_velocity_limit),
                "model_path": str(self.cfg.model_path),
                "teleop_keyboard": bool(self.cfg.teleop_keyboard),
                "scenario_name": self.cfg.scenario_name,
                "seed": int(self.cfg.seed),
                "reset_pos_xy_jitter": float(self.cfg.reset_pos_xy_jitter),
                "reset_yaw_jitter_deg": float(self.cfg.reset_yaw_jitter_deg),
                "reset_joint_pos_jitter": float(self.cfg.reset_joint_pos_jitter),
                "reset_joint_vel_jitter": float(self.cfg.reset_joint_vel_jitter),
                "history_ablation": self.cfg.history_ablation,
            },
            "event_log": self.event_log,
            "first_event_step": first_event_step,
            "post_event_summary": post_event_summary,
            "model_diagnostics": self._model_diagnostics(),
            "summary_metrics": {
                "metric_contract_version": "mujoco_runtime_named_obs_v2",
                "metric_source": {
                    "planar_velocity": "base_lin_vel_obs" if base_lin_vel_obs is not None else "body_velocity_local",
                    "yaw_velocity": "base_ang_vel_obs" if base_ang_vel_obs is not None else "body_velocity_local",
                    "projected_gravity": "projected_gravity_obs" if gravity_obs is not None else "body_orientation",
                },
                "reward_proxy_mean": float(np.mean(reward_proxy)) if reward_proxy else 0.0,
                "vel_err_step_mean": float(np.mean(vel_err)) if vel_err else 0.0,
                "yaw_err_step_mean": float(np.mean(yaw_err)) if yaw_err else 0.0,
                "base_height_mean": float(np.mean(base_height)) if base_height else 0.0,
                "base_tilt_projected_gravity_xy_mean": float(np.mean(base_tilt)) if base_tilt else 0.0,
                "action_abs_mean": float(np.mean(action_abs)) if action_abs else 0.0,
                "action_delta_mean": float(np.mean(action_delta)) if action_delta else 0.0,
                "joint_vel_abs_mean": float(np.mean(joint_vel_abs)) if joint_vel_abs else 0.0,
                "q_target_joint_err_mean": float(np.mean(q_target_joint_err)) if q_target_joint_err else 0.0,
                "ctrl_abs_mean": float(np.mean(ctrl_abs)) if ctrl_abs else 0.0,
                "ctrl_saturation_frac_mean": float(np.mean(ctrl_saturation_frac)) if ctrl_saturation_frac else 0.0,
                "latent_norm_mean": float(np.mean(latent_norm)) if latent_norm else 0.0,
                "latent_norm_max": float(np.max(latent_norm)) if latent_norm else 0.0,
                "latent_max_abs_mean": float(np.mean(latent_max_abs)) if latent_max_abs else 0.0,
                "latent_max_abs_max": float(np.max(latent_max_abs)) if latent_max_abs else 0.0,
                "actor_obs_norm_mean": float(np.mean(actor_obs_norm)) if actor_obs_norm else 0.0,
                "actor_obs_norm_max": float(np.max(actor_obs_norm)) if actor_obs_norm else 0.0,
                "foot_contact_fraction": {
                    name: float(count / max(num_steps, 1)) for name, count in foot_contact_counts.items()
                },
                "foot_height_mean": {
                    name: float(np.mean(values)) if values else 0.0 for name, values in foot_height_means.items()
                },
            },
            "latent_clamp_max_abs": float(self.cfg.latent_clamp_max_abs),
            "trace_steps_captured": len(trace),
            "trace": trace,
        }

    def run(self, num_steps: int) -> dict[str, Any]:
        self.reset()
        if self.cfg.viewer:
            import mujoco.viewer

            key_callback = self._viewer_key_callback if self.cfg.teleop_keyboard else None
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
                self._configure_viewer_camera(viewer)
                if self.cfg.teleop_keyboard:
                    self._print_teleop_help()
                return self._run_loop(num_steps, viewer=viewer)
        return self._run_loop(num_steps, viewer=None)
