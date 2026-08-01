#!/usr/bin/env python3
"""Evaluate mirrored Go2 leg asymmetry scenarios in Isaac Sim.

The scenario commands and 50 N lateral pushes match the MuJoCo
``mujoco_fr_asymmetry_v1`` suite. All dynamics randomization is collapsed to
nominal values so this test measures the policy and Isaac physics, not a
randomized evaluation distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from statistics import mean

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy"))
from crosssim_contract import (  # noqa: E402
    DEFAULT_PROFILE,
    load_profile,
    matched_runtime_contract,
    matched_scenario_dicts,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
parser.add_argument("--task", default="Go2-Blind-Rough-MJLAB-AsymPPO-V1")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--rollouts-per-scenario", type=int)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--push-start-step", type=int)
parser.add_argument("--push-duration-steps", type=int)
parser.add_argument("--push-force-n", type=float)
parser.add_argument("--obs-delay-steps", type=int)
parser.add_argument("--action-delay-steps", type=int)
parser.add_argument("--command-delay-steps", type=int)
parser.add_argument(
    "--real-time",
    action="store_true",
    help="Pace the finite evaluation at the environment control rate for visual inspection.",
)
parser.add_argument(
    "--reset-preset",
    choices=("none", "light"),
    default=None,
    help="Use the same light reset perturbation envelope as the MuJoCo asymmetry suite.",
)
parser.add_argument(
    "--json-out",
    type=Path,
    default=Path("artifacts/diagnostics/isaac_leg_asymmetry_v1.json"),
)
parser.add_argument(
    "--print-json",
    action="store_true",
    help="Print the complete JSON report in addition to writing it to disk.",
)
parser.add_argument("--progress-every", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

PROFILE_PATH, PROFILE = load_profile(args_cli.profile)
PROFILE_CONTRACT = matched_runtime_contract(PROFILE_PATH, PROFILE)
PROFILE_RESET = PROFILE["matched_reset"]
PROFILE_GAP = PROFILE["matched_hardware_gap"]
PROFILE_SCENARIOS = PROFILE["matched_scenarios"]
PROFILE_PUSH = PROFILE_SCENARIOS["push"]
args_cli.rollouts_per_scenario = args_cli.rollouts_per_scenario or int(PROFILE_RESET["rollouts_per_scenario"])
args_cli.max_steps = args_cli.max_steps or int(PROFILE_SCENARIOS["max_steps"])
args_cli.warmup_steps = args_cli.warmup_steps or int(PROFILE_SCENARIOS["warmup_steps"])
args_cli.seed = args_cli.seed if args_cli.seed is not None else int(PROFILE_RESET["seed_base"])
args_cli.push_start_step = args_cli.push_start_step if args_cli.push_start_step is not None else int(PROFILE_PUSH["start_control_step"])
args_cli.push_duration_steps = (
    args_cli.push_duration_steps
    if args_cli.push_duration_steps is not None
    else int(PROFILE_PUSH["duration_control_steps"])
)
args_cli.push_force_n = args_cli.push_force_n if args_cli.push_force_n is not None else float(PROFILE_PUSH["left_force_n"][1])
args_cli.obs_delay_steps = (
    args_cli.obs_delay_steps
    if args_cli.obs_delay_steps is not None
    else int(PROFILE_GAP["observation_delay_control_steps"])
)
args_cli.action_delay_steps = (
    args_cli.action_delay_steps
    if args_cli.action_delay_steps is not None
    else int(PROFILE_GAP["action_delay_control_steps"])
)
args_cli.command_delay_steps = (
    args_cli.command_delay_steps
    if args_cli.command_delay_steps is not None
    else int(PROFILE_GAP["command_delay_control_steps"])
)
args_cli.reset_preset = args_cli.reset_preset or str(PROFILE_RESET["preset"])

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import mdp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


SCENARIOS = tuple(
    (
        item["name"],
        tuple(item["command"]),
        1.0 if item["name"] == "asym_push_left" else -1.0 if item["name"] == "asym_push_right" else 0.0,
    )
    for item in matched_scenario_dicts(PROFILE)
)

JOINT_PAIRS = (
    ("front_hip", "FR_hip_joint", "FL_hip_joint"),
    ("front_thigh", "FR_thigh_joint", "FL_thigh_joint"),
    ("front_calf", "FR_calf_joint", "FL_calf_joint"),
    ("rear_hip", "RL_hip_joint", "RR_hip_joint"),
    ("rear_thigh", "RL_thigh_joint", "RR_thigh_joint"),
    ("rear_calf", "RL_calf_joint", "RR_calf_joint"),
)

MIRRORED_SCENARIOS = (
    ("lateral", "asym_lateral_left", "asym_lateral_right"),
    ("yaw", "asym_yaw_left", "asym_yaw_right"),
    ("push", "asym_push_left", "asym_push_right"),
)

METRICS = (
    "action_abs_mean",
    "q_target_err_abs_mean",
    "ctrl_abs_mean",
    "joint_vel_abs_mean",
)


def _set_param(term, key: str, value) -> None:
    if term is not None and key in term.params:
        term.params[key] = value


def _apply_nominal_flat_overrides(env_cfg) -> None:
    nominal = PROFILE["nominal_robot"]
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    # Keep the scanner on the plane. This task inherits a terrain-privileged
    # observation group, and Isaac constructs all configured groups even
    # though only the blind policy observation is passed to the actor.

    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is not None:
        for name in ("terrain_levels", "lin_vel_command_levels", "ang_vel_command_levels"):
            if hasattr(curriculum, name):
                setattr(curriculum, name, None)

    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    if hasattr(events, "push_robot"):
        events.push_robot = None

    physics_material = getattr(events, "physics_material", None)
    _set_param(physics_material, "static_friction_range", (nominal["ground_static_friction"],) * 2)
    _set_param(physics_material, "dynamic_friction_range", (nominal["ground_dynamic_friction"],) * 2)
    _set_param(physics_material, "restitution_range", (nominal["restitution"],) * 2)

    base_wrench = getattr(events, "base_external_force_torque", None)
    _set_param(base_wrench, "force_range", (0.0, 0.0))
    _set_param(base_wrench, "torque_range", (0.0, 0.0))

    base_mass = getattr(events, "add_base_mass", None)
    _set_param(base_mass, "mass_distribution_params", (0.0, 0.0))

    base_com = getattr(events, "base_com", None)
    _set_param(base_com, "com_range", {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)})

    for name in ("motor_strength", "motor_strength_hip_thigh", "motor_strength_calf"):
        gains = getattr(events, name, None)
        _set_param(gains, "stiffness_distribution_params", (1.0, 1.0))
        _set_param(gains, "damping_distribution_params", (1.0, 1.0))

    if args_cli.reset_preset == "light":
        xy_min, xy_max = PROFILE_RESET["base_xy_uniform_m"]
        yaw_min, yaw_max = PROFILE_RESET["base_yaw_uniform_deg"]
        joint_min, joint_max = PROFILE_RESET["joint_position_uniform_rad"]
        joint_vel_min, joint_vel_max = PROFILE_RESET["joint_velocity_uniform_rad_s"]
        pose_range = {
            "x": (xy_min, xy_max),
            "y": (xy_min, xy_max),
            "yaw": (yaw_min * 0.017453292519943295, yaw_max * 0.017453292519943295),
        }
        joint_position_range = (joint_min, joint_max)
        joint_velocity_range = (joint_vel_min, joint_vel_max)
    else:
        pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        joint_position_range = (0.0, 0.0)
        joint_velocity_range = (0.0, 0.0)

    reset_base = getattr(events, "reset_base", None)
    _set_param(reset_base, "pose_range", pose_range)
    _set_param(
        reset_base,
        "velocity_range",
        {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    )
    reset_joints = getattr(events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.func = mdp.reset_joints_by_offset
    _set_param(reset_joints, "position_range", joint_position_range)
    _set_param(reset_joints, "velocity_range", joint_velocity_range)


def _make_runner(env, runner_cfg):
    if runner_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    if runner_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    raise RuntimeError(f"Unsupported runner class: {runner_cfg.class_name}")


def _safe_load_runner(runner, checkpoint_path: str) -> None:
    try:
        runner.load(checkpoint_path)
        return
    except RuntimeError as exc:
        if "normalizer" not in str(exc):
            raise
    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    state = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if "normalizer" not in key
    }
    runner.alg.policy.load_state_dict(state, strict=False)


def _unwrap_obs(obs):
    return obs[0] if isinstance(obs, tuple) else obs


def _clone_obs(obs):
    return obs.clone()


def _tensor_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _first_env_tensor(data, name: str):
    value = getattr(data, name, None)
    if value is None:
        return None
    return _tensor_list(value[0])


def _actuator_snapshot(robot) -> dict[str, object]:
    snapshots = {}
    for name, actuator in robot.actuators.items():
        values = {}
        for field in (
            "stiffness",
            "damping",
            "effort_limit",
            "effort_limit_sim",
            "velocity_limit",
            "velocity_limit_sim",
            "_saturation_effort",
        ):
            value = getattr(actuator, field, None)
            if value is None:
                continue
            if hasattr(value, "ndim") and value.ndim > 1:
                value = value[0]
            values[field.removeprefix("_")] = _tensor_list(value)
        snapshots[name] = {
            "class": type(actuator).__name__,
            **values,
        }
    return snapshots


def _runtime_contract_snapshot(env_cfg, robot, action_term) -> dict[str, object]:
    data = robot.data
    physx = env_cfg.sim.physx
    spawn = env_cfg.scene.robot.spawn
    return {
        "timing": {
            "physics_dt_s": float(env_cfg.sim.dt),
            "control_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
            "decimation": int(env_cfg.decimation),
            "control_rate_hz": float(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
        },
        "asset": {
            "usd_path": str(getattr(spawn, "usd_path", "")),
            "body_names": list(robot.body_names),
            "joint_names": list(robot.joint_names),
        },
        "articulation": {
            "default_body_mass_kg": _first_env_tensor(data, "default_mass"),
            "default_body_inertia": _first_env_tensor(data, "default_inertia"),
            "default_joint_pos_rad": _tensor_list(data.default_joint_pos[0]),
            "default_joint_stiffness_nm_rad": _tensor_list(data.default_joint_stiffness[0]),
            "default_joint_damping_nm_s_rad": _tensor_list(data.default_joint_damping[0]),
            "soft_joint_pos_limits_rad": _tensor_list(data.soft_joint_pos_limits[0]),
            "soft_joint_vel_limits_rad_s": _tensor_list(data.soft_joint_vel_limits[0]),
            "action_joint_ids": _resolved_joint_ids(action_term, robot.num_joints),
            "action_scale": _tensor_list(action_term._scale),
            "action_offset": _tensor_list(action_term._offset),
        },
        "actuators": _actuator_snapshot(robot),
        "physx": {
            "solver_type_enum": int(physx.solver_type),
            "solver_type_name": "TGS" if int(physx.solver_type) == 1 else "PGS",
            "articulation_position_iterations": int(
                env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count
            ),
            "articulation_velocity_iterations": int(
                env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count
            ),
            "enable_ccd": bool(getattr(physx, "enable_ccd", False)),
            "enable_stabilization": bool(getattr(physx, "enable_stabilization", False)),
            "enable_external_forces_every_iteration": bool(
                physx.enable_external_forces_every_iteration
            ),
            "bounce_threshold_velocity": float(physx.bounce_threshold_velocity),
            "friction_correlation_distance": float(physx.friction_correlation_distance),
            "friction_offset_threshold": float(physx.friction_offset_threshold),
            "gpu_max_rigid_contact_count": int(physx.gpu_max_rigid_contact_count),
            "gpu_max_rigid_patch_count": int(physx.gpu_max_rigid_patch_count),
        },
    }


def _resolved_joint_ids(action_term, num_joints: int) -> list[int]:
    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        return list(range(num_joints))[joint_ids]
    if isinstance(joint_ids, torch.Tensor):
        return joint_ids.detach().cpu().tolist()
    return list(joint_ids)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1.0e-9:
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _force_commands(env, commands: torch.Tensor) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = commands
    term.time_left[:] = 1.0e6
    term.command_counter[:] = 1
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False


def _scenario_layout(device: str, dtype: torch.dtype) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    names: list[str] = []
    commands = []
    push_signs = []
    for scenario_name, command, push_sign in SCENARIOS:
        for _ in range(args_cli.rollouts_per_scenario):
            names.append(scenario_name)
            commands.append(command)
            push_signs.append(push_sign)
    return (
        names,
        torch.tensor(commands, device=device, dtype=dtype),
        torch.tensor(push_signs, device=device, dtype=dtype),
    )


def _metric_store(joint_names: list[str], termination_names: list[str]) -> dict[str, object]:
    return {
        "sample_count": 0,
        "reset_count": 0,
        "terminated_count": 0,
        "truncated_count": 0,
        "termination_counts": {name: 0 for name in termination_names},
        "base_height": [],
        "projected_gravity_x": [],
        "projected_gravity_y": [],
        "joint_values": {
            name: {metric: [] for metric in METRICS}
            for name in joint_names
        },
    }


def _finalize_scenario(store: dict[str, object]) -> dict[str, object]:
    joint_means = {
        joint_name: {
            metric: _mean(values)
            for metric, values in metrics.items()
        }
        for joint_name, metrics in store["joint_values"].items()
    }
    pair_comparisons = {}
    for label, numerator_name, denominator_name in JOINT_PAIRS:
        numerator = joint_means[numerator_name]
        denominator = joint_means[denominator_name]
        pair_comparisons[label] = {
            "numerator_joint": numerator_name,
            "denominator_joint": denominator_name,
            **{
                f"{metric}_ratio": _ratio(numerator[metric], denominator[metric])
                for metric in METRICS
            },
        }
    return {
        "sample_count": store["sample_count"],
        "reset_count": store["reset_count"],
        "terminated_count": store["terminated_count"],
        "truncated_count": store["truncated_count"],
        "termination_counts": store["termination_counts"],
        "base_height_mean": _mean(store["base_height"]),
        "projected_gravity_x_mean": _mean(store["projected_gravity_x"]),
        "projected_gravity_y_mean": _mean(store["projected_gravity_y"]),
        "joint_means": joint_means,
        "pair_comparisons": pair_comparisons,
    }


def main() -> int:
    if args_cli.rollouts_per_scenario < 1:
        raise ValueError("--rollouts-per-scenario must be at least 1")
    if args_cli.warmup_steps >= args_cli.max_steps:
        raise ValueError("--warmup-steps must be smaller than --max-steps")

    num_envs = len(SCENARIOS) * args_cli.rollouts_per_scenario
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    _apply_nominal_flat_overrides(env_cfg)

    raw_env = None
    env = None
    try:
        print("[INFO] Creating Isaac environment...", flush=True)
        raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        print("[INFO] Isaac environment created; wrapping for RSL-RL...", flush=True)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=runner_cfg.clip_actions)
        print("[INFO] RSL-RL wrapper ready; constructing runner...", flush=True)
        runner = _make_runner(env, runner_cfg)
        print("[INFO] Runner ready; loading checkpoint...", flush=True)
        _safe_load_runner(runner, args_cli.checkpoint)
        print("[INFO] Checkpoint loaded; preparing evaluation...", flush=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print("[INFO] Inference policy ready.", flush=True)
        policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
        if policy_nn is None:
            raise RuntimeError("Could not locate the loaded policy module.")

        robot = env.unwrapped.scene["robot"]
        print("[INFO] Robot articulation resolved.", flush=True)
        scenario_names, commands, push_signs = _scenario_layout(
            robot.device,
            robot.data.joint_pos.dtype,
        )
        print("[INFO] Scenario tensors created.", flush=True)
        scenario_env_ids = {
            name: torch.tensor(
                [idx for idx, env_name in enumerate(scenario_names) if env_name == name],
                device=robot.device,
                dtype=torch.long,
            )
            for name, _, _ in SCENARIOS
        }

        action_term_name = env.unwrapped.action_manager.active_terms[0]
        action_term = env.unwrapped.action_manager.get_term(action_term_name)
        print(f"[INFO] Action term resolved: {action_term_name}.", flush=True)
        action_joint_ids = _resolved_joint_ids(action_term, robot.num_joints)
        action_index_by_joint = {
            robot.joint_names[robot_joint_id]: action_idx
            for action_idx, robot_joint_id in enumerate(action_joint_ids)
        }
        robot_index_by_joint = {
            name: idx
            for idx, name in enumerate(robot.joint_names)
        }
        expected_joint_names = [name for _, fr_name, fl_name in JOINT_PAIRS for name in (fr_name, fl_name)]
        missing = [
            name
            for name in expected_joint_names
            if name not in action_index_by_joint or name not in robot_index_by_joint
        ]
        if missing:
            raise RuntimeError(f"Required joints are missing from the action or robot ordering: {missing}")

        base_body_ids, _ = robot.find_bodies("base")
        if len(base_body_ids) != 1:
            raise RuntimeError(f"Expected one base body, found {base_body_ids}")
        base_body_id = int(base_body_ids[0])
        zero_torques = torch.zeros((num_envs, 1, 3), device=robot.device)

        _force_commands(env, commands)
        print("[INFO] Fixed scenario commands installed.", flush=True)
        obs = _unwrap_obs(env.get_observations())
        print("[INFO] Initial observations acquired.", flush=True)
        obs_delay_buffer = [
            _clone_obs(obs)
            for _ in range(max(0, args_cli.obs_delay_steps))
        ]
        action_delay_buffer = [
            torch.zeros((num_envs, len(action_joint_ids)), device=robot.device)
            for _ in range(max(0, args_cli.action_delay_steps))
        ]

        termination_names = list(env.unwrapped.termination_manager.active_terms)
        stores = {
            name: _metric_store(expected_joint_names, termination_names)
            for name, _, _ in SCENARIOS
        }

        print(
            f"[INFO] Isaac leg asymmetry: {len(SCENARIOS)} scenarios, "
            f"{args_cli.rollouts_per_scenario} replicas each, {num_envs} envs total."
        )
        print(
            "[INFO] Nominal flat dynamics; command randomization disabled; "
            f"reset_preset={args_cli.reset_preset}; real_time={args_cli.real_time}.",
            flush=True,
        )
        for scenario_name, _, _ in SCENARIOS:
            env_ids = scenario_env_ids[scenario_name]
            print(
                f"[INFO] {scenario_name}: envs "
                f"{int(env_ids[0].item())}-{int(env_ids[-1].item())}"
            )

        with torch.inference_mode():
            for step_idx in range(args_cli.max_steps):
                step_start = time.perf_counter()
                _force_commands(env, commands)

                push_active = (
                    args_cli.push_start_step
                    <= step_idx
                    < args_cli.push_start_step + args_cli.push_duration_steps
                )
                if push_active:
                    forces = torch.zeros((num_envs, 1, 3), device=robot.device)
                    forces[:, 0, 1] = push_signs * args_cli.push_force_n
                    robot.permanent_wrench_composer.set_forces_and_torques(
                        forces=forces,
                        torques=zero_torques,
                        body_ids=[base_body_id],
                        is_global=True,
                    )

                if obs_delay_buffer:
                    policy_obs = obs_delay_buffer.pop(0)
                    obs_delay_buffer.append(_clone_obs(obs))
                else:
                    policy_obs = obs
                raw_actions = policy(policy_obs)
                if action_delay_buffer:
                    actions = action_delay_buffer.pop(0)
                    action_delay_buffer.append(raw_actions.clone())
                else:
                    actions = raw_actions

                step_out = env.step(actions)
                if len(step_out) == 5:
                    obs, _, terminated, truncated, _ = step_out
                    dones = terminated | truncated
                else:
                    obs, _, dones, _ = step_out
                    terminated = dones
                    truncated = torch.zeros_like(dones)
                obs = _unwrap_obs(obs)
                termination_fired = {
                    name: env.unwrapped.termination_manager.get_term(name).clone()
                    for name in termination_names
                }
                if push_active:
                    robot.permanent_wrench_composer.reset()
                _force_commands(env, commands)

                if step_idx >= args_cli.warmup_steps:
                    projected_gravity = robot.data.projected_gravity_b
                    q_target = action_term.processed_actions
                    joint_pos = robot.data.joint_pos
                    joint_vel = robot.data.joint_vel
                    applied_torque = robot.data.applied_torque

                    for scenario_name, _, _ in SCENARIOS:
                        env_ids = scenario_env_ids[scenario_name]
                        store = stores[scenario_name]
                        store["sample_count"] += int(env_ids.numel())
                        store["reset_count"] += int(dones[env_ids].sum().item())
                        store["terminated_count"] += int(terminated[env_ids].sum().item())
                        store["truncated_count"] += int(truncated[env_ids].sum().item())
                        for term_name, fired in termination_fired.items():
                            store["termination_counts"][term_name] += int(fired[env_ids].sum().item())
                        store["base_height"].extend(robot.data.root_pos_w[env_ids, 2].cpu().tolist())
                        store["projected_gravity_x"].extend(projected_gravity[env_ids, 0].cpu().tolist())
                        store["projected_gravity_y"].extend(projected_gravity[env_ids, 1].cpu().tolist())

                        for joint_name in expected_joint_names:
                            action_idx = action_index_by_joint[joint_name]
                            robot_idx = robot_index_by_joint[joint_name]
                            target_error = q_target[env_ids, action_idx] - joint_pos[env_ids, robot_idx]
                            values = store["joint_values"][joint_name]
                            values["action_abs_mean"].extend(actions[env_ids, action_idx].abs().cpu().tolist())
                            values["q_target_err_abs_mean"].extend(target_error.abs().cpu().tolist())
                            values["ctrl_abs_mean"].extend(applied_torque[env_ids, robot_idx].abs().cpu().tolist())
                            values["joint_vel_abs_mean"].extend(joint_vel[env_ids, robot_idx].abs().cpu().tolist())

                if bool(dones.any().item()):
                    policy_nn.reset(dones)

                if args_cli.progress_every > 0 and (
                    (step_idx + 1) % args_cli.progress_every == 0
                    or step_idx + 1 == args_cli.max_steps
                ):
                    reset_total = sum(int(store["reset_count"]) for store in stores.values())
                    print(
                        f"[INFO] step={step_idx + 1}/{args_cli.max_steps} "
                        f"push_active={push_active} reset_count={reset_total}",
                        flush=True,
                    )

                if args_cli.real_time:
                    sleep_time = env.unwrapped.step_dt - (time.perf_counter() - step_start)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

        scenarios = {
            name: _finalize_scenario(stores[name])
            for name, _, _ in SCENARIOS
        }
        mirrored = {}
        for label, positive_name, negative_name in MIRRORED_SCENARIOS:
            positive = scenarios[positive_name]
            negative = scenarios[negative_name]
            mirrored[label] = {
                "positive_scenario": positive_name,
                "negative_scenario": negative_name,
                "front_thigh_ctrl_ratio_delta": (
                    positive["pair_comparisons"]["front_thigh"]["ctrl_abs_mean_ratio"]
                    - negative["pair_comparisons"]["front_thigh"]["ctrl_abs_mean_ratio"]
                ),
                "front_thigh_error_ratio_delta": (
                    positive["pair_comparisons"]["front_thigh"]["q_target_err_abs_mean_ratio"]
                    - negative["pair_comparisons"]["front_thigh"]["q_target_err_abs_mean_ratio"]
                ),
                "projected_gravity_y_sum": (
                    positive["projected_gravity_y_mean"]
                    + negative["projected_gravity_y_mean"]
                ),
            }

        report = {
            "backend": "isaacsim",
            "crosssim_contract": PROFILE_CONTRACT,
            "resolved_evaluation": {
                "control_rate_hz": float(1.0 / env.unwrapped.step_dt),
                "physics_dt_s": float(env_cfg.sim.dt),
                "control_dt_s": float(env.unwrapped.step_dt),
                "max_steps": args_cli.max_steps,
                "warmup_steps": args_cli.warmup_steps,
                "seed_base": args_cli.seed,
                "rollouts_per_scenario": args_cli.rollouts_per_scenario,
                "reset_preset": args_cli.reset_preset,
                "reset_pos_xy_jitter": max(abs(v) for v in PROFILE_RESET["base_xy_uniform_m"]) if args_cli.reset_preset == "light" else 0.0,
                "reset_yaw_jitter_deg": max(abs(v) for v in PROFILE_RESET["base_yaw_uniform_deg"]) if args_cli.reset_preset == "light" else 0.0,
                "reset_joint_pos_jitter": max(abs(v) for v in PROFILE_RESET["joint_position_uniform_rad"]) if args_cli.reset_preset == "light" else 0.0,
                "reset_joint_vel_jitter": max(abs(v) for v in PROFILE_RESET["joint_velocity_uniform_rad_s"]) if args_cli.reset_preset == "light" else 0.0,
                "obs_delay_steps": args_cli.obs_delay_steps,
                "action_delay_steps": args_cli.action_delay_steps,
                "command_delay_steps": args_cli.command_delay_steps,
                "actuator_model": "isaac_dc_motor",
                "scenario_names": [name for name, _, _ in SCENARIOS],
                "terrain_mode": PROFILE_SCENARIOS["terrain_contract"]["mode"],
            },
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "seed": args_cli.seed,
            "rollouts_per_scenario": args_cli.rollouts_per_scenario,
            "max_steps": args_cli.max_steps,
            "warmup_steps": args_cli.warmup_steps,
            "environment": {
                "terrain": "plane",
                "terrain_contract": PROFILE_SCENARIOS["terrain_contract"],
                "static_friction": PROFILE["nominal_robot"]["ground_static_friction"],
                "dynamic_friction": PROFILE["nominal_robot"]["ground_dynamic_friction"],
                "mass_offset_kg": 0.0,
                "com_offset_m": [0.0, 0.0, 0.0],
                "stiffness_scale": 1.0,
                "damping_scale": 1.0,
                "reset_preset": args_cli.reset_preset,
                "reset_pos_xy_jitter_m": max(abs(v) for v in PROFILE_RESET["base_xy_uniform_m"]) if args_cli.reset_preset == "light" else 0.0,
                "reset_yaw_jitter_deg": max(abs(v) for v in PROFILE_RESET["base_yaw_uniform_deg"]) if args_cli.reset_preset == "light" else 0.0,
                "reset_joint_pos_jitter_rad": max(abs(v) for v in PROFILE_RESET["joint_position_uniform_rad"]) if args_cli.reset_preset == "light" else 0.0,
                "reset_joint_vel_jitter_rad_s": max(abs(v) for v in PROFILE_RESET["joint_velocity_uniform_rad_s"]) if args_cli.reset_preset == "light" else 0.0,
                "obs_delay_steps": args_cli.obs_delay_steps,
                "action_delay_steps": args_cli.action_delay_steps,
                "command_delay_steps": args_cli.command_delay_steps,
                "external_forces_every_physx_iteration": False,
            },
            "resolved_runtime_contract": _runtime_contract_snapshot(
                env_cfg,
                robot,
                action_term,
            ),
            "push": {
                "force_n": args_cli.push_force_n,
                "start_step": args_cli.push_start_step,
                "duration_steps": args_cli.push_duration_steps,
                "frame": "world",
                "body": "base",
            },
            "interpretation": {
                "ratio": "Above 1.0 means the numerator joint works harder or tracks worse than its mirror.",
                "fixed_asymmetry": "A ratio staying above 1.0 under mirrored commands suggests fixed policy/model asymmetry.",
                "load_dependent": "A ratio crossing 1.0 under mirrored commands suggests expected load-dependent behavior.",
                "reset_count": "Any non-zero count indicates a termination hidden by Isaac's automatic reset.",
                "latency_scope": (
                    "The default one-step full-observation and action delays match "
                    "mujoco_fr_asymmetry_matched_v2. Command delay is recorded but has no numerical "
                    "effect because each scenario command is fixed before the first policy step."
                ),
                "termination_counts": (
                    "Term-level counts split reset_count by Isaac termination term. "
                    "For a clean fixed-horizon evaluation, reset_count should mostly be time_out/truncated."
                ),
            },
            "scenarios": scenarios,
            "mirrored_pair_deltas": mirrored,
        }

        text = json.dumps(report, indent=2)
        if args_cli.print_json:
            print(text)
        args_cli.json_out.parent.mkdir(parents=True, exist_ok=True)
        args_cli.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"[INFO] Wrote report: {args_cli.json_out.resolve()}")
        print(
            "[INFO] Evaluation complete: "
            f"scenarios={len(scenarios)}, "
            f"total_resets={sum(item['reset_count'] for item in scenarios.values())}"
        )
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
