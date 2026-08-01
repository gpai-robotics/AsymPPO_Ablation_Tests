#!/usr/bin/env python3
"""Audit whether Teacher V3 actually depends on terrain/dynamics privilege.

This script answers the root-audit question directly:

- Does zeroing privileged groups change the teacher's action on the same obs?
- Does forcing those ablations during rollout materially degrade behavior?

It runs:
- a normal rollout
- counterfactual action checks on the same normal-rollout observations
- separate ablated rollouts for selected modes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


ABLATION_MODES = (
    "normal",
    "zero_terrain",
    "zero_dynamics",
    "zero_both",
    "shuffled_terrain",
    "shuffled_dynamics",
    "shuffled_both",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True, help="Teacher checkpoint path.")
parser.add_argument("--task", type=str, default="RMA-Go2-Privileged-Teacher-Rough-V3")
parser.add_argument("--terrain-type", type=str, default=None)
parser.add_argument("--terrain-level", type=int, default=-1)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument(
    "--modes",
    nargs="+",
    default=["normal", "zero_terrain", "zero_dynamics", "zero_both"],
    choices=ABLATION_MODES,
)
parser.add_argument("--trace-steps", type=int, default=200)
parser.add_argument("--progress-every", type=int, default=100, help="Print rollout progress every N steps. Set <= 0 to disable.")
parser.add_argument("--json-out", type=str, default=None)
parser.add_argument("--command-x", type=float, default=None)
parser.add_argument("--command-y", type=float, default=None)
parser.add_argument("--command-yaw", type=float, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


FOOT_LABELS = ("FL", "FR", "RL", "RR")


def _unwrap_obs(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _step_env(env, actions):
    step_out = env.step(actions)
    if len(step_out) == 5:
        obs, rewards, terminated, truncated, infos = step_out
        dones = terminated | truncated
        return obs, rewards, dones, infos
    if len(step_out) == 4:
        obs, rewards, dones, infos = step_out
        return obs, rewards, dones, infos
    raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")


def _force_isolated_terrain(env_cfg, terrain_type: str | None) -> None:
    if terrain_type is None:
        return
    env_cfg.scene.terrain.max_init_terrain_level = 9
    terrain_gen = env_cfg.scene.terrain.terrain_generator
    for key in terrain_gen.sub_terrains.keys():
        terrain_gen.sub_terrains[key].proportion = 0.0
    if terrain_type not in terrain_gen.sub_terrains:
        valid = list(terrain_gen.sub_terrains.keys())
        raise ValueError(f"Unknown terrain type '{terrain_type}'. Valid options: {valid}")
    terrain_gen.sub_terrains[terrain_type].proportion = 1.0


def _disable_terrain_curriculum_for_fixed_level(env_cfg, terrain_level: int | None) -> None:
    if terrain_level is None or terrain_level < 0:
        return
    if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None


def _force_terrain_level(env, terrain_level: int | None) -> None:
    if terrain_level is None or terrain_level < 0:
        return
    terrain = env.unwrapped.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None:
        return
    level = max(0, min(int(terrain_level), int(terrain.terrain_origins.shape[0]) - 1))
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.unwrapped.scene.env_origins[:] = terrain.env_origins


def _force_command(env_cfg) -> None:
    if args_cli.command_x is None and args_cli.command_y is None and args_cli.command_yaw is None:
        return
    cmd = env_cfg.commands.base_velocity
    if args_cli.command_x is not None:
        cmd.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
    if args_cli.command_y is not None:
        cmd.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
    if args_cli.command_yaw is not None:
        cmd.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.heading_command = False


def _extract_feet(env):
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    robot = env.unwrapped.scene["robot"]
    robot_feet = SceneEntityCfg("robot", body_names=".*_foot")
    sensor_feet = SceneEntityCfg("contact_forces", body_names=".*_foot")
    robot_feet.resolve(env.unwrapped.scene)
    sensor_feet.resolve(env.unwrapped.scene)
    return robot, contact_sensor, robot_feet.body_ids, sensor_feet.body_ids


def _foot_contact_state(contact_sensor, sensor_foot_ids: list[int]) -> torch.Tensor:
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :].norm(dim=-1).max(dim=1).values
    return forces > 1.0


def _apply_privilege_ablation(obs, mode: str):
    if mode == "normal":
        return obs
    obs = obs.clone()
    if mode in ("zero_terrain", "zero_both") and "terrain_privileged" in obs.keys():
        obs["terrain_privileged"] = torch.zeros_like(obs["terrain_privileged"])
    if mode in ("zero_dynamics", "zero_both") and "dynamics_privileged" in obs.keys():
        obs["dynamics_privileged"] = torch.zeros_like(obs["dynamics_privileged"])
    if mode in ("shuffled_terrain", "shuffled_both") and "terrain_privileged" in obs.keys():
        obs["terrain_privileged"] = obs["terrain_privileged"].roll(shifts=1, dims=0)
    if mode in ("shuffled_dynamics", "shuffled_both") and "dynamics_privileged" in obs.keys():
        obs["dynamics_privileged"] = obs["dynamics_privileged"].roll(shifts=1, dims=0)
    return obs


def _build_env_and_policy():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_level)
    _force_command(env_cfg)

    agent_cfg_obj = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = agent_cfg_obj.to_dict()
    if "policy" in agent_cfg and isinstance(agent_cfg["policy"], dict):
        agent_cfg["policy"]["pretrained_path"] = None

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print("[INFO] gym.make complete", flush=True)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg_obj.clip_actions)
    _force_terrain_level(env, args_cli.terrain_level)
    print("[INFO] env wrapper ready", flush=True)

    runner = OnPolicyRunner(env, agent_cfg, log_dir=os.path.dirname(args_cli.checkpoint), device=args_cli.device)
    print("[INFO] runner constructed", flush=True)
    runner.load(args_cli.checkpoint)
    print("[INFO] checkpoint loaded", flush=True)
    policy_nn = runner.alg.policy
    policy_nn.eval()
    print("[INFO] policy ready", flush=True)

    robot, contact_sensor, _, sensor_foot_ids = _extract_feet(env)
    print("[INFO] feet extracted", flush=True)
    return raw_env, env, policy_nn, robot, contact_sensor, sensor_foot_ids


def _run_rollout(mode: str, env, policy_nn, robot, contact_sensor, sensor_foot_ids):
    print(f"[INFO] _run_rollout start: mode={mode}", flush=True)
    env.reset()
    print(f"[INFO] env.reset complete: mode={mode}", flush=True)
    _force_terrain_level(env, args_cli.terrain_level)
    obs = _unwrap_obs(env.get_observations())
    print(f"[INFO] initial observations acquired: mode={mode}", flush=True)

    reward_mean = []
    reward_proxy = []
    vel_err = []
    yaw_err = []
    base_height = []
    base_tilt = []
    action_abs = []
    done_events = 0
    trace = []
    counterfactual_diffs = {other: [] for other in args_cli.modes if other != "normal"} if mode == "normal" else {}

    with torch.inference_mode():
        print(f"[INFO] entering rollout loop: mode={mode}", flush=True)
        for step_idx in range(args_cli.steps):
            ablated_obs = _apply_privilege_ablation(obs, mode)
            actions = policy_nn.act_inference(ablated_obs)

            if mode == "normal":
                for other in counterfactual_diffs:
                    other_actions = policy_nn.act_inference(_apply_privilege_ablation(obs, other))
                    diff = (other_actions - actions).abs().mean(dim=-1)
                    counterfactual_diffs[other].append(float(diff.mean().item()))

            obs, rewards, dones, _ = _step_env(env, actions)
            obs = _unwrap_obs(obs)

            command = env.unwrapped.command_manager.get_command("base_velocity")
            planar_vel = robot.data.root_lin_vel_b[:, :2]
            planar_cmd = command[:, :2]
            yaw_vel = robot.data.root_ang_vel_b[:, 2]
            yaw_cmd = command[:, 2]
            projected_gravity = getattr(robot.data, "projected_gravity_b", None)

            vel_err_step = torch.linalg.norm(planar_vel - planar_cmd, dim=-1)
            yaw_err_step = (yaw_vel - yaw_cmd).abs()
            base_tilt_step = (
                projected_gravity[:, :2].norm(dim=1)
                if projected_gravity is not None
                else torch.zeros_like(yaw_err_step)
            )
            contacts = _foot_contact_state(contact_sensor, sensor_foot_ids)

            reward_mean.append(float(rewards.mean().item()))
            reward_proxy.append(float(planar_vel.norm(dim=-1).mean().item()))
            vel_err.append(float(vel_err_step.mean().item()))
            yaw_err.append(float(yaw_err_step.mean().item()))
            base_height.append(float(robot.data.root_pos_w[:, 2].mean().item()))
            base_tilt.append(float(base_tilt_step.mean().item()))
            action_abs.append(float(actions.abs().mean().item()))
            done_events += int(dones.sum().item()) if isinstance(dones, torch.Tensor) else 0

            if args_cli.trace_steps < 0 or step_idx < args_cli.trace_steps:
                env0 = 0
                trace.append(
                    {
                        "step": step_idx,
                        "command": command[env0].detach().cpu().tolist(),
                        "root_pos_world": robot.data.root_pos_w[env0].detach().cpu().tolist(),
                        "base_lin_vel_local": robot.data.root_lin_vel_b[env0].detach().cpu().tolist(),
                        "base_ang_vel_local": robot.data.root_ang_vel_b[env0].detach().cpu().tolist(),
                        "projected_gravity": (
                            projected_gravity[env0].detach().cpu().tolist() if projected_gravity is not None else None
                        ),
                        "root_height": float(robot.data.root_pos_w[env0, 2].item()),
                        "base_tilt_xy_norm": float(base_tilt_step[env0].item()),
                        "vel_err": float(vel_err_step[env0].item()),
                        "yaw_err": float(yaw_err_step[env0].item()),
                        "terrain_privileged_head": (
                            obs["terrain_privileged"][env0, :8].detach().cpu().tolist()
                            if "terrain_privileged" in obs.keys()
                            else None
                        ),
                        "dynamics_privileged": (
                            obs["dynamics_privileged"][env0].detach().cpu().tolist()
                            if "dynamics_privileged" in obs.keys()
                            else None
                        ),
                        "foot_contact": {
                            label: bool(contacts[env0, idx].item()) for idx, label in enumerate(FOOT_LABELS)
                        },
                        "action": actions[env0].detach().cpu().tolist(),
                    }
                )

            if args_cli.progress_every > 0 and (
                (step_idx + 1) % args_cli.progress_every == 0 or (step_idx + 1) == args_cli.steps
            ):
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "progress_step": step_idx + 1,
                            "max_steps": args_cli.steps,
                            "reward_step_mean_so_far": sum(reward_mean) / max(len(reward_mean), 1),
                            "vel_err_step_mean_so_far": sum(vel_err) / max(len(vel_err), 1),
                            "yaw_err_step_mean_so_far": sum(yaw_err) / max(len(yaw_err), 1),
                            "done_events_per_env_so_far": done_events / max(args_cli.num_envs, 1),
                        }
                    ),
                    flush=True,
                )

            if isinstance(dones, torch.Tensor) and bool(dones.any().item()):
                policy_nn.reset(dones)

    summary = {
        "mode": mode,
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "terrain_type": args_cli.terrain_type,
        "terrain_level": args_cli.terrain_level,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "summary_metrics": {
            "reward_step_mean": float(sum(reward_mean) / max(len(reward_mean), 1)),
            "reward_proxy_mean": float(sum(reward_proxy) / max(len(reward_proxy), 1)),
            "vel_err_step_mean": float(sum(vel_err) / max(len(vel_err), 1)),
            "yaw_err_step_mean": float(sum(yaw_err) / max(len(yaw_err), 1)),
            "base_height_mean": float(sum(base_height) / max(len(base_height), 1)),
            "base_tilt_projected_gravity_xy_mean": float(sum(base_tilt) / max(len(base_tilt), 1)),
            "action_abs_mean": float(sum(action_abs) / max(len(action_abs), 1)),
            "done_events_per_env": float(done_events / max(args_cli.num_envs, 1)),
        },
        "counterfactual_action_diffs": {
            key: float(sum(vals) / max(len(vals), 1)) for key, vals in counterfactual_diffs.items()
        }
        if counterfactual_diffs
        else None,
        "trace_steps_captured": len(trace),
        "trace": trace,
    }
    print(f"[INFO] rollout summary ready: mode={mode}", flush=True)
    return summary


def _pairwise_trace_diffs(results_by_mode: dict[str, dict]) -> dict[str, dict[str, float]]:
    if "normal" not in results_by_mode:
        return {}
    normal_trace = results_by_mode["normal"]["trace"]
    out: dict[str, dict[str, float]] = {}
    for mode, result in results_by_mode.items():
        if mode == "normal":
            continue
        other_trace = result["trace"]
        count = min(len(normal_trace), len(other_trace))
        if count == 0:
            out[mode] = {"mean_action_abs_diff_vs_normal": 0.0}
            continue
        diffs = []
        for idx in range(count):
            a = torch.tensor(normal_trace[idx]["action"])
            b = torch.tensor(other_trace[idx]["action"])
            diffs.append(float((a - b).abs().mean().item()))
        out[mode] = {"mean_action_abs_diff_vs_normal": float(sum(diffs) / len(diffs))}
    return out


def main() -> int:
    if len(args_cli.modes) != 1:
        raise SystemExit(
            "Multi-mode runs in one Isaac process are unstable in this audit. "
            "Run one mode at a time, or use scripts/eval/run_teacher_v3_dependency_suite.py."
        )
    results_by_mode = {}
    raw_env = None
    env = None
    try:
        raw_env, env, policy_nn, robot, contact_sensor, sensor_foot_ids = _build_env_and_policy()
        for mode in args_cli.modes:
            print(f"[INFO] Running teacher dependency audit mode: {mode}", flush=True)
            results_by_mode[mode] = _run_rollout(mode, env, policy_nn, robot, contact_sensor, sensor_foot_ids)
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()

    summary = {
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "terrain_type": args_cli.terrain_type,
        "terrain_level": args_cli.terrain_level,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "modes": args_cli.modes,
        "results_by_mode": results_by_mode,
        "pairwise_trace_diffs_vs_normal": _pairwise_trace_diffs(results_by_mode),
    }

    print(json.dumps(summary, indent=2))
    if args_cli.json_out:
        out = Path(args_cli.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote JSON to: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
