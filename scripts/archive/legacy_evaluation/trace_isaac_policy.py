#!/usr/bin/env python3
"""Trace an IsaacLab policy rollout with MuJoCo-comparable gait diagnostics.

This script exists to make Isaac-side rollout logging comparable to the
MuJoCo-side `run_sim2sim.py` trace:

- per-step foot contacts
- per-step foot world positions
- root pose / velocity / tilt
- raw policy action
- processed joint-position targets
- live history latent norm for `Adapt-V3` policies
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument(
    "--trace-steps",
    type=int,
    default=-1,
    help="Number of initial control steps to trace. Use -1 to capture the full rollout.",
)
parser.add_argument("--json-out", type=str, default=None)
parser.add_argument("--command-x", type=float, default=None)
parser.add_argument("--command-y", type=float, default=None)
parser.add_argument("--command-yaw", type=float, default=None)
parser.add_argument(
    "--history-ablation",
    type=str,
    default="normal",
    choices=("normal", "zero", "frozen"),
    help="Optional runtime ablation for the deployable policy_history observation.",
)
parser.add_argument("--progress-every", type=int, default=0)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

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


def _extract_feet(env):
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    robot = env.unwrapped.scene["robot"]
    robot_feet = SceneEntityCfg("robot", body_names=".*_foot")
    sensor_feet = SceneEntityCfg("contact_forces", body_names=".*_foot")
    robot_feet.resolve(env.unwrapped.scene)
    sensor_feet.resolve(env.unwrapped.scene)
    return robot, contact_sensor, robot_feet.body_ids, sensor_feet.body_ids


def _force_command(env_cfg) -> None:
    if (
        args_cli.command_x is None
        and args_cli.command_y is None
        and args_cli.command_yaw is None
    ):
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


def _foot_contact_state(contact_sensor, sensor_foot_ids: list[int]) -> torch.Tensor:
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :].norm(dim=-1).max(dim=1).values
    return forces > 1.0


def _history_tail_head(history_obs: torch.Tensor, policy_dim: int) -> list[float]:
    if history_obs.ndim != 2 or history_obs.shape[-1] % policy_dim != 0:
        return []
    hist = history_obs[0].reshape(-1, policy_dim)
    return hist[-1, :12].detach().cpu().tolist()


def _apply_history_ablation(obs, mode: str, frozen_history: torch.Tensor | None):
    if mode == "normal" or "policy_history" not in obs.keys():
        return obs
    obs = obs.clone()
    if mode == "zero":
        obs["policy_history"] = torch.zeros_like(obs["policy_history"])
    elif mode == "frozen":
        if frozen_history is None:
            raise RuntimeError("Frozen history ablation requested without an initial frozen history snapshot.")
        obs["policy_history"] = frozen_history.clone()
    else:
        raise RuntimeError(f"Unsupported history ablation mode: {mode}")
    return obs


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _force_command(env_cfg)

    runner_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    raw_env = None
    env = None
    try:
        raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=runner_cfg.clip_actions)

        if runner_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
        elif runner_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
        else:
            raise RuntimeError(f"Unsupported runner class: {runner_cfg.class_name}")
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        robot, contact_sensor, foot_ids, sensor_foot_ids = _extract_feet(env)
        action_term_name = env.unwrapped.action_manager.active_terms[0]
        action_term = env.unwrapped.action_manager._terms[action_term_name]

        env.reset()
        obs = _unwrap_obs(env.get_observations())
        policy_obs_dim = int(obs["policy"].shape[-1])
        frozen_history_obs = (
            obs["policy_history"].detach().clone()
            if args_cli.history_ablation == "frozen" and "policy_history" in obs.keys()
            else None
        )

        reward_proxy = []
        vel_err = []
        yaw_err = []
        base_height = []
        base_tilt = []
        action_abs = []
        latent_norm = []
        latent_max_abs = []
        actor_obs_norm = []
        foot_contact_counts = {name: 0 for name in FOOT_LABELS}
        foot_height_means = {name: [] for name in FOOT_LABELS}
        trace = []

        with torch.inference_mode():
            for step_idx in range(args_cli.max_steps):
                ablated_obs = _apply_history_ablation(obs, args_cli.history_ablation, frozen_history_obs)
                current_policy_obs = ablated_obs["policy"]
                current_history_obs = ablated_obs["policy_history"]

                actions = policy(ablated_obs)
                latent = None
                actor_obs = None
                if hasattr(policy_nn, "adapt_from_history") and hasattr(policy_nn, "act_with_latent"):
                    latent = policy_nn.adapt_from_history(current_history_obs)
                    actor_obs = torch.cat([current_policy_obs, latent], dim=-1)

                obs, _, dones, _ = _step_env(env, actions)
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

                reward_proxy.append(float(planar_vel.norm(dim=-1).mean().item()))
                vel_err.append(float(vel_err_step.mean().item()))
                yaw_err.append(float(yaw_err_step.mean().item()))
                base_height.append(float(robot.data.root_pos_w[:, 2].mean().item()))
                base_tilt.append(float(base_tilt_step.mean().item()))
                action_abs.append(float(actions.abs().mean().item()))
                if latent is not None:
                    latent_norm.append(float(latent.norm(dim=-1).mean().item()))
                    latent_max_abs.append(float(latent.abs().max(dim=-1).values.mean().item()))
                if actor_obs is not None:
                    actor_obs_norm.append(float(actor_obs.norm(dim=-1).mean().item()))

                for idx, label in enumerate(FOOT_LABELS):
                    foot_contact_counts[label] += int(contacts[:, idx].sum().item())
                    foot_height_means[label].append(float(robot.data.body_pos_w[:, foot_ids[idx], 2].mean().item()))

                if args_cli.trace_steps < 0 or step_idx < args_cli.trace_steps:
                    env0 = 0
                    foot_pos_world = {
                        label: robot.data.body_pos_w[env0, foot_ids[idx]].detach().cpu().tolist()
                        for idx, label in enumerate(FOOT_LABELS)
                    }
                    foot_contact = {
                        label: bool(contacts[env0, idx].item())
                        for idx, label in enumerate(FOOT_LABELS)
                    }
                    joint_pos = robot.data.joint_pos[env0].detach().cpu()
                    joint_vel = robot.data.joint_vel[env0].detach().cpu()
                    default_joint_pos = robot.data.default_joint_pos[env0].detach().cpu()
                    trace.append(
                        {
                            "step": step_idx,
                            "command": command[env0].detach().cpu().tolist(),
                            "root_pos_world": robot.data.root_pos_w[env0].detach().cpu().tolist(),
                            "root_quat_world_wxyz": robot.data.root_quat_w[env0].detach().cpu().tolist(),
                            "world_lin_vel": robot.data.root_lin_vel_w[env0].detach().cpu().tolist(),
                            "base_lin_vel_local": robot.data.root_lin_vel_b[env0].detach().cpu().tolist(),
                            "base_ang_vel_local": robot.data.root_ang_vel_b[env0].detach().cpu().tolist(),
                            "projected_gravity": (
                                projected_gravity[env0].detach().cpu().tolist()
                                if projected_gravity is not None
                                else None
                            ),
                            "root_height": float(robot.data.root_pos_w[env0, 2].item()),
                            "base_tilt_xy_norm": float(base_tilt_step[env0].item()),
                            "vel_err": float(vel_err_step[env0].item()),
                            "yaw_err": float(yaw_err_step[env0].item()),
                            "joint_pos": joint_pos.tolist(),
                            "joint_pos_rel": (joint_pos - default_joint_pos).tolist(),
                            "joint_vel": joint_vel.tolist(),
                            "last_action_used_in_obs": current_policy_obs[env0, 36:48].detach().cpu().tolist(),
                            "policy_obs_head": current_policy_obs[env0, :12].detach().cpu().tolist(),
                            "history_tail_head": _history_tail_head(current_history_obs, policy_obs_dim),
                            "foot_pos_world": foot_pos_world,
                            "foot_contact": foot_contact,
                            "action": actions[env0].detach().cpu().tolist(),
                            "action_abs_mean": float(actions[env0].abs().mean().item()),
                            "latent": latent[env0].detach().cpu().tolist() if latent is not None else None,
                            "latent_norm": float(latent[env0].norm().item()) if latent is not None else None,
                            "latent_max_abs": float(latent[env0].abs().max().item()) if latent is not None else None,
                            "actor_obs_norm": float(actor_obs[env0].norm().item()) if actor_obs is not None else None,
                            "q_target": action_term.processed_actions[env0].detach().cpu().tolist(),
                            "raw_action_term": action_term.raw_actions[env0].detach().cpu().tolist(),
                        }
                    )

                if args_cli.progress_every > 0 and (
                    (step_idx + 1) % args_cli.progress_every == 0 or (step_idx + 1) == args_cli.max_steps
                ):
                    print(
                        json.dumps(
                            {
                                "progress_step": step_idx + 1,
                                "max_steps": args_cli.max_steps,
                                "reward_proxy_mean_so_far": sum(reward_proxy) / max(len(reward_proxy), 1),
                                "vel_err_step_mean_so_far": sum(vel_err) / max(len(vel_err), 1),
                                "base_height_mean_so_far": sum(base_height) / max(len(base_height), 1),
                            }
                        ),
                        flush=True,
                    )

                if isinstance(dones, torch.Tensor) and bool(dones.any().item()):
                    policy_nn.reset(dones)

        results = {
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "seed": args_cli.seed,
            "history_ablation": args_cli.history_ablation,
            "status": "completed_runtime_trace",
            "summary_metrics": {
                "reward_proxy_mean": float(sum(reward_proxy) / max(len(reward_proxy), 1)),
                "vel_err_step_mean": float(sum(vel_err) / max(len(vel_err), 1)),
                "yaw_err_step_mean": float(sum(yaw_err) / max(len(yaw_err), 1)),
                "base_height_mean": float(sum(base_height) / max(len(base_height), 1)),
                "base_tilt_projected_gravity_xy_mean": float(sum(base_tilt) / max(len(base_tilt), 1)),
                "action_abs_mean": float(sum(action_abs) / max(len(action_abs), 1)),
                "latent_norm_mean": float(sum(latent_norm) / max(len(latent_norm), 1)) if latent_norm else 0.0,
                "latent_norm_max": float(max(latent_norm)) if latent_norm else 0.0,
                "latent_max_abs_mean": float(sum(latent_max_abs) / max(len(latent_max_abs), 1))
                if latent_max_abs
                else 0.0,
                "latent_max_abs_max": float(max(latent_max_abs)) if latent_max_abs else 0.0,
                "actor_obs_norm_mean": float(sum(actor_obs_norm) / max(len(actor_obs_norm), 1))
                if actor_obs_norm
                else 0.0,
                "actor_obs_norm_max": float(max(actor_obs_norm)) if actor_obs_norm else 0.0,
                "foot_contact_fraction": {
                    label: float(foot_contact_counts[label] / max(args_cli.max_steps * args_cli.num_envs, 1))
                    for label in FOOT_LABELS
                },
                "foot_height_mean": {
                    label: float(sum(foot_height_means[label]) / max(len(foot_height_means[label]), 1))
                    for label in FOOT_LABELS
                },
            },
            "trace_steps_captured": len(trace),
            "trace": trace,
        }
        print(json.dumps(results, indent=2))
        if args_cli.json_out:
            out = Path(args_cli.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
