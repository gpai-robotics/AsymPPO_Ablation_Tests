#!/usr/bin/env python3
"""Evaluate a flat omni prior on a fixed teleop-style command schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--name", default="")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--steps-per-segment", type=int, default=180)
parser.add_argument("--warmup-steps", type=int, default=30)
parser.add_argument("--json-out", required=True)
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


SEGMENTS = [
    ("standstill", (0.0, 0.0, 0.0)),
    ("forward", (1.0, 0.0, 0.0)),
    ("backward", (-1.0, 0.0, 0.0)),
    ("left", (0.0, 0.4, 0.0)),
    ("right", (0.0, -0.4, 0.0)),
    ("yaw_left", (0.0, 0.0, 1.0)),
    ("yaw_right", (0.0, 0.0, -1.0)),
    ("forward_left", (1.0, 0.4, 0.0)),
    ("forward_right", (1.0, -0.4, 0.0)),
    ("forward_yaw_left", (1.0, 0.0, 1.0)),
    ("forward_yaw_right", (1.0, 0.0, -1.0)),
]

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


def _foot_contact_state(contact_sensor, sensor_foot_ids: list[int]) -> torch.Tensor:
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :].norm(dim=-1).max(dim=1).values
    return forces > 1.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _eval_segment(env, policy, policy_nn, robot, contact_sensor, foot_ids, sensor_foot_ids, name: str, command: tuple[float, float, float]) -> dict[str, object]:
    base_command = env.unwrapped.command_manager.get_command("base_velocity")

    vel_err_xy = []
    vel_err_yaw = []
    achieved_vx = []
    achieved_vy = []
    achieved_yaw = []
    root_height = []
    base_tilt = []
    action_abs = []
    foot_slide_proxy = []
    foot_contact_fraction = {label: [] for label in FOOT_LABELS}
    done_count = 0

    for step_idx in range(args_cli.steps_per_segment):
        with torch.inference_mode():
            base_command[:, 0] = float(command[0])
            base_command[:, 1] = float(command[1])
            base_command[:, 2] = float(command[2])
            actions = policy(env.get_observations())
            obs, _, dones, _ = _step_env(env, actions)
            policy_nn.reset(dones)

            contacts = _foot_contact_state(contact_sensor, sensor_foot_ids)
            planar_vel = robot.data.root_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_ang_vel_b[:, 2]
            projected_gravity = getattr(robot.data, "projected_gravity_b", None)
            foot_planar_speed = torch.linalg.norm(robot.data.body_lin_vel_w[:, foot_ids, :2], dim=-1)

            if step_idx >= args_cli.warmup_steps:
                target_xy = torch.tensor(command[:2], device=env.unwrapped.device, dtype=planar_vel.dtype).view(1, 2)
                vel_err_xy.append(float(torch.linalg.norm(planar_vel - target_xy, dim=-1).mean().item()))
                vel_err_yaw.append(float((yaw_vel - float(command[2])).abs().mean().item()))
                achieved_vx.append(float(planar_vel[:, 0].mean().item()))
                achieved_vy.append(float(planar_vel[:, 1].mean().item()))
                achieved_yaw.append(float(yaw_vel.mean().item()))
                root_height.append(float(robot.data.root_pos_w[:, 2].mean().item()))
                if projected_gravity is not None:
                    base_tilt.append(float(projected_gravity[:, :2].norm(dim=1).mean().item()))
                action_abs.append(float(actions.abs().mean().item()))

                contact_float = contacts.float()
                contact_weight = float(contact_float.sum().item())
                if contact_weight > 0.0:
                    slide = float((foot_planar_speed * contact_float).sum().item() / contact_weight)
                else:
                    slide = 0.0
                foot_slide_proxy.append(slide)
                for idx, label in enumerate(FOOT_LABELS):
                    foot_contact_fraction[label].append(float(contact_float[:, idx].mean().item()))

            if isinstance(dones, torch.Tensor) and bool(dones.any().item()):
                done_count += int(dones.sum().item())
                env.reset()
                obs = _unwrap_obs(obs)

        if args_cli.progress_every > 0 and (step_idx + 1) % args_cli.progress_every == 0:
            print(json.dumps({"segment": name, "step": step_idx + 1, "steps_per_segment": args_cli.steps_per_segment}), flush=True)

    return {
        "segment": name,
        "command": {"x": float(command[0]), "y": float(command[1]), "yaw": float(command[2])},
        "metrics": {
            "vel_err_xy_mean": _mean(vel_err_xy),
            "vel_err_yaw_mean": _mean(vel_err_yaw),
            "achieved_vx_mean": _mean(achieved_vx),
            "achieved_vy_mean": _mean(achieved_vy),
            "achieved_yaw_mean": _mean(achieved_yaw),
            "root_height_mean": _mean(root_height),
            "base_tilt_mean": _mean(base_tilt),
            "action_abs_mean": _mean(action_abs),
            "foot_slide_proxy_mean": _mean(foot_slide_proxy),
            "done_count": int(done_count),
            "foot_contact_fraction": {label: _mean(values) for label, values in foot_contact_fraction.items()},
        },
    }


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 1.0e6
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.heading_command = False

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
        env.reset()

        segments = []
        for segment_name, command in SEGMENTS:
            segments.append(
                _eval_segment(
                    env,
                    policy,
                    policy_nn,
                    robot,
                    contact_sensor,
                    foot_ids,
                    sensor_foot_ids,
                    segment_name,
                    command,
                )
            )

        result = {
            "name": args_cli.name or Path(args_cli.checkpoint).stem,
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "steps_per_segment": args_cli.steps_per_segment,
            "warmup_steps": args_cli.warmup_steps,
            "segments": segments,
        }

        out = Path(args_cli.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {out}")
        return 0
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
