#!/usr/bin/env python3
"""Export a compact forward-velocity tracking trace from IsaacLab."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--json-out", required=True)
parser.add_argument("--csv-out", default="")
parser.add_argument("--command-x", type=float, default=None)
parser.add_argument("--command-y", type=float, default=None)
parser.add_argument("--command-yaw", type=float, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


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

        env.reset()
        obs = _unwrap_obs(env.get_observations())
        robot = env.unwrapped.scene["robot"]

        time_s = []
        cmd_vx = []
        vx = []
        vy = []
        yaw_rate = []
        vel_err = []
        yaw_err = []
        csv_handle = None
        csv_writer = None
        if args_cli.csv_out:
            csv_path = Path(args_cli.csv_out)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_handle = csv_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_handle)
            csv_writer.writerow(["step", "time_s", "cmd_vx", "vx", "vy", "yaw_rate"])

        with torch.inference_mode():
            for step_idx in range(args_cli.max_steps):
                actions = policy(obs)
                obs, _, dones, _ = _step_env(env, actions)
                obs = _unwrap_obs(obs)
                command = env.unwrapped.command_manager.get_command("base_velocity")

                env0 = 0
                time_s.append(step_idx * env.unwrapped.step_dt)
                cmd_vx.append(float(command[env0, 0].item()))
                vx.append(float(robot.data.root_lin_vel_b[env0, 0].item()))
                vy.append(float(robot.data.root_lin_vel_b[env0, 1].item()))
                yaw_rate.append(float(robot.data.root_ang_vel_b[env0, 2].item()))
                vel_err.append(float((robot.data.root_lin_vel_b[env0, 0] - command[env0, 0]).abs().item()))
                yaw_err.append(float((robot.data.root_ang_vel_b[env0, 2] - command[env0, 2]).abs().item()))
                if csv_writer is not None:
                    csv_writer.writerow([step_idx, time_s[-1], cmd_vx[-1], vx[-1], vy[-1], yaw_rate[-1]])
                    if (step_idx + 1) % 50 == 0 and csv_handle is not None:
                        csv_handle.flush()

                if bool(dones[env0].item()):
                    runner.alg.actor_critic.reset(dones)

        results = {
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "seed": args_cli.seed,
            "max_steps": args_cli.max_steps,
            "series": {
                "time_s": time_s,
                "cmd_vx": cmd_vx,
                "vx": vx,
                "vy": vy,
                "yaw_rate": yaw_rate,
            },
            "summary_metrics": {
                "vel_err_step_mean": float(sum(vel_err) / max(len(vel_err), 1)),
                "yaw_err_step_mean": float(sum(yaw_err) / max(len(yaw_err), 1)),
            },
        }
        out = Path(args_cli.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {out}")
        if csv_handle is not None:
            csv_handle.flush()
            csv_handle.close()
            print(f"[INFO] Wrote {args_cli.csv_out}")
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
