"""Debug and validate privileged observations for Teacher V3.

This script is meant to answer a simple question before training:

"What privileged values is V3 actually getting from the simulator?"

It launches the V3 task headlessly, optionally applies deterministic
randomization overrides, resets the environment, and prints:

- observation group shapes
- summary stats for terrain/dynamics privilege
- direct consistency checks against live simulator state
- drift checks over a short zero-action rollout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect and validate Teacher V3 privileged observations.")
parser.add_argument("--task", type=str, default="RMA-Go2-Privileged-Teacher-Rough-V3")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=64, help="Short zero-action rollout length for drift checks.")
parser.add_argument("--sample-envs", type=int, default=4, help="How many env rows to print.")
parser.add_argument("--json-out", type=str, default=None, help="Optional JSON dump path.")
parser.add_argument("--static-friction", type=float, default=None)
parser.add_argument("--dynamic-friction", type=float, default=None)
parser.add_argument("--mass-offset", type=float, default=None)
parser.add_argument("--motor-stiffness-scale", type=float, default=None)
parser.add_argument("--motor-damping-scale", type=float, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[2]


def _apply_randomization_overrides(env_cfg) -> None:
    if args_cli.static_friction is not None and env_cfg.events.physics_material is not None:
        env_cfg.events.physics_material.params["static_friction_range"] = (
            args_cli.static_friction,
            args_cli.static_friction,
        )
    if args_cli.dynamic_friction is not None and env_cfg.events.physics_material is not None:
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (
            args_cli.dynamic_friction,
            args_cli.dynamic_friction,
        )
    if args_cli.mass_offset is not None and env_cfg.events.add_base_mass is not None:
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (
            args_cli.mass_offset,
            args_cli.mass_offset,
        )
    if args_cli.motor_stiffness_scale is not None and hasattr(env_cfg.events, "motor_strength"):
        env_cfg.events.motor_strength.params["stiffness_distribution_params"] = (
            args_cli.motor_stiffness_scale,
            args_cli.motor_stiffness_scale,
        )
    if args_cli.motor_damping_scale is not None and hasattr(env_cfg.events, "motor_strength"):
        env_cfg.events.motor_strength.params["damping_distribution_params"] = (
            args_cli.motor_damping_scale,
            args_cli.motor_damping_scale,
        )


def _tensor_stats(x: torch.Tensor) -> dict[str, float | list[int]]:
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _sample_rows(x: torch.Tensor, rows: int) -> list[list[float]]:
    take = min(rows, x.shape[0])
    return [[float(v) for v in row.tolist()] for row in x[:take]]


def _collect_live_dynamics(env) -> dict[str, torch.Tensor]:
    term_material = env.unwrapped.event_manager.get_term_cfg("physics_material").func
    term_mass = env.unwrapped.event_manager.get_term_cfg("add_base_mass").func
    robot = env.unwrapped.scene["robot"]
    joint_stiffness_scale = torch.zeros_like(robot.data.default_joint_stiffness)
    joint_damping_scale = torch.zeros_like(robot.data.default_joint_damping)
    for actuator in robot.actuators.values():
        joint_ids = actuator.joint_indices
        joint_stiffness_scale[:, joint_ids] = actuator.stiffness / torch.clamp(
            robot.data.default_joint_stiffness[:, joint_ids],
            min=1e-6,
        )
        joint_damping_scale[:, joint_ids] = actuator.damping / torch.clamp(
            robot.data.default_joint_damping[:, joint_ids],
            min=1e-6,
        )

    return {
        "static_friction": term_material.env_static_friction.clone(),
        "dynamic_friction": term_material.env_dynamic_friction.clone(),
        "base_mass_ratio": term_mass.env_base_mass_ratio.clone(),
        "joint_stiffness_scale": joint_stiffness_scale.clone(),
        "joint_damping_scale": joint_damping_scale.clone(),
    }


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _apply_randomization_overrides(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        obs, _ = env.reset(seed=args_cli.seed)
        obs_groups = env.unwrapped.observation_manager.compute()

        group_shapes = {}
        for name, value in obs_groups.items():
            if isinstance(value, dict):
                group_shapes[name] = {k: list(v.shape) for k, v in value.items()}
            else:
                group_shapes[name] = list(value.shape)

        terrain_priv = obs_groups["terrain_privileged"]
        dynamics_priv = obs_groups["dynamics_privileged"]
        live_dyn = _collect_live_dynamics(env)

        expected_dynamics = torch.cat(
            [
                live_dyn["static_friction"],
                live_dyn["dynamic_friction"],
                live_dyn["base_mass_ratio"],
                live_dyn["joint_stiffness_scale"],
                live_dyn["joint_damping_scale"],
            ],
            dim=-1,
        )

        consistency = {
            "static_friction_max_abs_diff": _max_abs_diff(dynamics_priv[:, 0:1], live_dyn["static_friction"]),
            "dynamic_friction_max_abs_diff": _max_abs_diff(dynamics_priv[:, 1:2], live_dyn["dynamic_friction"]),
            "base_mass_ratio_max_abs_diff": _max_abs_diff(dynamics_priv[:, 2:3], live_dyn["base_mass_ratio"]),
            "joint_stiffness_scale_max_abs_diff": _max_abs_diff(
                dynamics_priv[:, 3:15], live_dyn["joint_stiffness_scale"]
            ),
            "joint_damping_scale_max_abs_diff": _max_abs_diff(
                dynamics_priv[:, 15:27], live_dyn["joint_damping_scale"]
            ),
            "full_dynamics_vector_max_abs_diff": _max_abs_diff(dynamics_priv, expected_dynamics),
        }

        initial_dynamics = dynamics_priv.clone()
        initial_terrain = terrain_priv.clone()
        zero_action = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )

        for _ in range(args_cli.steps):
            env.step(zero_action)

        post_obs_groups = env.unwrapped.observation_manager.compute()
        post_dynamics = post_obs_groups["dynamics_privileged"]
        post_terrain = post_obs_groups["terrain_privileged"]

        drift = {
            "dynamics_max_abs_change_over_rollout": _max_abs_diff(post_dynamics, initial_dynamics),
            "terrain_max_abs_change_over_rollout": _max_abs_diff(post_terrain, initial_terrain),
        }

        result = {
            "task": args_cli.task,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "overrides": {
                "static_friction": args_cli.static_friction,
                "dynamic_friction": args_cli.dynamic_friction,
                "mass_offset": args_cli.mass_offset,
                "motor_stiffness_scale": args_cli.motor_stiffness_scale,
                "motor_damping_scale": args_cli.motor_damping_scale,
            },
            "group_shapes": group_shapes,
            "terrain_privileged_stats": _tensor_stats(terrain_priv),
            "dynamics_privileged_stats": _tensor_stats(dynamics_priv),
            "consistency": consistency,
            "drift": drift,
            "sample_dynamics_rows": _sample_rows(dynamics_priv, args_cli.sample_envs),
        }

        print("\n=== V3 Privilege Debug ===")
        print(json.dumps(result, indent=2))

        if args_cli.json_out is not None:
            output_path = Path(args_cli.json_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"[INFO] Wrote JSON to: {output_path}")

    finally:
        env.close()
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
