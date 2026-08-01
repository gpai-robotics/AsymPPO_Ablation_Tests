"""Validate the per-episode hidden switch mechanism for adaptation tasks.

This script answers a narrow question:

"Does each environment get exactly one real hidden-dynamics switch per episode,
and do the live simulator values match the sampled targets when that switch
fires?"

It works for both:
- RMA-Go2-Adaptation-Student-Rough-NoAdapt
- RMA-Go2-Adaptation-Student-Rough-History
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug adaptation per-episode switch integrity.")
parser.add_argument("--task", type=str, default="RMA-Go2-Adaptation-Student-Rough-NoAdapt")
parser.add_argument("--num-envs", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--switch-step", type=int, default=30)
parser.add_argument("--steps-after-switch", type=int, default=8)
parser.add_argument("--sample-envs", type=int, default=6)
parser.add_argument("--json-out", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


def _make_debug_friendly(env_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.adaptation_switch_step = args_cli.switch_step

    # We want to inspect the switch itself, not early fall terminations.
    if hasattr(env_cfg.terminations, "base_contact"):
        env_cfg.terminations.base_contact = None
    if hasattr(env_cfg.terminations, "base_orientation"):
        env_cfg.terminations.base_orientation = None
    if hasattr(env_cfg.terminations, "base_height"):
        env_cfg.terminations.base_height = None
    if hasattr(env_cfg.terminations, "low_progress"):
        env_cfg.terminations.low_progress = None


def _tensor_rows(x: torch.Tensor, rows: int) -> list[list[float]]:
    take = min(rows, x.shape[0])
    return [[float(v) for v in row.tolist()] for row in x[:take]]


def _live_friction(env) -> tuple[torch.Tensor, torch.Tensor]:
    term = env.unwrapped.event_manager.get_term_cfg("physics_material").func
    if hasattr(term, "env_static_friction") and hasattr(term, "env_dynamic_friction"):
        return term.env_static_friction.clone(), term.env_dynamic_friction.clone()

    robot = env.unwrapped.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()
    static = materials[:, :, 0].mean(dim=1, keepdim=True).to(env.unwrapped.device)
    dynamic = materials[:, :, 1].mean(dim=1, keepdim=True).to(env.unwrapped.device)
    return static, dynamic


def _live_base_mass_ratio(env) -> torch.Tensor:
    term = env.unwrapped.event_manager.get_term_cfg("add_base_mass").func
    if hasattr(term, "env_base_mass_ratio"):
        return term.env_base_mass_ratio.clone()

    robot = env.unwrapped.scene["robot"]
    masses = robot.root_physx_view.get_masses()
    default = robot.data.default_mass.cpu()
    ratio = masses / torch.clamp(default, min=1e-6)
    return ratio.mean(dim=1, keepdim=True).to(env.unwrapped.device)


def _live_motor_scales(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    stiffness_scale = torch.zeros_like(robot.data.default_joint_stiffness)
    damping_scale = torch.zeros_like(robot.data.default_joint_damping)
    for actuator in robot.actuators.values():
        joint_ids = actuator.joint_indices
        stiffness_scale[:, joint_ids] = actuator.stiffness / torch.clamp(
            robot.data.default_joint_stiffness[:, joint_ids],
            min=1e-6,
        )
        damping_scale[:, joint_ids] = actuator.damping / torch.clamp(
            robot.data.default_joint_damping[:, joint_ids],
            min=1e-6,
        )
    return stiffness_scale, damping_scale


def _collect_live_dynamics(env) -> dict[str, torch.Tensor]:
    static, dynamic = _live_friction(env)
    mass_ratio = _live_base_mass_ratio(env)
    stiffness, damping = _live_motor_scales(env)
    return {
        "static_friction": static,
        "dynamic_friction": dynamic,
        "base_mass_ratio": mass_ratio,
        "joint_stiffness_scale": stiffness,
        "joint_damping_scale": damping,
    }


def _scenario_counts(scenarios: torch.Tensor) -> dict[str, int]:
    mapping = {
        0: "ultra_low_friction",
        1: "very_heavy",
        2: "very_weak_motor",
    }
    return {mapping[idx]: int((scenarios == idx).sum().item()) for idx in mapping}


def _validate_post_switch(env) -> dict[str, float]:
    live = _collect_live_dynamics(env)
    target_static = env.unwrapped._switch_static_friction
    target_dynamic = env.unwrapped._switch_dynamic_friction
    target_mass = env.unwrapped._switch_mass_offset
    target_stiffness = env.unwrapped._switch_motor_stiffness_scale
    target_damping = env.unwrapped._switch_motor_damping_scale
    robot = env.unwrapped.scene["robot"]

    metrics: dict[str, float] = {}

    friction_ids = (~torch.isnan(target_static)).nonzero(as_tuple=False).squeeze(-1)
    if len(friction_ids) > 0:
        metrics["friction_static_max_abs_diff"] = float(
            (live["static_friction"][friction_ids, 0] - target_static[friction_ids]).abs().max().item()
        )
        metrics["friction_dynamic_max_abs_diff"] = float(
            (live["dynamic_friction"][friction_ids, 0] - target_dynamic[friction_ids]).abs().max().item()
        )
    else:
        metrics["friction_static_max_abs_diff"] = 0.0
        metrics["friction_dynamic_max_abs_diff"] = 0.0

    mass_ids = (~torch.isnan(target_mass)).nonzero(as_tuple=False).squeeze(-1)
    if len(mass_ids) > 0:
        default_mass_mean = robot.data.default_mass[mass_ids].mean(dim=1).to(env.unwrapped.device)
        expected_mass_ratio = (robot.data.default_mass[mass_ids].mean(dim=1).to(env.unwrapped.device) + target_mass[mass_ids]) / torch.clamp(
            default_mass_mean,
            min=1e-6,
        )
        metrics["mass_ratio_max_abs_diff"] = float(
            (live["base_mass_ratio"][mass_ids, 0] - expected_mass_ratio).abs().max().item()
        )
    else:
        metrics["mass_ratio_max_abs_diff"] = 0.0

    motor_ids = (~torch.isnan(target_stiffness)).nonzero(as_tuple=False).squeeze(-1)
    if len(motor_ids) > 0:
        metrics["motor_stiffness_scale_max_abs_diff"] = float(
            (live["joint_stiffness_scale"][motor_ids] - target_stiffness[motor_ids].unsqueeze(-1)).abs().max().item()
        )
        metrics["motor_damping_scale_max_abs_diff"] = float(
            (live["joint_damping_scale"][motor_ids] - target_damping[motor_ids].unsqueeze(-1)).abs().max().item()
        )
    else:
        metrics["motor_stiffness_scale_max_abs_diff"] = 0.0
        metrics["motor_damping_scale_max_abs_diff"] = 0.0

    return metrics


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    _make_debug_friendly(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset(seed=args_cli.seed)

        initial_targets = {
            "static_friction_nan_count": int(torch.isnan(env.unwrapped._switch_static_friction).sum().item()),
            "dynamic_friction_nan_count": int(torch.isnan(env.unwrapped._switch_dynamic_friction).sum().item()),
            "mass_offset_nan_count": int(torch.isnan(env.unwrapped._switch_mass_offset).sum().item()),
            "motor_stiffness_nan_count": int(torch.isnan(env.unwrapped._switch_motor_stiffness_scale).sum().item()),
            "motor_damping_nan_count": int(torch.isnan(env.unwrapped._switch_motor_damping_scale).sum().item()),
            "scenario_counts": _scenario_counts(env.unwrapped._switch_scenario.clone()),
        }

        num_envs = env.unwrapped.num_envs
        zero_action = torch.zeros(
            (num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        transition_counts = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)
        transition_steps = torch.full((num_envs,), -1, dtype=torch.long, device=env.unwrapped.device)
        prev_applied = env.unwrapped._switch_applied.clone()

        total_steps = args_cli.switch_step + args_cli.steps_after_switch
        for step_idx in range(total_steps):
            env.step(zero_action)
            current_applied = env.unwrapped._switch_applied.clone()
            newly_applied = current_applied & (~prev_applied)
            transition_counts += newly_applied.long()
            transition_steps[newly_applied] = step_idx + 1
            prev_applied = current_applied

        live_validation = _validate_post_switch(env)

        result = {
            "task": args_cli.task,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "switch_step": args_cli.switch_step,
            "steps_after_switch": args_cli.steps_after_switch,
            "initial_targets": initial_targets,
            "transition_summary": {
                "max_transition_count": int(transition_counts.max().item()),
                "min_transition_count": int(transition_counts.min().item()),
                "all_envs_switched_exactly_once": bool(torch.all(transition_counts == 1).item()),
                "unique_transition_steps": sorted({int(v) for v in transition_steps.tolist()}),
                "expected_transition_step": int(args_cli.switch_step),
                "sample_transition_steps": transition_steps[: min(args_cli.sample_envs, num_envs)].tolist(),
            },
            "post_switch_live_validation": live_validation,
            "sample_targets": {
                "scenario_ids": env.unwrapped._switch_scenario[: min(args_cli.sample_envs, num_envs)].tolist(),
                "static_friction": env.unwrapped._switch_static_friction[: min(args_cli.sample_envs, num_envs)].tolist(),
                "dynamic_friction": env.unwrapped._switch_dynamic_friction[: min(args_cli.sample_envs, num_envs)].tolist(),
                "mass_offset": env.unwrapped._switch_mass_offset[: min(args_cli.sample_envs, num_envs)].tolist(),
                "motor_stiffness_scale": env.unwrapped._switch_motor_stiffness_scale[: min(args_cli.sample_envs, num_envs)].tolist(),
                "motor_damping_scale": env.unwrapped._switch_motor_damping_scale[: min(args_cli.sample_envs, num_envs)].tolist(),
            },
            "sample_live_dynamics": {
                key: _tensor_rows(value, args_cli.sample_envs)
                for key, value in _collect_live_dynamics(env).items()
            },
        }

        print("\n=== Adaptation Switch Debug ===")
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
