"""Play an RMA checkpoint on isolated terrains with controlled randomization."""

from __future__ import annotations

import argparse
import json
import os
import time
import types

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play checkpoint with isolated terrain and controlled dynamics.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, default="RMA-Go2-Blind-Baseline-Rough-WarmStart", help="Registered task name.")
parser.add_argument("--terrain-type", type=str, default=None, help="Isolated terrain name (e.g. pyramid_stairs, boxes).")
parser.add_argument("--terrain-level", type=int, default=-1, help="Fixed terrain curriculum level. Use -1 for spread up to max_init_terrain_level.")
parser.add_argument(
    "--nominal-dynamics",
    action="store_true",
    default=False,
    help="Force nominal fixed dynamics instead of task randomization: friction 0.8/0.7, zero mass offset, unit actuator gains.",
)
LATENT_MODES = [
    "normal",
    "zero",
    "frozen",
    "shuffled",
    "no_terrain",
    "shuffled_terrain",
    "no_dynamics",
    "shuffled_dynamics",
]
parser.add_argument("--latent-mode", type=str, default="normal", choices=LATENT_MODES)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--steps", type=int, default=-1, help="Max steps. Use -1 to run until window closes.")
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match simulator dt.")
parser.add_argument("--teleop-keyboard", action="store_true", default=False, help="Enable keyboard teleop for base velocity commands.")
parser.add_argument(
    "--history-length",
    type=int,
    default=None,
    help="Optional override for deployable policy history length when the task env exposes one.",
)
parser.add_argument(
    "--command-profile",
    type=str,
    default="task",
    choices=["task", "standstill", "forward"],
    help="Override task-sampled commands with a fixed controller-focused command profile.",
)
parser.add_argument("--forced-lin-x", type=float, default=0.55, help="Forced x velocity when using --command-profile forward.")
parser.add_argument("--forced-lin-y", type=float, default=0.0, help="Forced y velocity when using --command-profile forward.")
parser.add_argument("--forced-ang-z", type=float, default=0.0, help="Forced yaw velocity when using --command-profile forward.")
parser.add_argument("--debug-live", action="store_true", default=False, help="Print live policy/runtime diagnostics while the viewer runs.")
parser.add_argument("--debug-every", type=int, default=30, help="Emit live debug every N steps when --debug-live is enabled.")
parser.add_argument("--debug-env-index", type=int, default=0, help="Environment index to summarize in live debug output.")
parser.add_argument("--debug-jsonl", type=str, default=None, help="Optional JSONL file path for live debug samples.")

# Optional deterministic overrides (min=max)
parser.add_argument("--static-friction", type=float, default=None, help="Set static friction to a fixed value.")
parser.add_argument("--dynamic-friction", type=float, default=None, help="Set dynamic friction to a fixed value.")
parser.add_argument("--mass-offset", type=float, default=None, help="Set base mass offset to a fixed value.")
parser.add_argument("--motor-stiffness-scale", type=float, default=None, help="Scale actuator stiffness.")
parser.add_argument("--motor-damping-scale", type=float, default=None, help="Scale actuator damping.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rma_go2_lab  # noqa: F401
import isaaclab_tasks  # noqa: F401


def _tensor_stats_1d(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "norm": float(x.norm().item()),
    }


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.shape != b.shape or a.numel() == 0:
        return None
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())


def _maybe_detach(obs, key: str):
    if key not in obs:
        return None
    value = obs[key]
    if isinstance(value, torch.Tensor):
        return value.detach()
    return None


def _reset_frozen_latent_cache(policy_nn, dones: torch.Tensor | None = None) -> None:
    cache = getattr(policy_nn, "_frozen_history_latent_cache", None)
    if cache is None:
        return
    if dones is None:
        policy_nn._frozen_history_latent_cache = None
        return
    if not isinstance(dones, torch.Tensor):
        policy_nn._frozen_history_latent_cache = None
        return
    done_mask = dones.to(device=cache.device).bool().view(-1)
    if done_mask.numel() != cache.shape[0]:
        policy_nn._frozen_history_latent_cache = None
        return
    if bool(done_mask.any().item()):
        cache = cache.clone()
        cache[done_mask] = torch.nan
        policy_nn._frozen_history_latent_cache = cache


def _install_frozen_latent_mode(policy_nn) -> None:
    if getattr(policy_nn, "_frozen_latent_mode_installed", False):
        return
    if not hasattr(policy_nn, "adapt_from_history"):
        raise RuntimeError("Frozen latent mode requires an adaptation policy with adapt_from_history().")

    original_adapt_from_history = policy_nn.adapt_from_history

    def _adapt_from_history_frozen(self, history_obs: torch.Tensor) -> torch.Tensor:
        latent = original_adapt_from_history(history_obs)
        if getattr(self, "latent_mode", "normal") != "frozen":
            return latent
        cache = getattr(self, "_frozen_history_latent_cache", None)
        if cache is None or cache.shape != latent.shape:
            cache = torch.full_like(latent, torch.nan)
        else:
            cache = cache.to(device=latent.device, dtype=latent.dtype)
        needs_fill = torch.isnan(cache).any(dim=-1)
        if bool(needs_fill.any().item()):
            cache = cache.clone()
            cache[needs_fill] = latent[needs_fill]
        self._frozen_history_latent_cache = cache
        return cache

    policy_nn._original_adapt_from_history = original_adapt_from_history
    policy_nn.adapt_from_history = types.MethodType(_adapt_from_history_frozen, policy_nn)
    policy_nn._frozen_history_latent_cache = None
    policy_nn._frozen_latent_mode_installed = True


def _policy_debug_snapshot(env, obs, policy_nn, actions: torch.Tensor, step_idx: int, env_index: int) -> dict[str, object]:
    env_index = int(max(0, min(env_index, env.num_envs - 1)))
    snapshot: dict[str, object] = {"step": int(step_idx), "env_index": env_index}

    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")
    root_lin_vel = robot.data.root_lin_vel_b[:, :2]
    root_ang_vel = robot.data.root_ang_vel_b[:, 2]
    posture_gravity_xy = robot.data.projected_gravity_b[:, :2]

    command_xy = command[:, :2]
    vel_err_xy = (root_lin_vel - command_xy).norm(dim=-1)
    yaw_err = (root_ang_vel - command[:, 2]).abs()

    snapshot["command"] = {
        "lin_x": float(command[env_index, 0].item()),
        "lin_y": float(command[env_index, 1].item()),
        "ang_z": float(command[env_index, 2].item()),
        "planar_speed_mean": float(command_xy.norm(dim=-1).mean().item()),
    }
    snapshot["tracking"] = {
        "planar_speed_mean": float(root_lin_vel.norm(dim=-1).mean().item()),
        "vel_err_xy_mean": float(vel_err_xy.mean().item()),
        "vel_err_xy_env": float(vel_err_xy[env_index].item()),
        "yaw_err_mean": float(yaw_err.mean().item()),
        "yaw_err_env": float(yaw_err[env_index].item()),
        "tilt_xy_mean": float(posture_gravity_xy.norm(dim=-1).mean().item()),
        "base_height_mean": float(robot.data.root_pos_w[:, 2].mean().item()),
    }
    snapshot["actions"] = {
        "env": actions[env_index].detach().cpu().tolist(),
        "summary": _tensor_stats_1d(actions.detach()),
    }

    if hasattr(policy_nn, "encode_history_latent") and "policy_history" in obs:
        history_latent = policy_nn.encode_history_latent(obs).detach()
        snapshot["phi_latent"] = {
            "summary": _tensor_stats_1d(history_latent),
            "env": history_latent[env_index].cpu().tolist(),
        }

        if hasattr(policy_nn, "encode_history_bottleneck"):
            try:
                bottleneck = policy_nn.encode_history_bottleneck(obs).detach()
                snapshot["phi_bottleneck"] = {
                    "summary": _tensor_stats_1d(bottleneck),
                    "env": bottleneck[env_index].cpu().tolist(),
                }
            except Exception as exc:  # defensive: not all variants expose a valid bottleneck path
                snapshot["phi_bottleneck_error"] = str(exc)

        if hasattr(policy_nn, "predict_history_dynamics"):
            try:
                predicted_dynamics = policy_nn.predict_history_dynamics(obs).detach()
                snapshot["phi_predicted_dynamics"] = {
                    "summary": _tensor_stats_1d(predicted_dynamics),
                    "env": predicted_dynamics[env_index].cpu().tolist(),
                }
                if "dynamics_privileged" in obs:
                    dynamics_priv = obs["dynamics_privileged"].detach()
                    dyn_err = (predicted_dynamics - dynamics_priv).abs()
                    snapshot["phi_dynamics_error"] = {
                        "mae_mean": float(dyn_err.mean().item()),
                        "mae_env": float(dyn_err[env_index].mean().item()),
                    }
            except Exception as exc:
                snapshot["phi_predicted_dynamics_error"] = str(exc)

    if hasattr(policy_nn, "encode_extrinsics_latent") and "dynamics_privileged" in obs:
        try:
            mu_latent = policy_nn.encode_extrinsics_latent(obs).detach()
            snapshot["mu_latent"] = {
                "summary": _tensor_stats_1d(mu_latent),
                "env": mu_latent[env_index].cpu().tolist(),
            }
            if "phi_latent" in snapshot:
                phi_latent = policy_nn.encode_history_latent(obs).detach()
                snapshot["mu_phi_compare"] = {
                    "cosine_mean": _cosine_mean(mu_latent, phi_latent),
                    "l2_mean": float((mu_latent - phi_latent).norm(dim=-1).mean().item()),
                    "l2_env": float((mu_latent[env_index] - phi_latent[env_index]).norm().item()),
                }
        except Exception as exc:
            snapshot["mu_latent_error"] = str(exc)

    policy_obs = _maybe_detach(obs, "policy")
    if hasattr(policy_nn, "act_with_latent") and policy_obs is not None and hasattr(policy_nn, "encode_history_latent"):
        try:
            phi_latent = policy_nn.encode_history_latent(obs).detach()
            action_mean = policy_nn.act_with_latent(policy_obs, phi_latent).detach()
            snapshot["pi_action_mean"] = {
                "summary": _tensor_stats_1d(action_mean),
                "env": action_mean[env_index].cpu().tolist(),
            }
        except Exception as exc:
            snapshot["pi_action_mean_error"] = str(exc)

    return snapshot


def _safe_load_runner(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    checkpoint = None
    try:
        runner.load(checkpoint_path)
        return
    except (RuntimeError, ValueError, KeyError) as exc:
        message = str(exc)
        if not (
            "normalizer" in message
            or "optimizer_state_dict" in message
            or "parameter group" in message
            or "size mismatch" in message
        ):
            raise
        print(f"[WARN] Standard runner.load() failed: {message}")
        print("[WARN] Retrying with model-state-only fallback load.")

    def _filter_compatible_state(module: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target_state = module.state_dict()
        compatible = {}
        for key, value in source_state.items():
            if key not in target_state:
                continue
            if target_state[key].shape != value.shape:
                continue
            compatible[key] = value
        return compatible

    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    model_state = checkpoint["model_state_dict"]
    filtered_state = {k: v for k, v in model_state.items() if "normalizer" not in k}
    compatible_state = _filter_compatible_state(runner.alg.policy, filtered_state)
    resumed = runner.alg.policy.load_state_dict(compatible_state, strict=False)
    if hasattr(resumed, "missing_keys") and hasattr(resumed, "unexpected_keys"):
        print(f"[INFO] Fallback policy load complete. Missing keys: {list(resumed.missing_keys)}")
        print(f"[INFO] Fallback policy load complete. Unexpected keys: {list(resumed.unexpected_keys)}")
    else:
        print(f"[INFO] Fallback policy load complete. Return type: {type(resumed).__name__}")


def _force_isolated_terrain(env_cfg, terrain_type: str | None) -> None:
    if terrain_type is None:
        return
    if terrain_type == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if hasattr(env_cfg.scene, "height_scanner"):
            env_cfg.scene.height_scanner = None
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
    if args_cli.terrain_type == "plane":
        if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
        return
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

def _apply_randomization_overrides(env_cfg) -> None:
    static_friction = 0.8 if args_cli.nominal_dynamics and args_cli.static_friction is None else args_cli.static_friction
    dynamic_friction = 0.7 if args_cli.nominal_dynamics and args_cli.dynamic_friction is None else args_cli.dynamic_friction
    mass_offset = 0.0 if args_cli.nominal_dynamics and args_cli.mass_offset is None else args_cli.mass_offset
    motor_stiffness_scale = 1.0 if args_cli.nominal_dynamics and args_cli.motor_stiffness_scale is None else args_cli.motor_stiffness_scale
    motor_damping_scale = 1.0 if args_cli.nominal_dynamics and args_cli.motor_damping_scale is None else args_cli.motor_damping_scale

    if (
        static_friction is not None
        and env_cfg.events.physics_material is not None
    ):
        env_cfg.events.physics_material.params["static_friction_range"] = (
            static_friction,
            static_friction,
        )
    if (
        dynamic_friction is not None
        and env_cfg.events.physics_material is not None
    ):
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (
            dynamic_friction,
            dynamic_friction,
        )
    if mass_offset is not None and env_cfg.events.add_base_mass is not None:
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (
            mass_offset,
            mass_offset,
        )
    if motor_stiffness_scale is not None and hasattr(env_cfg.events, "motor_strength"):
        env_cfg.events.motor_strength.params["stiffness_distribution_params"] = (
            motor_stiffness_scale,
            motor_stiffness_scale,
        )
    if motor_damping_scale is not None and hasattr(env_cfg.events, "motor_strength"):
        env_cfg.events.motor_strength.params["damping_distribution_params"] = (
            motor_damping_scale,
            motor_damping_scale,
        )


def _apply_history_length_override(env_cfg) -> None:
    if args_cli.history_length is None:
        return
    if hasattr(env_cfg, "adaptation_history_length"):
        env_cfg.adaptation_history_length = int(args_cli.history_length)
    if hasattr(env_cfg, "policy_history_length"):
        env_cfg.policy_history_length = int(args_cli.history_length)


def _set_forced_velocity_command(env, command: torch.Tensor) -> None:
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:, :] = command
    command_term.time_left[:] = 1.0e6
    command_term.command_counter[:] = 1
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "heading_target"):
        command_term.heading_target[:] = env.unwrapped.scene["robot"].data.heading_w


def _forced_command(env) -> torch.Tensor | None:
    if args_cli.command_profile == "task":
        return None
    robot = env.unwrapped.scene["robot"]
    command = torch.zeros((env.num_envs, 3), device=robot.device, dtype=robot.data.root_lin_vel_b.dtype)
    if args_cli.command_profile == "standstill":
        return command
    if args_cli.command_profile == "forward":
        command[:, 0] = args_cli.forced_lin_x
        command[:, 1] = args_cli.forced_lin_y
        command[:, 2] = args_cli.forced_ang_z
        return command
    raise ValueError(f"Unsupported command profile: {args_cli.command_profile}")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_level)
    _apply_randomization_overrides(env_cfg)
    _apply_history_length_override(env_cfg)

    agent_cfg_obj = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = agent_cfg_obj.to_dict()
    if "policy" in agent_cfg and isinstance(agent_cfg["policy"], dict):
        agent_cfg["policy"]["pretrained_path"] = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_obj.clip_actions)
    _force_terrain_level(env, args_cli.terrain_level)

    runner = OnPolicyRunner(env, agent_cfg, log_dir=os.path.dirname(args_cli.checkpoint), device=args_cli.device)
    _safe_load_runner(runner, args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy
    policy_nn.eval()
    if args_cli.latent_mode == "frozen":
        _install_frozen_latent_mode(policy_nn)
    if hasattr(policy_nn, "latent_mode"):
        policy_nn.latent_mode = args_cli.latent_mode

    print("=== Play Configuration ===")
    print(f"task={args_cli.task}")
    print(f"checkpoint={args_cli.checkpoint}")
    print(f"terrain_type={args_cli.terrain_type}")
    print(f"terrain_level={args_cli.terrain_level}")
    print(f"nominal_dynamics={args_cli.nominal_dynamics}")
    print(f"latent_mode={args_cli.latent_mode}")
    print(f"history_length={args_cli.history_length}")
    print(f"command_profile={args_cli.command_profile}")
    print(f"num_envs={args_cli.num_envs}, seed={args_cli.seed}")
    print(
        "overrides: "
        f"static_friction={args_cli.static_friction}, dynamic_friction={args_cli.dynamic_friction}, "
        f"mass_offset={args_cli.mass_offset}, stiffness_scale={args_cli.motor_stiffness_scale}, "
        f"damping_scale={args_cli.motor_damping_scale}"
    )
    print(f"debug_live={args_cli.debug_live}, debug_every={args_cli.debug_every}, debug_env_index={args_cli.debug_env_index}")
    if args_cli.debug_jsonl:
        print(f"debug_jsonl={args_cli.debug_jsonl}")

    teleop = None
    if args_cli.teleop_keyboard:
        teleop = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=0.5,
                omega_z_sensitivity=1.0,
                sim_device=env.unwrapped.device,
            )
        )
        teleop.reset()
        print(teleop)

    env.reset()
    _force_terrain_level(env, args_cli.terrain_level)
    forced_command = _forced_command(env)
    if forced_command is not None:
        _set_forced_velocity_command(env, forced_command)
    _reset_frozen_latent_cache(policy_nn)
    obs = env.get_observations()
    debug_stream = None
    if args_cli.debug_jsonl:
        os.makedirs(os.path.dirname(args_cli.debug_jsonl), exist_ok=True) if os.path.dirname(args_cli.debug_jsonl) else None
        debug_stream = open(args_cli.debug_jsonl, "a", encoding="utf-8")
    dt = env.unwrapped.step_dt
    step_count = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if teleop is not None:
                cmd = teleop.advance().view(1, 3)
                base_command = env.unwrapped.command_manager.get_command("base_velocity")
                base_command[:, 0] = cmd[:, 0]
                base_command[:, 1] = cmd[:, 1]
                base_command[:, 2] = cmd[:, 2]
            elif forced_command is not None:
                _set_forced_velocity_command(env, forced_command)
            actions = policy(obs)
            if args_cli.debug_live and args_cli.debug_every > 0 and (step_count % args_cli.debug_every == 0):
                snapshot = _policy_debug_snapshot(env, obs, policy_nn, actions, step_count, args_cli.debug_env_index)
                print(json.dumps(snapshot), flush=True)
                if debug_stream is not None:
                    debug_stream.write(json.dumps(snapshot) + "\n")
                    debug_stream.flush()
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            _reset_frozen_latent_cache(policy_nn, dones)
            if isinstance(dones, torch.Tensor) and bool(dones.any().item()):
                env.reset()
                _force_terrain_level(env, args_cli.terrain_level)
                if forced_command is not None:
                    _set_forced_velocity_command(env, forced_command)
                _reset_frozen_latent_cache(policy_nn, dones=None)
                obs = env.get_observations()

        step_count += 1
        if args_cli.steps > 0 and step_count >= args_cli.steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if debug_stream is not None:
        debug_stream.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
