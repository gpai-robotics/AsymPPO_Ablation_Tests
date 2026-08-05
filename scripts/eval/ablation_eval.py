"""Play any registered RSL-RL policy checkpoint.

Use this for history/asymmetric-PPO policies. The older ``play_flat_prior.py``
is flat-expert specific and should not be used for rough history checkpoints.
"""

from __future__ import annotations

import argparse
import json
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a registered RSL-RL policy checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, required=True, help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=1200, help="Recorded video length in env steps.")
parser.add_argument(
    "--eval-json-out",
    type=str,
    default="",
    help="Optional path to write rollout metrics as JSON when playback exits.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop playback after this many env steps. Use 0 for continuous playback.",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "artifacts/evaluations/clips/play_policy"),
    help="Folder used when --video is enabled.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument("--teleop-keyboard", action="store_true", default=False, help="Override base velocity commands from keyboard.")
parser.add_argument(
    "--teleop-lin-slew",
    type=float,
    default=8.0,
    help="Max linear command change in m/s^2 for keyboard teleop. Higher is more responsive; use <=0 to disable.",
)
parser.add_argument(
    "--teleop-yaw-slew",
    type=float,
    default=12.0,
    help="Max yaw-rate command change in rad/s^2 for keyboard teleop. Higher is more responsive; use <=0 to disable.",
)
parser.add_argument(
    "--print-env-info",
    action="store_true",
    default=False,
    help="Print terrain and realized randomization info during playback.",
)
parser.add_argument(
    "--env-info-interval",
    type=int,
    default=250,
    help="Playback steps between --print-env-info reports.",
)
parser.add_argument(
    "--nominal-env",
    action="store_true",
    default=False,
    help="Use normal deterministic Go2 playback: no pushes, nominal friction, zero added mass/COM, nominal gains.",
)
parser.add_argument(
    "--terrain-type",
    type=str,
    default=None,
    help=(
        "Force playback onto one terrain family/preset. Examples: plane, random_rough, boxes, "
        "pyramid_stairs, pyramid_stairs_inv, hf_pyramid_slope, hf_pyramid_slope_inv, mixed_all, mixed_geom_ood."
    ),
)
parser.add_argument(
    "--terrain-level",
    type=int,
    default=-1,
    help="Fixed terrain curriculum level. Use -1 to keep the task/default terrain spread.",
)
parser.add_argument("--step-height", type=float, default=None, help="Fixed stair step height. Applies to stair terrains.")
parser.add_argument("--step-height-min", type=float, default=None, help="Minimum stair step height.")
parser.add_argument("--step-height-max", type=float, default=None, help="Maximum stair step height.")
parser.add_argument("--step-width", type=float, default=None, help="Fixed stair step width.")
parser.add_argument("--step-width-min", type=float, default=None, help="Fixed stair step width if equal to --step-width-max.")
parser.add_argument("--step-width-max", type=float, default=None, help="Fixed stair step width if equal to --step-width-min.")
parser.add_argument("--platform-width", type=float, default=None, help="Fixed stair/obstacle platform width.")
parser.add_argument("--platform-width-min", type=float, default=None, help="Fixed platform width if equal to --platform-width-max.")
parser.add_argument("--platform-width-max", type=float, default=None, help="Fixed platform width if equal to --platform-width-min.")
parser.add_argument("--num-steps", type=int, default=None, help="Requested stair count. Warns if terrain backend cannot set it directly.")
parser.add_argument("--roughness-amplitude", type=float, default=None, help="Fixed random_rough height amplitude.")
parser.add_argument("--roughness-frequency", type=float, default=None, help="Requested roughness frequency. Warns if unsupported by the terrain backend.")
parser.add_argument("--roughness-scale", type=float, default=None, help="Requested roughness scale. Warns if unsupported by the terrain backend.")
parser.add_argument("--roughness-noise", type=float, default=None, help="Set random_rough noise_step if available.")
parser.add_argument("--box-height", type=float, default=None, help="Fixed boxes/random-grid height.")
parser.add_argument("--box-height-min", type=float, default=None, help="Minimum boxes/random-grid height.")
parser.add_argument("--box-height-max", type=float, default=None, help="Maximum boxes/random-grid height.")
parser.add_argument("--box-width", type=float, default=None, help="Set boxes grid_width if available.")
parser.add_argument("--box-spacing", type=float, default=None, help="Requested box spacing. Warns if unsupported by the terrain backend.")
parser.add_argument("--obstacle-density", type=float, default=None, help="Requested obstacle density. Warns if unsupported by the terrain backend.")
parser.add_argument("--slope-angle", type=float, default=None, help="Fixed slope coefficient for slope terrains.")
parser.add_argument("--static-friction", type=float, default=None, help="Fixed static friction.")
parser.add_argument("--static-friction-min", type=float, default=None, help="Minimum static friction.")
parser.add_argument("--static-friction-max", type=float, default=None, help="Maximum static friction.")
parser.add_argument("--dynamic-friction", type=float, default=None, help="Fixed dynamic friction.")
parser.add_argument("--dynamic-friction-min", type=float, default=None, help="Minimum dynamic friction.")
parser.add_argument("--dynamic-friction-max", type=float, default=None, help="Maximum dynamic friction.")
parser.add_argument("--restitution", type=float, default=None, help="Fixed restitution.")
parser.add_argument("--restitution-min", type=float, default=None, help="Minimum restitution.")
parser.add_argument("--restitution-max", type=float, default=None, help="Maximum restitution.")
parser.add_argument("--added-mass", type=float, default=None, help="Fixed additive base mass offset in kg.")
parser.add_argument("--com-x", type=float, default=None, help="Fixed base COM x offset.")
parser.add_argument("--com-y", type=float, default=None, help="Fixed base COM y offset.")
parser.add_argument("--com-z", type=float, default=None, help="Fixed base COM z offset.")
parser.add_argument("--motor-stiffness-scale", type=float, default=None, help="Fixed global actuator stiffness scale.")
parser.add_argument("--motor-damping-scale", type=float, default=None, help="Fixed global actuator damping scale.")
parser.add_argument("--hip-stiffness", type=float, default=None, help="Requested hip stiffness scale. Warns unless a matching event exists.")
parser.add_argument("--thigh-stiffness", type=float, default=None, help="Requested thigh stiffness scale. Warns unless a matching event exists.")
parser.add_argument("--calf-stiffness", type=float, default=None, help="Requested calf stiffness scale. Warns unless a matching event exists.")
parser.add_argument("--hip-damping", type=float, default=None, help="Requested hip damping scale. Warns unless a matching event exists.")
parser.add_argument("--thigh-damping", type=float, default=None, help="Requested thigh damping scale. Warns unless a matching event exists.")
parser.add_argument("--calf-damping", type=float, default=None, help="Requested calf damping scale. Warns unless a matching event exists.")
parser.add_argument("--disable-pushes", action="store_true", default=False, help="Disable interval push disturbance events.")
parser.add_argument("--push-force", type=float, default=None, help="Requested force push. Warns for velocity-push envs.")
parser.add_argument("--push-torque", type=float, default=None, help="Requested torque push. Warns for velocity-push envs.")
parser.add_argument("--push-interval", type=float, default=None, help="Fixed push interval in seconds.")
parser.add_argument("--push-velocity-x", type=float, default=None, help="Fixed velocity push range magnitude on x.")
parser.add_argument("--push-velocity-y", type=float, default=None, help="Fixed velocity push range magnitude on y.")
parser.add_argument("--push-velocity-yaw", type=float, default=None, help="Fixed velocity push range magnitude on yaw.")
parser.add_argument("--obs-delay", type=int, default=None, help="Requested observation delay in policy steps. Warns if unsupported.")
parser.add_argument("--action-delay", type=int, default=None, help="Requested action delay in policy steps. Warns if unsupported.")
parser.add_argument("--obs-noise-scale", type=float, default=None, help="Scale all configured additive observation noise ranges.")
parser.add_argument("--imu-noise", type=float, default=None, help="Scale IMU-like base angular velocity noise.")
parser.add_argument("--joint-position-noise", type=float, default=None, help="Scale joint position observation noise.")
parser.add_argument("--joint-velocity-noise", type=float, default=None, help="Scale joint velocity observation noise.")
parser.add_argument("--gravity-noise", type=float, default=None, help="Scale projected gravity observation noise.")
parser.add_argument("--height-scanner-noise", type=float, default=None, help="Scale height scan observation noise.")
parser.add_argument("--fixed-command", action="store_true", default=False, help="Force commands to --cmd-vx/--cmd-vy/--cmd-yaw during rollout.")
parser.add_argument("--cmd-vx", type=float, default=None, help="Fixed command vx. Enables fixed command when any cmd component is set.")
parser.add_argument("--cmd-vy", type=float, default=None, help="Fixed command vy. Enables fixed command when any cmd component is set.")
parser.add_argument("--cmd-yaw", type=float, default=None, help="Fixed command yaw rate. Enables fixed command when any cmd component is set.")
parser.add_argument("--spawn-height", type=float, default=None, help="Fixed reset spawn z offset if reset_base supports pose_range z.")
parser.add_argument("--spawn-roll", type=float, default=None, help="Fixed reset spawn roll if reset_base supports pose_range roll.")
parser.add_argument("--spawn-pitch", type=float, default=None, help="Fixed reset spawn pitch if reset_base supports pose_range pitch.")
parser.add_argument("--spawn-yaw", type=float, default=None, help="Fixed reset spawn yaw.")
parser.add_argument("--disable-friction-randomization", action="store_true", default=False, help="Set friction ranges to nominal fixed values.")
parser.add_argument("--disable-com-randomization", action="store_true", default=False, help="Set COM offsets to zero.")
parser.add_argument("--disable-motor-randomization", action="store_true", default=False, help="Set actuator gain scales to 1.")
parser.add_argument("--disable-mass-randomization", action="store_true", default=False, help="Set added mass range to zero.")
parser.add_argument(
    "--deterministic-env",
    action="store_true",
    default=False,
    help="Collapse supported randomization ranges to deterministic midpoints.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not str(args_cli.checkpoint).strip():
    parser.error("--checkpoint is empty. Set ASYMPPO_CKPT or pass an absolute checkpoint path.")
checkpoint_path = Path(args_cli.checkpoint).expanduser()
if not checkpoint_path.is_file():
    parser.error(f"--checkpoint does not exist or is not a file: {checkpoint_path}")
args_cli.checkpoint = str(checkpoint_path)

if args_cli.video:
    args_cli.enable_cameras = True

print("[PLAY] Launching Isaac Sim app...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PLAY] Isaac Sim app launched.", flush=True)

import gymnasium as gym
import torch
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


def _safe_load_runner(runner: OnPolicyRunner | DistillationRunner, checkpoint_path: str) -> None:
    try:
        runner.load(checkpoint_path)
        return
    except RuntimeError as exc:
        message = str(exc)
        if "normalizer" not in message:
            raise
        print("[WARN] Standard runner.load() failed due to normalizer-key mismatch.")
        print("[WARN] Retrying with checkpoint normalizer entries filtered out for playback.")

    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    model_state = checkpoint["model_state_dict"]
    filtered_state = {key: value for key, value in model_state.items() if "normalizer" not in key}
    resumed = runner.alg.policy.load_state_dict(filtered_state, strict=False)
    print(f"[INFO] Fallback policy load complete. Missing keys: {list(resumed.missing_keys)}")
    print(f"[INFO] Fallback policy load complete. Unexpected keys: {list(resumed.unexpected_keys)}")


def _make_runner(env, runner_cfg):
    if runner_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    if runner_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    raise RuntimeError(f"Unsupported runner class: {runner_cfg.class_name}")


def _tensor_stats(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float().flatten()
    if tensor.numel() == 0:
        return "empty"
    return (
        f"mean={tensor.mean().item():.3f} "
        f"min={tensor.min().item():.3f} "
        f"max={tensor.max().item():.3f}"
    )


def _scalar_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _write_eval_json(
    path: str,
    *,
    args_cli,
    timestep: int,
    applied_overrides: list[str],
    override_warnings: list[str],
    metrics: dict[str, list[float]],
    reset_count: int,
) -> None:
    if not path:
        return
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": timestep,
        "terrain_type": args_cli.terrain_type,
        "terrain_level": args_cli.terrain_level,
        "nominal_env": args_cli.nominal_env,
        "fixed_command": {
            "enabled": bool(
                args_cli.fixed_command
                or any(value is not None for value in (args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_yaw))
            ),
            "vx": args_cli.cmd_vx,
            "vy": args_cli.cmd_vy,
            "yaw": args_cli.cmd_yaw,
        },
        "applied_overrides": applied_overrides,
        "warnings": override_warnings,
        "reset_count": reset_count,
        "metrics": {name: _scalar_summary(values) for name, values in metrics.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[PLAY] Wrote eval JSON: {output_path}", flush=True)


def _get_event_func(env, term_name: str):
    try:
        return env.event_manager.get_term_cfg(term_name).func
    except Exception:
        return None


def _get_event_cfg(env, term_name: str):
    try:
        return env.event_manager.get_term_cfg(term_name)
    except Exception:
        return None


def _terrain_info(env) -> list[str]:
    lines = []
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return ["terrain: unavailable"]

    levels = getattr(terrain, "terrain_levels", None)
    types = getattr(terrain, "terrain_types", None)
    terrain_gen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_names = list(getattr(terrain_gen, "sub_terrains", {}).keys()) if terrain_gen is not None else []

    if levels is not None:
        lines.append(f"terrain_level: {_tensor_stats(levels)} env0={int(levels[0].item())}")
    else:
        lines.append("terrain_level: unavailable")

    if types is not None:
        env0_type_id = int(types[0].item())
        env0_type_name = terrain_names[env0_type_id] if 0 <= env0_type_id < len(terrain_names) else str(env0_type_id)
        unique_ids, counts = torch.unique(types.detach().cpu(), return_counts=True)
        counts_by_type = []
        for type_id, count in zip(unique_ids.tolist(), counts.tolist()):
            name = terrain_names[type_id] if 0 <= int(type_id) < len(terrain_names) else str(type_id)
            counts_by_type.append(f"{name}:{count}")
        lines.append(f"terrain_type: env0={env0_type_name} counts={', '.join(counts_by_type)}")
    else:
        terrain_type = getattr(getattr(terrain, "cfg", None), "terrain_type", None)
        lines.append(f"terrain_type: {terrain_type or 'unavailable'}")

    return lines


def _randomization_info(env) -> list[str]:
    lines = []

    physics_material = _get_event_func(env, "physics_material")
    physics_material_cfg = _get_event_cfg(env, "physics_material")
    static_friction = getattr(physics_material, "env_static_friction", None)
    dynamic_friction = getattr(physics_material, "env_dynamic_friction", None)
    if static_friction is not None:
        lines.append(f"static_friction: {_tensor_stats(static_friction)} env0={float(static_friction[0].item()):.3f}")
    if dynamic_friction is not None:
        lines.append(f"dynamic_friction: {_tensor_stats(dynamic_friction)} env0={float(dynamic_friction[0].item()):.3f}")
    if physics_material_cfg is not None:
        restitution_range = physics_material_cfg.params.get("restitution_range")
        if restitution_range is not None:
            lines.append(f"restitution_range_cfg: {restitution_range}")

    add_base_mass = _get_event_func(env, "add_base_mass")
    add_base_mass_cfg = _get_event_cfg(env, "add_base_mass")
    base_mass_ratio = getattr(add_base_mass, "env_base_mass_ratio", None)
    if base_mass_ratio is not None:
        lines.append(f"base_mass_ratio: {_tensor_stats(base_mass_ratio)} env0={float(base_mass_ratio[0].item()):.3f}")
    if add_base_mass_cfg is not None:
        lines.append(f"added_mass_range_cfg: {add_base_mass_cfg.params.get('mass_distribution_params')}")

    base_com_cfg = _get_event_cfg(env, "base_com")
    if base_com_cfg is not None:
        lines.append(f"base_com_range_cfg: {base_com_cfg.params.get('com_range')}")

    push_cfg = _get_event_cfg(env, "push_robot")
    if push_cfg is not None:
        lines.append(f"push_interval_cfg: {getattr(push_cfg, 'interval_range_s', None)}")
        lines.append(f"push_velocity_range_cfg: {push_cfg.params.get('velocity_range')}")
    else:
        lines.append("push_robot: disabled/unavailable")

    try:
        robot = env.scene["robot"]
        stiffness_scales = torch.zeros_like(robot.data.default_joint_stiffness)
        damping_scales = torch.zeros_like(robot.data.default_joint_damping)
        for actuator in robot.actuators.values():
            joint_ids = actuator.joint_indices
            stiffness_scales[:, joint_ids] = actuator.stiffness / torch.clamp(
                robot.data.default_joint_stiffness[:, joint_ids],
                min=1e-6,
            )
            damping_scales[:, joint_ids] = actuator.damping / torch.clamp(
                robot.data.default_joint_damping[:, joint_ids],
                min=1e-6,
            )
        lines.append(f"joint_stiffness_scale: {_tensor_stats(stiffness_scales)} env0_mean={stiffness_scales[0].mean().item():.3f}")
        lines.append(f"joint_damping_scale: {_tensor_stats(damping_scales)} env0_mean={damping_scales[0].mean().item():.3f}")
    except Exception:
        lines.append("joint_gain_scales: unavailable")

    if not lines:
        lines.append("randomization: no tracked values available")
    return lines


def _print_env_info(env, step: int) -> None:
    print(f"\n[ENV-INFO] step={step}")
    for line in _terrain_info(env):
        print(f"[ENV-INFO] {line}")
    for line in _randomization_info(env):
        print(f"[ENV-INFO] {line}")


def _event(events, name: str):
    term = getattr(events, name, None)
    return term if term is not None else None


def _set_param(term, key: str, value) -> None:
    if term is not None and key in term.params:
        term.params[key] = value


def _range_from_fixed_or_min_max(name: str, fixed, min_value, max_value, warnings: list[str]):
    if fixed is not None:
        return (float(fixed), float(fixed))
    if min_value is None and max_value is None:
        return None
    if min_value is None or max_value is None:
        warnings.append(f"{name}: both min and max must be provided; override ignored.")
        return None
    if float(min_value) > float(max_value):
        raise ValueError(f"{name}: min must be <= max.")
    return (float(min_value), float(max_value))


def _fixed_from_scalar_or_equal_range(name: str, fixed, min_value, max_value, warnings: list[str]):
    if fixed is not None:
        return float(fixed)
    if min_value is None and max_value is None:
        return None
    if min_value is None or max_value is None:
        warnings.append(f"{name}: both min and max must be provided; override ignored.")
        return None
    if float(min_value) != float(max_value):
        warnings.append(f"{name}: this terrain backend accepts a scalar, not a range; override ignored.")
        return None
    return float(min_value)


def _midpoint_range(value):
    if isinstance(value, tuple) and len(value) == 2:
        midpoint = 0.5 * (float(value[0]) + float(value[1]))
        return (midpoint, midpoint)
    return value


def _record(applied: list[str], key: str, value) -> None:
    applied.append(f"{key}: {value}")


def _terrain_generator(env_cfg):
    return getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)


def _terrain_subconfigs(env_cfg, names: tuple[str, ...] | None = None):
    terrain_gen = _terrain_generator(env_cfg)
    if terrain_gen is None:
        return []
    sub_terrains = getattr(terrain_gen, "sub_terrains", {})
    if names is None:
        return list(sub_terrains.items())
    return [(name, sub_terrains[name]) for name in names if name in sub_terrains]


def _set_subterrain_attr(env_cfg, terrain_names: tuple[str, ...], attr: str, value, applied: list[str], warnings: list[str]) -> None:
    matched = False
    for name, cfg in _terrain_subconfigs(env_cfg, terrain_names):
        if hasattr(cfg, attr):
            setattr(cfg, attr, value)
            _record(applied, f"terrain.{name}.{attr}", value)
            matched = True
    if not matched:
        warnings.append(f"terrain.{attr}: no selected terrain config exposes this field.")


def _set_material_value(env_cfg, key: str, value_range, applied: list[str], warnings: list[str]) -> None:
    if value_range is None:
        return
    events = getattr(env_cfg, "events", None)
    physics_material = _event(events, "physics_material") if events is not None else None
    if physics_material is not None and key in physics_material.params:
        physics_material.params[key] = value_range
        _record(applied, f"events.physics_material.{key}", value_range)
    else:
        warnings.append(f"events.physics_material.{key}: unsupported by this task.")

    material_key = key.replace("_range", "")
    terrain_material = getattr(getattr(getattr(env_cfg.scene, "terrain", None), "physics_material", None), material_key, None)
    if terrain_material is not None:
        setattr(env_cfg.scene.terrain.physics_material, material_key, float(value_range[0]))
        _record(applied, f"scene.terrain.physics_material.{material_key}", float(value_range[0]))


def _set_command_ranges(env_cfg, args_cli, applied: list[str]) -> None:
    fixed_command_requested = args_cli.fixed_command or any(
        value is not None for value in (args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_yaw)
    )
    if not fixed_command_requested:
        return

    cmd = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if cmd is None:
        return
    vx = 0.0 if args_cli.cmd_vx is None else float(args_cli.cmd_vx)
    vy = 0.0 if args_cli.cmd_vy is None else float(args_cli.cmd_vy)
    yaw = 0.0 if args_cli.cmd_yaw is None else float(args_cli.cmd_yaw)
    cmd.heading_command = False
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (vy, vy)
    cmd.ranges.ang_vel_z = (yaw, yaw)
    if hasattr(cmd.ranges, "heading"):
        cmd.ranges.heading = (0.0, 0.0)
    if hasattr(cmd, "limit_ranges"):
        cmd.limit_ranges.lin_vel_x = (vx, vx)
        cmd.limit_ranges.lin_vel_y = (vy, vy)
        cmd.limit_ranges.ang_vel_z = (yaw, yaw)
    _record(applied, "fixed_command", (vx, vy, yaw))


def _scale_observation_noise(env_cfg, args_cli, applied: list[str], warnings: list[str]) -> None:
    scale_by_term = {
        "base_ang_vel": args_cli.imu_noise,
        "projected_gravity": args_cli.gravity_noise,
        "joint_pos": args_cli.joint_position_noise,
        "joint_vel": args_cli.joint_velocity_noise,
        "height_scan": args_cli.height_scanner_noise,
    }
    global_scale = args_cli.obs_noise_scale
    if global_scale is None and all(value is None for value in scale_by_term.values()):
        return

    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        warnings.append("observation noise: task has no observations config.")
        return

    changed = 0
    for group_name, group_cfg in vars(observations).items():
        if group_name.startswith("_") or group_cfg is None:
            continue
        for term_name, term_cfg in vars(group_cfg).items():
            if term_name.startswith("_") or term_cfg is None:
                continue
            noise = getattr(term_cfg, "noise", None)
            if noise is None or not hasattr(noise, "n_min") or not hasattr(noise, "n_max"):
                continue
            scale = scale_by_term.get(term_name, None)
            if scale is None:
                scale = global_scale
            if scale is None:
                continue
            noise.n_min *= float(scale)
            noise.n_max *= float(scale)
            changed += 1
            _record(applied, f"observations.{group_name}.{term_name}.noise_scale", float(scale))

    if changed == 0:
        warnings.append("observation noise: no additive uniform noise terms were found to scale.")


def _apply_nominal_env_overrides(env_cfg) -> None:
    events = getattr(env_cfg, "events", None)
    if events is not None:
        # Disable active disturbances, but keep reset/randomization event structure intact.
        if hasattr(events, "push_robot"):
            events.push_robot = None

        physics_material = _event(events, "physics_material")
        _set_param(physics_material, "static_friction_range", (1.0, 1.0))
        _set_param(physics_material, "dynamic_friction_range", (1.0, 1.0))
        _set_param(physics_material, "restitution_range", (0.0, 0.0))

        base_external_force_torque = _event(events, "base_external_force_torque")
        _set_param(base_external_force_torque, "force_range", (0.0, 0.0))
        _set_param(base_external_force_torque, "torque_range", (0.0, 0.0))

        add_base_mass = _event(events, "add_base_mass")
        _set_param(add_base_mass, "mass_distribution_params", (0.0, 0.0))

        base_com = _event(events, "base_com")
        _set_param(base_com, "com_range", {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)})

        for name in ("motor_strength", "motor_strength_hip_thigh", "motor_strength_calf"):
            motor_strength = _event(events, name)
            _set_param(motor_strength, "stiffness_distribution_params", (1.0, 1.0))
            _set_param(motor_strength, "damping_distribution_params", (1.0, 1.0))


def _apply_cli_environment_overrides(env_cfg, args_cli) -> tuple[list[str], list[str]]:
    """Apply all playback/evaluation environment overrides in one place."""
    applied: list[str] = []
    warnings: list[str] = []

    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    if args_cli.terrain_type is not None:
        _record(applied, "terrain_type", args_cli.terrain_type)

    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_type, args_cli.terrain_level)
    if args_cli.terrain_level >= 0:
        _record(applied, "terrain_level", args_cli.terrain_level)

    if args_cli.nominal_env:
        _apply_nominal_env_overrides(env_cfg)
        _record(applied, "nominal_env", "pushes disabled; friction=1.0; restitution=0.0; mass/COM/gains nominal")

    step_height_range = _range_from_fixed_or_min_max(
        "step_height", args_cli.step_height, args_cli.step_height_min, args_cli.step_height_max, warnings
    )
    if step_height_range is not None:
        _set_subterrain_attr(env_cfg, ("pyramid_stairs", "pyramid_stairs_inv"), "step_height_range", step_height_range, applied, warnings)

    step_width = _fixed_from_scalar_or_equal_range(
        "step_width", args_cli.step_width, args_cli.step_width_min, args_cli.step_width_max, warnings
    )
    if step_width is not None:
        _set_subterrain_attr(env_cfg, ("pyramid_stairs", "pyramid_stairs_inv"), "step_width", step_width, applied, warnings)

    platform_width = _fixed_from_scalar_or_equal_range(
        "platform_width", args_cli.platform_width, args_cli.platform_width_min, args_cli.platform_width_max, warnings
    )
    if platform_width is not None:
        _set_subterrain_attr(env_cfg, ("pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"), "platform_width", platform_width, applied, warnings)

    if args_cli.num_steps is not None:
        warnings.append("num_steps: IsaacLab pyramid-stair terrain derives step count from size/step_width/platform_width; no direct override exists.")

    if args_cli.roughness_amplitude is not None:
        _set_subterrain_attr(
            env_cfg,
            ("random_rough",),
            "noise_range",
            (-abs(float(args_cli.roughness_amplitude)), abs(float(args_cli.roughness_amplitude))),
            applied,
            warnings,
        )
    if args_cli.roughness_noise is not None:
        _set_subterrain_attr(env_cfg, ("random_rough",), "noise_step", float(args_cli.roughness_noise), applied, warnings)
    if args_cli.roughness_frequency is not None:
        warnings.append("roughness_frequency: random_rough terrain has no frequency field in the active IsaacLab generator.")
    if args_cli.roughness_scale is not None:
        warnings.append("roughness_scale: active terrain config has no generic roughness scale field.")

    box_height_range = _range_from_fixed_or_min_max(
        "box_height", args_cli.box_height, args_cli.box_height_min, args_cli.box_height_max, warnings
    )
    if box_height_range is not None:
        _set_subterrain_attr(env_cfg, ("boxes",), "grid_height_range", box_height_range, applied, warnings)
    if args_cli.box_width is not None:
        _set_subterrain_attr(env_cfg, ("boxes",), "grid_width", float(args_cli.box_width), applied, warnings)
    if args_cli.box_spacing is not None:
        warnings.append("box_spacing: MeshRandomGridTerrainCfg exposes grid_width, not independent spacing.")
    if args_cli.obstacle_density is not None:
        warnings.append("obstacle_density: active boxes terrain does not expose a density parameter.")

    if args_cli.slope_angle is not None:
        slope_range = (abs(float(args_cli.slope_angle)), abs(float(args_cli.slope_angle)))
        _set_subterrain_attr(env_cfg, ("hf_pyramid_slope", "hf_pyramid_slope_inv"), "slope_range", slope_range, applied, warnings)

    if args_cli.deterministic_env:
        for terrain_name, terrain_cfg in _terrain_subconfigs(env_cfg):
            for attr in ("step_height_range", "noise_range", "grid_height_range", "slope_range", "box_height_range"):
                if hasattr(terrain_cfg, attr):
                    collapsed = _midpoint_range(getattr(terrain_cfg, attr))
                    setattr(terrain_cfg, attr, collapsed)
                    _record(applied, f"deterministic.terrain.{terrain_name}.{attr}", collapsed)

    static_friction = _range_from_fixed_or_min_max(
        "static_friction", args_cli.static_friction, args_cli.static_friction_min, args_cli.static_friction_max, warnings
    )
    dynamic_friction = _range_from_fixed_or_min_max(
        "dynamic_friction", args_cli.dynamic_friction, args_cli.dynamic_friction_min, args_cli.dynamic_friction_max, warnings
    )
    restitution = _range_from_fixed_or_min_max(
        "restitution", args_cli.restitution, args_cli.restitution_min, args_cli.restitution_max, warnings
    )
    if args_cli.disable_friction_randomization:
        static_friction = (1.0, 1.0)
        dynamic_friction = (1.0, 1.0)
        _record(applied, "disable_friction_randomization", "static/dynamic friction fixed to 1.0")
    _set_material_value(env_cfg, "static_friction_range", static_friction, applied, warnings)
    _set_material_value(env_cfg, "dynamic_friction_range", dynamic_friction, applied, warnings)
    _set_material_value(env_cfg, "restitution_range", restitution, applied, warnings)

    events = getattr(env_cfg, "events", None)
    if events is not None:
        if args_cli.disable_pushes and hasattr(events, "push_robot"):
            events.push_robot = None
            _record(applied, "events.push_robot", "disabled")

        push_robot = _event(events, "push_robot")
        if push_robot is not None:
            if args_cli.push_interval is not None:
                push_robot.interval_range_s = (float(args_cli.push_interval), float(args_cli.push_interval))
                _record(applied, "events.push_robot.interval_range_s", push_robot.interval_range_s)
            velocity_range = push_robot.params.get("velocity_range")
            if isinstance(velocity_range, dict):
                for arg_name, axis in (("push_velocity_x", "x"), ("push_velocity_y", "y"), ("push_velocity_yaw", "yaw")):
                    value = getattr(args_cli, arg_name)
                    if value is not None:
                        velocity_range[axis] = (-abs(float(value)), abs(float(value)))
                        _record(applied, f"events.push_robot.velocity_range.{axis}", velocity_range[axis])
        elif any(getattr(args_cli, name) is not None for name in ("push_interval", "push_velocity_x", "push_velocity_y", "push_velocity_yaw")):
            warnings.append("push_robot: requested push override, but this task has no active push_robot event.")

        if args_cli.push_force is not None or args_cli.push_torque is not None:
            warnings.append("push_force/push_torque: active Go2 tasks use velocity pushes, not force/torque push impulses.")

        add_base_mass = _event(events, "add_base_mass")
        mass_range = None
        if args_cli.disable_mass_randomization:
            mass_range = (0.0, 0.0)
        elif args_cli.added_mass is not None:
            mass_range = (float(args_cli.added_mass), float(args_cli.added_mass))
        if mass_range is not None:
            _set_param(add_base_mass, "mass_distribution_params", mass_range)
            _record(applied, "events.add_base_mass.mass_distribution_params", mass_range)

        base_com = _event(events, "base_com")
        if args_cli.disable_com_randomization:
            com_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
            _set_param(base_com, "com_range", com_range)
            _record(applied, "events.base_com.com_range", com_range)
        elif any(value is not None for value in (args_cli.com_x, args_cli.com_y, args_cli.com_z)):
            existing = dict(base_com.params.get("com_range", {})) if base_com is not None else {}
            for axis, value in (("x", args_cli.com_x), ("y", args_cli.com_y), ("z", args_cli.com_z)):
                if value is not None:
                    existing[axis] = (float(value), float(value))
            _set_param(base_com, "com_range", existing)
            _record(applied, "events.base_com.com_range", existing)

        motor_scale = (1.0, 1.0) if args_cli.disable_motor_randomization else None
        stiffness_range = motor_scale if motor_scale is not None else (
            (float(args_cli.motor_stiffness_scale), float(args_cli.motor_stiffness_scale))
            if args_cli.motor_stiffness_scale is not None
            else None
        )
        damping_range = motor_scale if motor_scale is not None else (
            (float(args_cli.motor_damping_scale), float(args_cli.motor_damping_scale))
            if args_cli.motor_damping_scale is not None
            else None
        )
        if stiffness_range is not None or damping_range is not None:
            matched = False
            for name in ("motor_strength", "motor_strength_hip_thigh", "motor_strength_calf"):
                motor_strength = _event(events, name)
                if motor_strength is None:
                    continue
                matched = True
                if stiffness_range is not None:
                    _set_param(motor_strength, "stiffness_distribution_params", stiffness_range)
                    _record(applied, f"events.{name}.stiffness_distribution_params", stiffness_range)
                if damping_range is not None:
                    _set_param(motor_strength, "damping_distribution_params", damping_range)
                    _record(applied, f"events.{name}.damping_distribution_params", damping_range)
            if not matched:
                warnings.append("motor randomization: no recognized actuator gain randomization event exists.")

        group_requests = {
            "hip_stiffness": args_cli.hip_stiffness,
            "thigh_stiffness": args_cli.thigh_stiffness,
            "calf_stiffness": args_cli.calf_stiffness,
            "hip_damping": args_cli.hip_damping,
            "thigh_damping": args_cli.thigh_damping,
            "calf_damping": args_cli.calf_damping,
        }
        for name, value in group_requests.items():
            if value is not None:
                warnings.append(f"{name}: active config exposes global gain randomization only; separate joint-group override not applied.")

        reset_base = _event(events, "reset_base")
        if any(value is not None for value in (args_cli.spawn_height, args_cli.spawn_roll, args_cli.spawn_pitch, args_cli.spawn_yaw)):
            pose_range = dict(reset_base.params.get("pose_range", {})) if reset_base is not None else {}
            for key, value in (
                ("z", args_cli.spawn_height),
                ("roll", args_cli.spawn_roll),
                ("pitch", args_cli.spawn_pitch),
                ("yaw", args_cli.spawn_yaw),
            ):
                if value is not None:
                    pose_range[key] = (float(value), float(value))
            _set_param(reset_base, "pose_range", pose_range)
            _record(applied, "events.reset_base.pose_range", pose_range)

        if args_cli.deterministic_env:
            for term_name in ("physics_material", "add_base_mass", "base_com", "motor_strength"):
                term = _event(events, term_name)
                if term is None:
                    continue
                for key, value in list(term.params.items()):
                    if key.endswith("_range") or key.endswith("_params"):
                        term.params[key] = _midpoint_range(value)
                        _record(applied, f"deterministic.events.{term_name}.{key}", term.params[key])
                    elif key == "com_range" and isinstance(value, dict):
                        term.params[key] = {axis: _midpoint_range(axis_range) for axis, axis_range in value.items()}
                        _record(applied, f"deterministic.events.{term_name}.{key}", term.params[key])

    if args_cli.obs_delay is not None:
        warnings.append("obs_delay: not applied. This env has no generic observation-delay buffer in config.")
    if args_cli.action_delay is not None:
        warnings.append("action_delay: not applied. This env has no generic action-delay buffer in config.")

    _scale_observation_noise(env_cfg, args_cli, applied, warnings)
    _set_command_ranges(env_cfg, args_cli, applied)
    return applied, warnings


def _force_isolated_terrain(env_cfg, terrain_type: str | None) -> None:
    if terrain_type is None:
        return

    terrain = env_cfg.scene.terrain
    if terrain_type == "plane":
        terrain.terrain_type = "plane"
        terrain.terrain_generator = None
        # Do not remove the scanner: actor-blind tasks may still retain a
        # critic-only terrain observation group that Isaac constructs at
        # runtime. The scanner can safely ray-cast against the plane.
        return

    terrain_gen = terrain.terrain_generator
    if terrain_gen is None:
        raise ValueError(f"Cannot force terrain type '{terrain_type}' because this task has no terrain generator.")

    terrain.max_init_terrain_level = 9
    for key in terrain_gen.sub_terrains.keys():
        terrain_gen.sub_terrains[key].proportion = 0.0

    presets = {
        "mixed_all": (
            "boxes",
            "pyramid_stairs",
            "pyramid_stairs_inv",
            "random_rough",
            "hf_pyramid_slope",
            "hf_pyramid_slope_inv",
        ),
        "mixed_geom_ood": (
            "boxes",
            "pyramid_stairs",
            "pyramid_stairs_inv",
        ),
    }
    terrain_names = presets.get(terrain_type, (terrain_type,))
    missing = [name for name in terrain_names if name not in terrain_gen.sub_terrains]
    if missing:
        valid = list(terrain_gen.sub_terrains.keys()) + list(presets.keys()) + ["plane"]
        raise ValueError(f"Unknown terrain type/preset '{terrain_type}'. Missing {missing}. Valid options: {valid}")

    for name in terrain_names:
        terrain_gen.sub_terrains[name].proportion = 1.0


def _disable_terrain_curriculum_for_fixed_level(env_cfg, terrain_type: str | None, terrain_level: int) -> None:
    if terrain_type == "plane" or terrain_level >= 0:
        curriculum = getattr(env_cfg, "curriculum", None)
        if curriculum is not None and hasattr(curriculum, "terrain_levels"):
            curriculum.terrain_levels = None


def _force_terrain_level(env, terrain_level: int) -> None:
    if terrain_level < 0:
        return
    terrain = env.unwrapped.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None:
        return
    level = max(0, min(int(terrain_level), int(terrain.terrain_origins.shape[0]) - 1))
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.unwrapped.scene.env_origins[:] = terrain.env_origins


def _print_evaluation_configuration(args_cli, applied: list[str], warnings: list[str]) -> None:
    fixed_command = args_cli.fixed_command or any(
        value is not None for value in (args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_yaw)
    )
    print("\n========== Evaluation Configuration ==========")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Task: {args_cli.task}")
    print(f"Seed: {args_cli.seed}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Terrain type: {args_cli.terrain_type if args_cli.terrain_type is not None else 'task default'}")
    print(f"Terrain level: {args_cli.terrain_level if args_cli.terrain_level >= 0 else 'task/curriculum default'}")
    print(f"Nominal env: {args_cli.nominal_env}")
    print(f"Deterministic env: {args_cli.deterministic_env}")
    print(f"Fixed command: {fixed_command}")
    if fixed_command:
        print(
            "Command: "
            f"vx={0.0 if args_cli.cmd_vx is None else args_cli.cmd_vx}, "
            f"vy={0.0 if args_cli.cmd_vy is None else args_cli.cmd_vy}, "
            f"yaw={0.0 if args_cli.cmd_yaw is None else args_cli.cmd_yaw}"
        )
    print(f"Teleop keyboard: {args_cli.teleop_keyboard}")
    print(f"Video: {args_cli.video}")
    print(f"Max steps: {args_cli.max_steps if args_cli.max_steps > 0 else 'continuous'}")
    print("Applied overrides:")
    if applied:
        for item in applied:
            print(f"  - {item}")
    else:
        print("  - none")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
    print("==============================================\n")


def main() -> None:
    print("[PLAY] Loading task and runner configs...", flush=True)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    print("[PLAY] Configs loaded.", flush=True)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    user_log_dir = Path.home() / ".cache" / "isaacsim" / "isaaclab" / "logs"
    user_log_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(env_cfg, "sim"):
        env_cfg.sim.log_dir = str(user_log_dir)
    if hasattr(env_cfg, "log_dir"):
        env_cfg.log_dir = str(user_log_dir)
    applied_overrides, override_warnings = _apply_cli_environment_overrides(env_cfg, args_cli)
    _print_evaluation_configuration(args_cli, applied_overrides, override_warnings)

    print("[PLAY] Creating Gym environment...", flush=True)
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    except Exception:
        print("[PLAY][ERROR] Gym environment creation failed:", flush=True)
        traceback.print_exc()
        raise
    print("[PLAY] Gym environment created.", flush=True)

    if args_cli.video:
        video_folder = Path(args_cli.video_folder).resolve()
        video_folder.mkdir(parents=True, exist_ok=True)
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "fps": int(1 / env.unwrapped.step_dt),
            "disable_logger": True,
        }
        print("[INFO] Recording policy playback video.")
        print(f"[INFO] video_folder: {video_folder}")
        print(f"[INFO] video_length: {args_cli.video_length}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    _force_terrain_level(env, args_cli.terrain_level)
    print("[PLAY] Creating runner and loading checkpoint...", flush=True)
    runner = _make_runner(env, runner_cfg)
    _safe_load_runner(runner, args_cli.checkpoint)
    print("[PLAY] Checkpoint loaded.", flush=True)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    print(f"[INFO] Loaded checkpoint from: {args_cli.checkpoint}")
    print(f"[INFO] task: {args_cli.task}")
    print(f"[INFO] num_envs: {args_cli.num_envs}")

    teleop = None
    fixed_command_requested = args_cli.fixed_command or any(
        value is not None for value in (args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_yaw)
    )
    fixed_command = None
    if fixed_command_requested:
        fixed_command = torch.tensor(
            [
                0.0 if args_cli.cmd_vx is None else float(args_cli.cmd_vx),
                0.0 if args_cli.cmd_vy is None else float(args_cli.cmd_vy),
                0.0 if args_cli.cmd_yaw is None else float(args_cli.cmd_yaw),
            ],
            device=env.unwrapped.device,
        )
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

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0
    reset_count = 0
    metrics = {
        "base_height": [],
        "base_tilt_projected_gravity_xy": [],
        "vel_err_xy": [],
        "yaw_err": [],
        "action_abs_mean": [],
    }
    teleop_command = None
    if args_cli.print_env_info:
        _print_env_info(env.unwrapped, timestep)

    print("[PLAY] Entering rollout loop.", flush=True)
    while simulation_app.is_running():
        start_time = time.time() if args_cli.real_time else None

        with torch.inference_mode():
            if teleop is not None:
                desired_cmd = teleop.advance().view(1, 3)
                if teleop_command is None:
                    teleop_command = torch.zeros_like(desired_cmd)
                if args_cli.teleop_lin_slew > 0.0:
                    lin_delta = (desired_cmd[:, :2] - teleop_command[:, :2]).clamp(
                        -args_cli.teleop_lin_slew * dt,
                        args_cli.teleop_lin_slew * dt,
                    )
                    teleop_command[:, :2] += lin_delta
                else:
                    teleop_command[:, :2] = desired_cmd[:, :2]
                if args_cli.teleop_yaw_slew > 0.0:
                    yaw_delta = (desired_cmd[:, 2] - teleop_command[:, 2]).clamp(
                        -args_cli.teleop_yaw_slew * dt,
                        args_cli.teleop_yaw_slew * dt,
                    )
                    teleop_command[:, 2] += yaw_delta
                else:
                    teleop_command[:, 2] = desired_cmd[:, 2]

                base_command = env.unwrapped.command_manager.get_command("base_velocity")
                base_command[:, 0] = teleop_command[:, 0]
                base_command[:, 1] = teleop_command[:, 1]
                base_command[:, 2] = teleop_command[:, 2]
            elif fixed_command is not None:
                base_command = env.unwrapped.command_manager.get_command("base_velocity")
                base_command[:, 0] = fixed_command[0]
                base_command[:, 1] = fixed_command[1]
                base_command[:, 2] = fixed_command[2]

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            robot = env.unwrapped.scene["robot"]
            base_command = env.unwrapped.command_manager.get_command("base_velocity")
            root_pos_w = robot.data.root_pos_w
            projected_gravity_b = robot.data.projected_gravity_b
            root_lin_vel_b = robot.data.root_lin_vel_b
            root_ang_vel_b = robot.data.root_ang_vel_b
            metrics["base_height"].append(float(root_pos_w[:, 2].mean().item()))
            metrics["base_tilt_projected_gravity_xy"].append(float(projected_gravity_b[:, :2].norm(dim=1).mean().item()))
            metrics["vel_err_xy"].append(float((root_lin_vel_b[:, :2] - base_command[:, :2]).norm(dim=1).mean().item()))
            metrics["yaw_err"].append(float((root_ang_vel_b[:, 2] - base_command[:, 2]).abs().mean().item()))
            metrics["action_abs_mean"].append(float(actions.abs().mean().item()))
            reset_count += int(dones.sum().item())
            policy_nn.reset(dones)
            if args_cli.terrain_level >= 0 and torch.any(dones):
                _force_terrain_level(env, args_cli.terrain_level)
            if fixed_command is not None and torch.any(dones):
                base_command = env.unwrapped.command_manager.get_command("base_velocity")
                base_command[:, 0] = fixed_command[0]
                base_command[:, 1] = fixed_command[1]
                base_command[:, 2] = fixed_command[2]

            if (
                args_cli.print_env_info
                and args_cli.env_info_interval > 0
                and timestep > 0
                and timestep % args_cli.env_info_interval == 0
            ):
                _print_env_info(env.unwrapped, timestep)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break

        if args_cli.real_time and start_time is not None:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"[PLAY] Rollout loop exited at timestep={timestep}.", flush=True)
    _write_eval_json(
        args_cli.eval_json_out,
        args_cli=args_cli,
        timestep=timestep,
        applied_overrides=applied_overrides,
        override_warnings=override_warnings,
        metrics=metrics,
        reset_count=reset_count,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
