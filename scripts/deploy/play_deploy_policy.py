#!/usr/bin/env python3
"""Deployment-side sim rehearsal entrypoint.

This runs a packaged deployment candidate through IsaacLab using only the
deployable runtime contract recorded in the bundle:

- exported artifact
- deployable observation groups only
- no privileged teacher/runtime shortcuts
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle-dir", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--json-out", type=str, default=None)
parser.add_argument("--command-x", type=float, default=None)
parser.add_argument("--command-y", type=float, default=None)
parser.add_argument("--command-yaw", type=float, default=None)
parser.add_argument("--trace-steps", type=int, default=0)
parser.add_argument(
    "--probe-kind",
    type=str,
    default=None,
    choices=("friction", "mass", "motor"),
    help=(
        "Run an adaptation-proof probe by forcing one hidden-dynamics switch "
        "mid-rollout using the existing adaptation env machinery."
    ),
)
parser.add_argument(
    "--probe-shift-step",
    type=int,
    default=500,
    help="Control step at which the hidden-dynamics probe switch is scheduled.",
)
parser.add_argument(
    "--probe-window",
    type=int,
    default=100,
    help="Number of steps to summarize before and after the scheduled probe shift.",
)
parser.add_argument(
    "--probe-snapshot-radius",
    type=int,
    default=5,
    help="Capture detailed latent/action/dynamics snapshots for this many steps around the probe shift.",
)
parser.add_argument(
    "--progress-every",
    type=int,
    default=0,
    help="Print a short progress line every N control steps. Disabled when 0.",
)
parser.add_argument(
    "--compare-source",
    action="store_true",
    help="Also compare exported-policy actions against the rebuilt frozen source policy on the same deployable observations.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401
from rma_go2_lab.models.blind.history_actor_critic import TemporalBlindActorCritic
from rma_go2_lab.models.adaptation.rma_v3_actor_critic import RmaV3ActorCritic


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


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _slice_mean(values: list[float], start: int, end: int) -> float:
    clipped_start = max(start, 0)
    clipped_end = min(end, len(values))
    if clipped_end <= clipped_start:
        return 0.0
    window = values[clipped_start:clipped_end]
    return float(sum(window) / len(window)) if window else 0.0


def _tensor_list_mean(values: list[list[float]]) -> list[float]:
    if not values:
        return []
    dim = len(values[0])
    totals = [0.0] * dim
    for row in values:
        for idx, value in enumerate(row):
            totals[idx] += float(value)
    return [total / len(values) for total in totals]


def _find_torchscript_artifact(bundle_dir: Path, manifest: dict) -> Path:
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(".torchscript.pt"):
            artifact_path = bundle_dir / artifact
            if artifact_path.exists():
                return artifact_path
    raise SystemExit(
        "Could not find a TorchScript artifact in the deployment bundle. "
        "Expected one of the exported_artifacts entries to end with '.torchscript.pt'."
    )


def _validate_manifest(manifest: dict) -> None:
    supported_kinds = {"blind_adaptive_student", "blind_history_policy"}
    if manifest.get("policy_kind") not in supported_kinds:
        raise SystemExit(
            f"Unsupported policy kind for deploy rehearsal: {manifest.get('policy_kind')}. "
            f"Expected one of: {sorted(supported_kinds)}"
        )

    groups = manifest.get("deployable_observation_groups")
    if groups != ["policy", "policy_history"]:
        raise SystemExit(
            f"Unsupported deployable observation contract: {groups}. "
            "This rehearsal currently supports exactly ['policy', 'policy_history']."
        )


def _build_source_policy(obs, task: str, checkpoint_path: Path, device, policy_kind: str):
    runner_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    policy_cfg = runner_cfg.policy
    if policy_kind == "blind_adaptive_student":
        source_policy = RmaV3ActorCritic(
            obs=obs,
            obs_groups=runner_cfg.obs_groups,
            num_actions=12,
            actor_obs_normalization=policy_cfg.actor_obs_normalization,
            critic_obs_normalization=policy_cfg.critic_obs_normalization,
            actor_hidden_dims=policy_cfg.actor_hidden_dims,
            critic_hidden_dims=policy_cfg.critic_hidden_dims,
            activation=policy_cfg.activation,
            init_noise_std=policy_cfg.init_noise_std,
            latent_dim=policy_cfg.latent_dim,
            extrinsics_encoder_hidden_dims=getattr(policy_cfg, "extrinsics_encoder_hidden_dims", [128, 64]),
            extrinsics_encoder_mode=getattr(policy_cfg, "extrinsics_encoder_mode", "mlp"),
            extrinsics_identity_init=getattr(policy_cfg, "extrinsics_identity_init", False),
            dynamics_decoder_hidden_dims=getattr(policy_cfg, "dynamics_decoder_hidden_dims", [64]),
            dynamics_decoder_mode=getattr(policy_cfg, "dynamics_decoder_mode", "mlp"),
            dynamics_decoder_identity_init=getattr(policy_cfg, "dynamics_decoder_identity_init", False),
            adaptation_hidden_dims=getattr(policy_cfg, "adaptation_hidden_dims", [256, 128]),
            adaptation_bottleneck_dim=getattr(policy_cfg, "adaptation_bottleneck_dim", None),
            adaptation_decoder_hidden_dims=getattr(policy_cfg, "adaptation_decoder_hidden_dims", []),
            adaptation_residual_mode=getattr(policy_cfg, "adaptation_residual_mode", False),
            adaptation_residual_scale=getattr(policy_cfg, "adaptation_residual_scale", 1.0),
            adaptation_encoder_type=getattr(policy_cfg, "adaptation_encoder_type", "mlp"),
            temporal_channels=getattr(policy_cfg, "temporal_channels", [64, 64]),
            temporal_kernel_size=getattr(policy_cfg, "temporal_kernel_size", 3),
            history_feature_dim=getattr(policy_cfg, "history_feature_dim", 64),
            policy_group_name=policy_cfg.policy_group_name,
            history_group_name=policy_cfg.history_group_name,
            terrain_group_name=policy_cfg.terrain_group_name,
            dynamics_group_name=policy_cfg.dynamics_group_name,
        ).to(device)
    elif policy_kind == "blind_history_policy":
        source_policy = TemporalBlindActorCritic(
            obs=obs,
            obs_groups=runner_cfg.obs_groups,
            num_actions=12,
            actor_obs_normalization=policy_cfg.actor_obs_normalization,
            critic_obs_normalization=policy_cfg.critic_obs_normalization,
            actor_hidden_dims=policy_cfg.actor_hidden_dims,
            critic_hidden_dims=policy_cfg.critic_hidden_dims,
            activation=policy_cfg.activation,
            init_noise_std=policy_cfg.init_noise_std,
            history_group_name=policy_cfg.history_group_name,
            temporal_channels=policy_cfg.temporal_channels,
            temporal_kernel_size=policy_cfg.temporal_kernel_size,
            history_feature_dim=policy_cfg.history_feature_dim,
            history_target_dim=policy_cfg.history_target_dim,
            history_target_hidden_dims=policy_cfg.history_target_hidden_dims,
        ).to(device)
    else:
        raise SystemExit(f"Unsupported source policy kind for deploy rehearsal: {policy_kind}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    source_policy.eval()
    return source_policy


def _source_feature_and_actions(source_policy, policy_kind: str, policy_obs: torch.Tensor, history_obs: torch.Tensor):
    if policy_kind == "blind_adaptive_student":
        feature = source_policy.adapt_from_history(history_obs)
        live_actions = source_policy.act_with_latent(policy_obs, feature)
        return feature, live_actions

    if policy_kind == "blind_history_policy":
        obs_dict = {
            "policy": policy_obs,
            source_policy.history_group_name: history_obs,
        }
        feature = source_policy.encode_history_feature(obs_dict)
        actor_obs = torch.cat([policy_obs, feature], dim=-1)
        actor_obs = source_policy.actor_obs_normalizer(actor_obs)
        live_actions = source_policy.actor(actor_obs)
        return feature, live_actions

    raise SystemExit(f"Unsupported source policy kind for feature extraction: {policy_kind}")


def _actions_from_source_feature(source_policy, policy_kind: str, policy_obs: torch.Tensor, feature: torch.Tensor):
    if policy_kind == "blind_adaptive_student":
        return source_policy.act_with_latent(policy_obs, feature)

    if policy_kind == "blind_history_policy":
        actor_obs = torch.cat([policy_obs, feature], dim=-1)
        actor_obs = source_policy.actor_obs_normalizer(actor_obs)
        return source_policy.actor(actor_obs)

    raise SystemExit(f"Unsupported source policy kind for feature-conditioned actions: {policy_kind}")


def _configure_probe_env(env_cfg) -> None:
    if args_cli.probe_kind is None:
        return

    env_cfg.adaptation_switch_step = int(args_cli.probe_shift_step)
    env_cfg.adaptation_switch_episode_prob = 1.0
    env_cfg.adaptation_enable_friction_switch = args_cli.probe_kind == "friction"
    env_cfg.adaptation_enable_mass_switch = args_cli.probe_kind == "mass"
    env_cfg.adaptation_enable_motor_switch = args_cli.probe_kind == "motor"


def _summarize_dynamics_probe(dynamics_obs: torch.Tensor | None) -> dict | None:
    if dynamics_obs is None:
        return None
    summary = {
        "dim": int(dynamics_obs.shape[-1]),
        "mean": dynamics_obs.mean(dim=0).cpu().tolist(),
    }
    if dynamics_obs.shape[-1] >= 27:
        summary.update(
            {
                "static_friction_mean": float(dynamics_obs[:, 0].mean().item()),
                "dynamic_friction_mean": float(dynamics_obs[:, 1].mean().item()),
                "base_mass_ratio_mean": float(dynamics_obs[:, 2].mean().item()),
                "joint_stiffness_scale_mean": float(dynamics_obs[:, 3:15].mean().item()),
                "joint_damping_scale_mean": float(dynamics_obs[:, 15:27].mean().item()),
            }
        )
    return summary


def main() -> int:
    bundle_dir = Path(args_cli.bundle_dir)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing bundle manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest)
    policy_kind = manifest.get("policy_kind")
    artifact_path = _find_torchscript_artifact(bundle_dir, manifest)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _configure_probe_env(env_cfg)
    if (
        args_cli.command_x is not None
        or args_cli.command_y is not None
        or args_cli.command_yaw is not None
    ):
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

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        device = env.unwrapped.device
        policy = torch.jit.load(str(artifact_path), map_location=device)
        policy.eval()
        source_policy = None
        print(
            json.dumps(
                {
                    "deploy_rehearsal_stage": "policy_loaded",
                    "artifact_path": str(artifact_path),
                    "device": str(device),
                    "policy_kind": policy_kind,
                }
            ),
            flush=True,
        )

        reward_names = list(env.unwrapped.reward_manager.active_terms)
        termination_names = list(env.unwrapped.termination_manager.active_terms)
        reward_signed_sums = {name: 0.0 for name in reward_names}
        termination_counts = {name: 0 for name in termination_names}
        completed_episode_lengths: list[int] = []
        live_episode_steps = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)

        obs = _unwrap_obs(env.reset())
        policy_obs = obs["policy"]
        history_obs = obs["policy_history"]
        dynamics_obs = obs.get("dynamics_privileged") if isinstance(obs, dict) else None
        print(
            json.dumps(
                {
                    "deploy_rehearsal_stage": "env_reset_complete",
                    "policy_obs_shape": list(policy_obs.shape) if isinstance(policy_obs, torch.Tensor) else None,
                    "history_obs_shape": list(history_obs.shape) if isinstance(history_obs, torch.Tensor) else None,
                    "has_dynamics_obs": dynamics_obs is not None,
                }
            ),
            flush=True,
        )
        if not isinstance(policy_obs, torch.Tensor) or not isinstance(history_obs, torch.Tensor):
            raise SystemExit("Deploy rehearsal expected tensor observation groups 'policy' and 'policy_history'.")
        if args_cli.compare_source or args_cli.probe_kind is not None:
            source_checkpoint = Path(manifest["source_checkpoint"])
            source_policy = _build_source_policy(obs, args_cli.task, source_checkpoint, device, policy_kind)
            print(
                json.dumps(
                    {
                        "deploy_rehearsal_stage": "source_policy_loaded",
                        "source_checkpoint": str(source_checkpoint),
                    }
                ),
                flush=True,
            )

        action_abs_sum = 0.0
        action_abs_diff_sum = 0.0
        action_mse_sum = 0.0
        action_max_abs_diff = 0.0
        vel_err_sum = 0.0
        yaw_err_sum = 0.0
        base_height_sum = 0.0
        base_tilt_sum = 0.0
        total_steps = 0
        trace = []
        latent_norm_per_step: list[float] = []
        latent_delta_from_prev_per_step: list[float] = []
        latent_delta_from_initial_per_step: list[float] = []
        latent_cosine_to_initial_per_step: list[float] = []
        history_norm_per_step: list[float] = []
        history_delta_from_prev_per_step: list[float] = []
        history_delta_from_initial_per_step: list[float] = []
        vel_err_per_step: list[float] = []
        yaw_err_per_step: list[float] = []
        action_abs_per_step: list[float] = []
        base_height_per_step: list[float] = []
        probe_dynamics_per_step: list[dict] = []
        latent_mean_per_step: list[list[float]] = []
        latent_std_per_step: list[list[float]] = []
        live_vs_frozen_latent_action_diff_per_step: list[float] = []
        live_vs_zero_latent_action_diff_per_step: list[float] = []
        frozen_vs_zero_latent_action_diff_per_step: list[float] = []
        live_vs_frozen_latent_action_mse_per_step: list[float] = []
        live_vs_zero_latent_action_mse_per_step: list[float] = []
        switch_applied_step = None
        prev_source_latent = None
        initial_source_latent = None
        prev_history_obs = None
        initial_history_obs = history_obs.detach().clone()

        with torch.inference_mode():
            for step_idx in range(args_cli.max_steps):
                if step_idx == 0:
                    print(
                        json.dumps(
                            {
                                "deploy_rehearsal_stage": "first_step_begin",
                                "step": step_idx,
                            }
                        ),
                        flush=True,
                    )
                history_norm_per_step.append(float(history_obs.norm(dim=-1).mean().item()))
                if prev_history_obs is None:
                    history_delta_from_prev_per_step.append(0.0)
                else:
                    history_delta_from_prev_per_step.append(
                        float((history_obs - prev_history_obs).norm(dim=-1).mean().item())
                    )
                history_delta_from_initial_per_step.append(
                    float((history_obs - initial_history_obs).norm(dim=-1).mean().item())
                )
                prev_history_obs = history_obs.detach().clone()

                actions = policy(policy_obs, history_obs)
                if step_idx == 0:
                    print(
                        json.dumps(
                            {
                                "deploy_rehearsal_stage": "first_forward_complete",
                                "action_shape": list(actions.shape),
                                "action_abs_mean": float(actions.abs().mean().item()),
                            }
                        ),
                        flush=True,
                    )
                if source_policy is not None:
                    source_latent, source_actions = _source_feature_and_actions(
                        source_policy,
                        policy_kind,
                        policy_obs,
                        history_obs,
                    )
                    if initial_source_latent is None:
                        initial_source_latent = source_latent.detach().clone()
                    latent_norm_per_step.append(float(source_latent.norm(dim=-1).mean().item()))
                    if prev_source_latent is None:
                        latent_delta_from_prev_per_step.append(0.0)
                    else:
                        latent_delta_from_prev_per_step.append(
                            float((source_latent - prev_source_latent).norm(dim=-1).mean().item())
                        )
                    latent_delta_from_initial_per_step.append(
                        float((source_latent - initial_source_latent).norm(dim=-1).mean().item())
                    )
                    cosine = torch.nn.functional.cosine_similarity(source_latent, initial_source_latent, dim=-1)
                    latent_cosine_to_initial_per_step.append(float(cosine.mean().item()))
                    latent_mean_per_step.append(source_latent.mean(dim=0).cpu().tolist())
                    latent_std_per_step.append(source_latent.std(dim=0, unbiased=False).cpu().tolist())
                    prev_source_latent = source_latent.detach().clone()
                    frozen_latent_actions = _actions_from_source_feature(
                        source_policy,
                        policy_kind,
                        policy_obs,
                        initial_source_latent,
                    )
                    zero_latent_actions = _actions_from_source_feature(
                        source_policy,
                        policy_kind,
                        policy_obs,
                        torch.zeros_like(source_latent),
                    )
                    live_vs_frozen_latent_action_diff_per_step.append(
                        float((source_actions - frozen_latent_actions).abs().mean().item())
                    )
                    live_vs_zero_latent_action_diff_per_step.append(
                        float((source_actions - zero_latent_actions).abs().mean().item())
                    )
                    frozen_vs_zero_latent_action_diff_per_step.append(
                        float((frozen_latent_actions - zero_latent_actions).abs().mean().item())
                    )
                    live_vs_frozen_latent_action_mse_per_step.append(
                        float(((source_actions - frozen_latent_actions) ** 2).mean().item())
                    )
                    live_vs_zero_latent_action_mse_per_step.append(
                        float(((source_actions - zero_latent_actions) ** 2).mean().item())
                    )
                    if args_cli.compare_source:
                        action_diff = actions - source_actions
                        action_abs_diff_sum += float(action_diff.abs().mean().item())
                        action_mse_sum += float((action_diff**2).mean().item())
                        action_max_abs_diff = max(action_max_abs_diff, float(action_diff.abs().max().item()))
                obs, _, dones, _ = _step_env(env, actions)
                if step_idx == 0:
                    print(
                        json.dumps(
                            {
                                "deploy_rehearsal_stage": "first_env_step_complete",
                                "done_any": bool(dones.any().item()) if isinstance(dones, torch.Tensor) else None,
                            }
                        ),
                        flush=True,
                    )
                obs = _unwrap_obs(obs)
                policy_obs = obs["policy"]
                history_obs = obs["policy_history"]
                dynamics_obs = obs.get("dynamics_privileged") if isinstance(obs, dict) else None
                live_episode_steps += 1
                total_steps += 1

                robot = env.unwrapped.scene["robot"]
                command = env.unwrapped.command_manager.get_command("base_velocity")
                planar_vel = robot.data.root_lin_vel_b[:, :2]
                planar_cmd = command[:, :2]
                yaw_vel = robot.data.root_ang_vel_b[:, 2]
                yaw_cmd = command[:, 2]

                vel_err = torch.linalg.norm(planar_vel - planar_cmd, dim=-1)
                yaw_err = (yaw_vel - yaw_cmd).abs()
                action_abs_sum += float(actions.abs().mean().item())
                vel_err_sum += float(vel_err.mean().item())
                yaw_err_sum += float(yaw_err.mean().item())
                base_height_sum += float(robot.data.root_pos_w[:, 2].mean().item())
                vel_err_per_step.append(float(vel_err.mean().item()))
                yaw_err_per_step.append(float(yaw_err.mean().item()))
                action_abs_per_step.append(float(actions.abs().mean().item()))
                base_height_per_step.append(float(robot.data.root_pos_w[:, 2].mean().item()))

                projected_gravity = getattr(robot.data, "projected_gravity_b", None)
                if projected_gravity is not None:
                    base_tilt_sum += float(projected_gravity[:, :2].norm(dim=1).mean().item())

                if args_cli.probe_kind is not None:
                    probe_dynamics_per_step.append(_summarize_dynamics_probe(dynamics_obs))
                    env_switch_applied = getattr(env.unwrapped, "_switch_applied", None)
                    if (
                        switch_applied_step is None
                        and isinstance(env_switch_applied, torch.Tensor)
                        and bool(env_switch_applied.any().item())
                    ):
                        switch_applied_step = step_idx

                if step_idx < args_cli.trace_steps:
                    trace_item = {
                        "step": step_idx,
                        "command_mean": command.mean(dim=0).cpu().tolist(),
                        "base_lin_vel_local_mean": robot.data.root_lin_vel_b.mean(dim=0).cpu().tolist(),
                        "base_ang_vel_local_mean": robot.data.root_ang_vel_b.mean(dim=0).cpu().tolist(),
                        "projected_gravity_mean": (
                            projected_gravity.mean(dim=0).cpu().tolist()
                            if projected_gravity is not None
                            else None
                        ),
                        "root_pos_world_mean": robot.data.root_pos_w.mean(dim=0).cpu().tolist(),
                        "joint_pos_mean": robot.data.joint_pos.mean(dim=0).cpu().tolist(),
                        "joint_vel_mean": robot.data.joint_vel.mean(dim=0).cpu().tolist(),
                        "action_mean": actions.mean(dim=0).cpu().tolist(),
                        "action_abs_mean": float(actions.abs().mean().item()),
                        "vel_err_mean": float(vel_err.mean().item()),
                        "yaw_err_mean": float(yaw_err.mean().item()),
                    }
                    if source_policy is not None:
                        trace_item["latent_norm_mean"] = latent_norm_per_step[-1]
                        trace_item["latent_delta_from_prev_mean"] = latent_delta_from_prev_per_step[-1]
                        trace_item["latent_delta_from_initial_mean"] = latent_delta_from_initial_per_step[-1]
                    if args_cli.probe_kind is not None:
                        trace_item["dynamics_probe_mean"] = probe_dynamics_per_step[-1]
                        trace_item["switch_applied"] = (
                            bool(getattr(env.unwrapped, "_switch_applied", torch.zeros(1, dtype=torch.bool))[0].item())
                            if hasattr(env.unwrapped, "_switch_applied")
                            else None
                        )
                    trace.append(trace_item)

                if args_cli.progress_every > 0 and (
                    (step_idx + 1) % args_cli.progress_every == 0 or (step_idx + 1) == args_cli.max_steps
                ):
                    print(
                        json.dumps(
                            {
                                "progress_step": step_idx + 1,
                                "max_steps": args_cli.max_steps,
                                "action_abs_mean_so_far": action_abs_sum / max(total_steps, 1),
                                "vel_err_step_mean_so_far": vel_err_sum / max(total_steps, 1),
                                "yaw_err_step_mean_so_far": yaw_err_sum / max(total_steps, 1),
                                "base_height_mean_so_far": base_height_sum / max(total_steps, 1),
                            }
                        ),
                        flush=True,
                    )

                step_reward = env.unwrapped.reward_manager._step_reward
                for idx, name in enumerate(reward_names):
                    reward_signed_sums[name] += float(step_reward[:, idx].mean().item())

                for name in termination_names:
                    fired = env.unwrapped.termination_manager.get_term(name)
                    termination_counts[name] += int(fired.sum().item())

                if isinstance(dones, torch.Tensor) and dones.any():
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    completed_episode_lengths.extend(live_episode_steps[done_ids].cpu().tolist())
                    live_episode_steps[done_ids] = 0

        results = {
            "task": args_cli.task,
            "bundle_dir": str(bundle_dir),
            "policy_name": manifest.get("policy_name"),
            "policy_kind": manifest.get("policy_kind"),
            "artifact_used": str(artifact_path),
            "deployable_observation_groups": manifest.get("deployable_observation_groups"),
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "seed": args_cli.seed,
            "fixed_command": {
                "x": args_cli.command_x,
                "y": args_cli.command_y,
                "yaw": args_cli.command_yaw,
            },
            "status": "completed_rehearsal",
            "summary_metrics": {
                "action_abs_mean": action_abs_sum / max(total_steps, 1),
                "vel_err_step_mean": vel_err_sum / max(total_steps, 1),
                "yaw_err_step_mean": yaw_err_sum / max(total_steps, 1),
                "base_height_mean": base_height_sum / max(total_steps, 1),
                "base_tilt_projected_gravity_xy_mean": base_tilt_sum / max(total_steps, 1),
                "episode_length_mean": _mean_or_zero([float(v) for v in completed_episode_lengths]),
                "num_completed_episodes": len(completed_episode_lengths),
            },
            "reward_mean_per_step": {
                name: reward_signed_sums[name] / max(total_steps, 1) for name in reward_names
            },
            "termination_counts": termination_counts,
            "termination_fraction_of_env_steps": {
                name: termination_counts[name] / float(args_cli.num_envs * max(total_steps, 1))
                for name in termination_names
            },
            "trace_steps_captured": len(trace),
            "trace": trace,
        }
        if args_cli.compare_source:
            results["parity_metrics"] = {
                "compared_against_source_checkpoint": manifest["source_checkpoint"],
                "action_abs_diff_mean": action_abs_diff_sum / max(total_steps, 1),
                "action_mse_mean": action_mse_sum / max(total_steps, 1),
                "action_max_abs_diff": action_max_abs_diff,
            }
        if args_cli.probe_kind is not None:
            shift_step = int(args_cli.probe_shift_step)
            window = int(args_cli.probe_window)
            snapshot_radius = int(args_cli.probe_snapshot_radius)
            snapshot_start = max(0, shift_step - snapshot_radius)
            snapshot_end = min(len(latent_mean_per_step), shift_step + snapshot_radius + 1)
            latent_snapshots = []
            for idx in range(snapshot_start, snapshot_end):
                latent_snapshots.append(
                    {
                        "step": idx,
                        "latent_mean": latent_mean_per_step[idx] if idx < len(latent_mean_per_step) else None,
                        "latent_std": latent_std_per_step[idx] if idx < len(latent_std_per_step) else None,
                        "latent_norm_mean": latent_norm_per_step[idx] if idx < len(latent_norm_per_step) else None,
                        "latent_delta_from_prev_mean": (
                            latent_delta_from_prev_per_step[idx]
                            if idx < len(latent_delta_from_prev_per_step)
                            else None
                        ),
                        "latent_delta_from_initial_mean": (
                            latent_delta_from_initial_per_step[idx]
                            if idx < len(latent_delta_from_initial_per_step)
                            else None
                        ),
                        "latent_cosine_to_initial_mean": (
                            latent_cosine_to_initial_per_step[idx]
                            if idx < len(latent_cosine_to_initial_per_step)
                            else None
                        ),
                        "history_norm_mean": history_norm_per_step[idx] if idx < len(history_norm_per_step) else None,
                        "history_delta_from_prev_mean": (
                            history_delta_from_prev_per_step[idx]
                            if idx < len(history_delta_from_prev_per_step)
                            else None
                        ),
                        "history_delta_from_initial_mean": (
                            history_delta_from_initial_per_step[idx]
                            if idx < len(history_delta_from_initial_per_step)
                            else None
                        ),
                        "live_vs_frozen_latent_action_abs_diff_mean": (
                            live_vs_frozen_latent_action_diff_per_step[idx]
                            if idx < len(live_vs_frozen_latent_action_diff_per_step)
                            else None
                        ),
                        "live_vs_zero_latent_action_abs_diff_mean": (
                            live_vs_zero_latent_action_diff_per_step[idx]
                            if idx < len(live_vs_zero_latent_action_diff_per_step)
                            else None
                        ),
                        "vel_err_step_mean": vel_err_per_step[idx] if idx < len(vel_err_per_step) else None,
                        "yaw_err_step_mean": yaw_err_per_step[idx] if idx < len(yaw_err_per_step) else None,
                        "action_abs_mean": action_abs_per_step[idx] if idx < len(action_abs_per_step) else None,
                        "base_height_mean": base_height_per_step[idx] if idx < len(base_height_per_step) else None,
                        "dynamics_probe_mean": probe_dynamics_per_step[idx] if idx < len(probe_dynamics_per_step) else None,
                    }
                )
            results["adaptation_probe"] = {
                "probe_kind": args_cli.probe_kind,
                "probe_shift_step": shift_step,
                "probe_window": window,
                "probe_snapshot_radius": snapshot_radius,
                "source_checkpoint_for_latent_probe": manifest["source_checkpoint"],
                "switch_applied_observed_step": switch_applied_step,
                "pre_window_summary": {
                    "latent_norm_mean": _slice_mean(latent_norm_per_step, shift_step - window, shift_step),
                    "latent_delta_from_prev_mean": _slice_mean(
                        latent_delta_from_prev_per_step, shift_step - window, shift_step
                    ),
                    "latent_delta_from_initial_mean": _slice_mean(
                        latent_delta_from_initial_per_step, shift_step - window, shift_step
                    ),
                    "latent_cosine_to_initial_mean": _slice_mean(
                        latent_cosine_to_initial_per_step, shift_step - window, shift_step
                    ),
                    "history_norm_mean": _slice_mean(history_norm_per_step, shift_step - window, shift_step),
                    "history_delta_from_prev_mean": _slice_mean(
                        history_delta_from_prev_per_step, shift_step - window, shift_step
                    ),
                    "history_delta_from_initial_mean": _slice_mean(
                        history_delta_from_initial_per_step, shift_step - window, shift_step
                    ),
                    "live_vs_frozen_latent_action_abs_diff_mean": _slice_mean(
                        live_vs_frozen_latent_action_diff_per_step, shift_step - window, shift_step
                    ),
                    "live_vs_zero_latent_action_abs_diff_mean": _slice_mean(
                        live_vs_zero_latent_action_diff_per_step, shift_step - window, shift_step
                    ),
                    "frozen_vs_zero_latent_action_abs_diff_mean": _slice_mean(
                        frozen_vs_zero_latent_action_diff_per_step, shift_step - window, shift_step
                    ),
                    "vel_err_step_mean": _slice_mean(vel_err_per_step, shift_step - window, shift_step),
                    "yaw_err_step_mean": _slice_mean(yaw_err_per_step, shift_step - window, shift_step),
                    "action_abs_mean": _slice_mean(action_abs_per_step, shift_step - window, shift_step),
                    "base_height_mean": _slice_mean(base_height_per_step, shift_step - window, shift_step),
                },
                "post_window_summary": {
                    "latent_norm_mean": _slice_mean(latent_norm_per_step, shift_step, shift_step + window),
                    "latent_delta_from_prev_mean": _slice_mean(
                        latent_delta_from_prev_per_step, shift_step, shift_step + window
                    ),
                    "latent_delta_from_initial_mean": _slice_mean(
                        latent_delta_from_initial_per_step, shift_step, shift_step + window
                    ),
                    "latent_cosine_to_initial_mean": _slice_mean(
                        latent_cosine_to_initial_per_step, shift_step, shift_step + window
                    ),
                    "history_norm_mean": _slice_mean(history_norm_per_step, shift_step, shift_step + window),
                    "history_delta_from_prev_mean": _slice_mean(
                        history_delta_from_prev_per_step, shift_step, shift_step + window
                    ),
                    "history_delta_from_initial_mean": _slice_mean(
                        history_delta_from_initial_per_step, shift_step, shift_step + window
                    ),
                    "live_vs_frozen_latent_action_abs_diff_mean": _slice_mean(
                        live_vs_frozen_latent_action_diff_per_step, shift_step, shift_step + window
                    ),
                    "live_vs_zero_latent_action_abs_diff_mean": _slice_mean(
                        live_vs_zero_latent_action_diff_per_step, shift_step, shift_step + window
                    ),
                    "frozen_vs_zero_latent_action_abs_diff_mean": _slice_mean(
                        frozen_vs_zero_latent_action_diff_per_step, shift_step, shift_step + window
                    ),
                    "vel_err_step_mean": _slice_mean(vel_err_per_step, shift_step, shift_step + window),
                    "yaw_err_step_mean": _slice_mean(yaw_err_per_step, shift_step, shift_step + window),
                    "action_abs_mean": _slice_mean(action_abs_per_step, shift_step, shift_step + window),
                    "base_height_mean": _slice_mean(base_height_per_step, shift_step, shift_step + window),
                },
                "dynamics_probe_pre_mean": probe_dynamics_per_step[max(shift_step - 1, 0)]
                if probe_dynamics_per_step
                else None,
                "dynamics_probe_post_mean": probe_dynamics_per_step[min(shift_step, len(probe_dynamics_per_step) - 1)]
                if probe_dynamics_per_step
                else None,
                "latent_ablation_summary": {
                    "live_vs_frozen_latent_action_abs_diff_mean_full_rollout": _mean_or_zero(
                        live_vs_frozen_latent_action_diff_per_step
                    ),
                    "live_vs_zero_latent_action_abs_diff_mean_full_rollout": _mean_or_zero(
                        live_vs_zero_latent_action_diff_per_step
                    ),
                    "frozen_vs_zero_latent_action_abs_diff_mean_full_rollout": _mean_or_zero(
                        frozen_vs_zero_latent_action_diff_per_step
                    ),
                    "live_vs_frozen_latent_action_mse_mean_full_rollout": _mean_or_zero(
                        live_vs_frozen_latent_action_mse_per_step
                    ),
                    "live_vs_zero_latent_action_mse_mean_full_rollout": _mean_or_zero(
                        live_vs_zero_latent_action_mse_per_step
                    ),
                },
                "latent_mean_pre_window": _tensor_list_mean(
                    latent_mean_per_step[max(shift_step - window, 0): min(shift_step, len(latent_mean_per_step))]
                ),
                "latent_mean_post_window": _tensor_list_mean(
                    latent_mean_per_step[max(shift_step, 0): min(shift_step + window, len(latent_mean_per_step))]
                ),
                "latent_snapshots_around_shift": latent_snapshots,
            }

        print(json.dumps(results, indent=2))
        if args_cli.json_out:
            Path(args_cli.json_out).write_text(json.dumps(results, indent=2) + "\n")
        return 0
    except Exception as exc:  # pragma: no cover - deployment debug path
        print(
            json.dumps(
                {
                    "deploy_rehearsal_stage": "exception",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            flush=True,
        )
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
