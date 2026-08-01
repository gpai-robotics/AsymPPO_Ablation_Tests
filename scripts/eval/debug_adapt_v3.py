"""Debug and validate Adapt-V3 latent wiring before long training.

This script checks the explicit RMA-style latent contract on real env
observations:

- env/task registration and observation groups
- extrinsics latent path `mu(e_t) -> z_t`
- history latent path `phi(history) -> z_hat_t`
- actor-path consistency:
  - Phase 1 should satisfy `act_inference(obs) == pi(x_t, mu(e_t))`
  - Phase 2 should satisfy `act_inference(obs) == pi(x_t, phi(history_t))`
- latent sensitivity:
  - zeroing the latent should change the action
  - shuffling the latent across envs should change the action
- optional short rollout past the hidden switch horizon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect and validate Adapt-V3 latent wiring.")
parser.add_argument("--task", type=str, default="RMA-Go2-Adapt-V3-Phase1-StageA")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--switch-step", type=int, default=30)
parser.add_argument("--steps-after-switch", type=int, default=8)
parser.add_argument("--sample-envs", type=int, default=4)
parser.add_argument("--json-out", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from tensordict import TensorDict

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401
from rma_go2_lab.models.adaptation.adapt_v3_ppo_cfg import Go2AdaptV3Phase1PPORunnerCfg, Go2AdaptV3Phase2PPORunnerCfg
from rma_go2_lab.models.adaptation.rma_v3_actor_critic import RmaV3ActorCritic


def _make_debug_env_cfg(env_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.adaptation_switch_step = args_cli.switch_step


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


def _mask_from_policy(obs_td: TensorDict, threshold: float) -> torch.Tensor:
    command = obs_td["policy"][:, 9:12]
    return (torch.linalg.norm(command, dim=-1) > threshold).float()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    active = torch.count_nonzero(mask)
    if active == 0:
        return 0.0
    return float((values * mask).sum().item() / (mask.sum().item() + 1e-6))


def _build_student(
    obs_td: TensorDict,
    runner_cfg: Go2AdaptV3Phase1PPORunnerCfg | Go2AdaptV3Phase2PPORunnerCfg,
    device: torch.device,
) -> RmaV3ActorCritic:
    policy_cfg = runner_cfg.policy
    student = RmaV3ActorCritic(
        obs=obs_td,
        obs_groups=runner_cfg.obs_groups,
        num_actions=12,
        actor_obs_normalization=policy_cfg.actor_obs_normalization,
        critic_obs_normalization=policy_cfg.critic_obs_normalization,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        activation=policy_cfg.activation,
        init_noise_std=policy_cfg.init_noise_std,
        latent_dim=policy_cfg.latent_dim,
        extrinsics_encoder_hidden_dims=policy_cfg.extrinsics_encoder_hidden_dims,
        adaptation_hidden_dims=policy_cfg.adaptation_hidden_dims,
        policy_group_name=policy_cfg.policy_group_name,
        history_group_name=policy_cfg.history_group_name,
        terrain_group_name=policy_cfg.terrain_group_name,
        dynamics_group_name=policy_cfg.dynamics_group_name,
        actor_init_path=policy_cfg.actor_init_path,
        extrinsics_init_path=getattr(policy_cfg, "extrinsics_init_path", None),
    ).to(device)
    if args_cli.checkpoint is not None:
        checkpoint = torch.load(args_cli.checkpoint, map_location=device)
        student.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[INFO] Loaded trained Adapt-V3 checkpoint from: {args_cli.checkpoint}")
    student.eval()
    return student


def _phase_mode(student: RmaV3ActorCritic) -> str:
    actor_groups = student.obs_groups["policy"]
    if any(group in actor_groups for group in student.extrinsics_group_names):
        return "phase1_extrinsics_actor"
    if student.history_group_name in actor_groups:
        return "phase2_history_actor"
    raise RuntimeError(f"Could not infer Adapt-V3 phase mode from actor groups: {actor_groups}")


def _compute_batch_debug(
    obs_td: TensorDict,
    student: RmaV3ActorCritic,
    latent_threshold: float,
) -> dict:
    with torch.no_grad():
        current_policy_obs = obs_td[student.policy_group_name]
        terrain_priv = obs_td[student.terrain_group_name] if student.terrain_group_name is not None else None
        dynamics_priv = obs_td[student.dynamics_group_name]
        extrinsics_input = student._get_extrinsics_input(obs_td)
        extrinsics_latent = student.encode_extrinsics_latent(obs_td)
        history_latent = student.encode_history_latent(obs_td)
        inference_actions = student.act_inference(obs_td)

        extrinsics_actions = student.act_with_latent(current_policy_obs, extrinsics_latent)
        history_actions = student.act_with_latent(current_policy_obs, history_latent)
        zero_actions = student.act_with_latent(current_policy_obs, torch.zeros_like(extrinsics_latent))

        shuffled_latent = extrinsics_latent.roll(shifts=1, dims=0)
        shuffled_actions = student.act_with_latent(current_policy_obs, shuffled_latent)

        terrain_zero_actions = None
        terrain_zero_delta = None
        if terrain_priv is not None:
            terrain_zero_obs = obs_td.clone()
            terrain_zero_obs[student.terrain_group_name] = torch.zeros_like(terrain_priv)
            terrain_zero_latent = student.encode_extrinsics_latent(terrain_zero_obs)
            terrain_zero_actions = student.act_with_latent(current_policy_obs, terrain_zero_latent)
            terrain_zero_delta = ((inference_actions - terrain_zero_actions) ** 2).mean(dim=-1)

        mask = _mask_from_policy(obs_td, latent_threshold)
        latent_mse_per_env = ((history_latent - extrinsics_latent) ** 2).mean(dim=-1)
        latent_cosine_per_env = torch.nn.functional.cosine_similarity(history_latent, extrinsics_latent, dim=-1)
        extrinsics_modular_delta = ((inference_actions - extrinsics_actions) ** 2).mean(dim=-1)
        history_modular_delta = ((inference_actions - history_actions) ** 2).mean(dim=-1)
        zero_latent_delta = ((inference_actions - zero_actions) ** 2).mean(dim=-1)
        shuffled_latent_delta = ((inference_actions - shuffled_actions) ** 2).mean(dim=-1)
        history_vs_extrinsics_action_delta = ((history_actions - extrinsics_actions) ** 2).mean(dim=-1)

    mode = _phase_mode(student)
    if mode == "phase1_extrinsics_actor":
        actor_path_consistency = extrinsics_modular_delta
        actor_path_consistency_name = "inference_vs_pi_x_mu_e"
    else:
        actor_path_consistency = history_modular_delta
        actor_path_consistency_name = "inference_vs_pi_x_phi_history"

    return {
        "phase_mode": mode,
        "terrain_group_name": student.terrain_group_name,
        "terrain_privileged_stats": _tensor_stats(terrain_priv) if terrain_priv is not None else None,
        "dynamics_privileged_stats": _tensor_stats(dynamics_priv),
        "extrinsics_input_stats": _tensor_stats(extrinsics_input),
        "extrinsics_latent_stats": _tensor_stats(extrinsics_latent),
        "history_latent_stats": _tensor_stats(history_latent),
        "inference_action_stats": _tensor_stats(inference_actions),
        "extrinsics_action_stats": _tensor_stats(extrinsics_actions),
        "history_action_stats": _tensor_stats(history_actions),
        "zero_latent_action_stats": _tensor_stats(zero_actions),
        "mask_active_frac": float(mask.mean().item()),
        "masked_history_vs_extrinsics_latent_mse": _masked_mean(latent_mse_per_env, mask),
        "masked_history_vs_extrinsics_latent_cosine": _masked_mean(latent_cosine_per_env, mask),
        actor_path_consistency_name: _masked_mean(actor_path_consistency, mask),
        "masked_inference_vs_pi_x_zero_latent": _masked_mean(zero_latent_delta, mask),
        "masked_inference_vs_pi_x_shuffled_mu_e": _masked_mean(shuffled_latent_delta, mask),
        "masked_inference_vs_pi_x_zero_terrain": (
            _masked_mean(terrain_zero_delta, mask) if terrain_zero_delta is not None else 0.0
        ),
        "masked_pi_x_phi_history_vs_pi_x_mu_e": _masked_mean(history_vs_extrinsics_action_delta, mask),
        "sample_terrain_privileged_rows": _sample_rows(terrain_priv, args_cli.sample_envs) if terrain_priv is not None else [],
        "sample_dynamics_privileged_rows": _sample_rows(dynamics_priv, args_cli.sample_envs),
        "sample_extrinsics_input_rows": _sample_rows(extrinsics_input, args_cli.sample_envs),
        "sample_extrinsics_latent_rows": _sample_rows(extrinsics_latent, args_cli.sample_envs),
        "sample_history_latent_rows": _sample_rows(history_latent, args_cli.sample_envs),
        "sample_inference_action_rows": _sample_rows(inference_actions, args_cli.sample_envs),
        "sample_extrinsics_action_rows": _sample_rows(extrinsics_actions, args_cli.sample_envs),
        "sample_history_action_rows": _sample_rows(history_actions, args_cli.sample_envs),
        "sample_zero_terrain_action_rows": (
            _sample_rows(terrain_zero_actions, args_cli.sample_envs) if terrain_zero_actions is not None else []
        ),
    }


def _latent_change(pre_latent: torch.Tensor, post_latent: torch.Tensor) -> dict[str, float]:
    per_env = ((post_latent - pre_latent) ** 2).mean(dim=-1)
    return {
        "mean_mse": float(per_env.mean().item()),
        "max_mse": float(per_env.max().item()),
    }


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    _make_debug_env_cfg(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        obs_dict, _ = env.reset(seed=args_cli.seed)
        obs_td = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs]).to(env.unwrapped.device)

        student = _build_student(obs_td, runner_cfg, env.unwrapped.device)

        with torch.no_grad():
            pre_extrinsics_latent = student.encode_extrinsics_latent(obs_td)
            pre_history_latent = student.encode_history_latent(obs_td)

        latent_threshold = getattr(runner_cfg.algorithm, "latent_command_threshold", 0.1)
        pre_switch = _compute_batch_debug(obs_td=obs_td, student=student, latent_threshold=latent_threshold)

        zero_action = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        total_steps = args_cli.switch_step + args_cli.steps_after_switch
        for _ in range(total_steps):
            obs_dict, _, _, _, _ = env.step(zero_action)

        post_obs_td = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs]).to(env.unwrapped.device)
        with torch.no_grad():
            post_extrinsics_latent = student.encode_extrinsics_latent(post_obs_td)
            post_history_latent = student.encode_history_latent(post_obs_td)
        post_switch = _compute_batch_debug(obs_td=post_obs_td, student=student, latent_threshold=latent_threshold)

        result = {
            "task": args_cli.task,
            "checkpoint": args_cli.checkpoint,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "switch_step": args_cli.switch_step,
            "steps_after_switch": args_cli.steps_after_switch,
            "obs_group_shapes": {key: list(value.shape) for key, value in obs_dict.items()},
            "policy_obs_groups": list(runner_cfg.obs_groups["policy"]),
            "critic_obs_groups": list(runner_cfg.obs_groups["critic"]),
            "latent_dim": runner_cfg.policy.latent_dim,
            "pre_switch": pre_switch,
            "post_switch": post_switch,
            "pre_to_post_extrinsics_latent_change": _latent_change(pre_extrinsics_latent, post_extrinsics_latent),
            "pre_to_post_history_latent_change": _latent_change(pre_history_latent, post_history_latent),
            "switch_reached_frac_final": float(env.unwrapped._switch_applied.float().mean().item()),
        }

        print("\n=== Adapt-V3 Debug ===")
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
