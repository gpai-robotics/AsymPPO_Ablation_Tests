"""Debug and validate Adapt-V1 before spending full training time.

This script checks the first explicit latent-prediction path end to end:

- env/task registration and observation groups
- student history latent shape and stats
- frozen teacher latent target shape and stats
- masked latent regression loss on real env observations
- teacher action imitation delta on the same batch
- optional short rollout past the hidden switch horizon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect and validate Adapt-V1 latent wiring.")
parser.add_argument("--task", type=str, default="RMA-Go2-Adaptation-Student-Rough-History-V1")
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
from rma_go2_lab.models.adaptation.adapt_v1_ppo_cfg import Go2AdaptationStudentV1PPORunnerCfg
from rma_go2_lab.models.adaptation.actor_critic import HistoryEncoderStudentActorCritic
from rma_go2_lab.models.adaptation.frozen_v3_expert import FrozenV3Expert


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


def _build_student(obs_td: TensorDict, runner_cfg: Go2AdaptationStudentV1PPORunnerCfg, device: torch.device):
    policy_cfg = runner_cfg.policy
    student = HistoryEncoderStudentActorCritic(
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
        history_encoder_hidden_dims=policy_cfg.history_encoder_hidden_dims,
        history_group_name=policy_cfg.history_group_name,
        actor_init_path=policy_cfg.actor_init_path,
    ).to(device)
    student.eval()
    return student


def _mask_from_policy(obs_td: TensorDict, threshold: float) -> torch.Tensor:
    command = obs_td["policy"][:, 9:12]
    return (torch.linalg.norm(command, dim=-1) > threshold).float()


def _compute_batch_debug(
    obs_td: TensorDict,
    student: HistoryEncoderStudentActorCritic,
    teacher: FrozenV3Expert,
    latent_threshold: float,
) -> dict:
    with torch.no_grad():
        student_latent = student.encode_history_latent(obs_td)
        teacher_latent = teacher.get_latent_target(obs_td)
        student_actions = student.act_inference(obs_td)
        teacher_actions = teacher(obs_td)

        mask = _mask_from_policy(obs_td, latent_threshold)
        latent_mse_per_env = ((student_latent - teacher_latent) ** 2).mean(dim=-1)
        action_mse_per_env = ((student_actions - teacher_actions) ** 2).mean(dim=-1)
        cosine_per_env = torch.nn.functional.cosine_similarity(student_latent, teacher_latent, dim=-1)

        active = torch.count_nonzero(mask)
        if active > 0:
            masked_latent_mse = float((latent_mse_per_env * mask).sum().item() / (mask.sum().item() + 1e-6))
            masked_latent_cosine = float((cosine_per_env * mask).sum().item() / (mask.sum().item() + 1e-6))
            masked_action_mse = float((action_mse_per_env * mask).sum().item() / (mask.sum().item() + 1e-6))
        else:
            masked_latent_mse = 0.0
            masked_latent_cosine = 0.0
            masked_action_mse = 0.0

    return {
        "student_latent_stats": _tensor_stats(student_latent),
        "teacher_latent_stats": _tensor_stats(teacher_latent),
        "student_action_stats": _tensor_stats(student_actions),
        "teacher_action_stats": _tensor_stats(teacher_actions),
        "mask_active_frac": float(mask.mean().item()),
        "masked_latent_mse": masked_latent_mse,
        "masked_latent_cosine": masked_latent_cosine,
        "masked_action_mse": masked_action_mse,
        "sample_student_latent_rows": _sample_rows(student_latent, args_cli.sample_envs),
        "sample_teacher_latent_rows": _sample_rows(teacher_latent, args_cli.sample_envs),
        "sample_student_action_rows": _sample_rows(student_actions, args_cli.sample_envs),
        "sample_teacher_action_rows": _sample_rows(teacher_actions, args_cli.sample_envs),
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
        teacher = FrozenV3Expert(
            checkpoint_path=runner_cfg.algorithm.v3_expert_path,
            device=env.unwrapped.device,
        ).to(env.unwrapped.device)

        pre_switch = _compute_batch_debug(
            obs_td=obs_td,
            student=student,
            teacher=teacher,
            latent_threshold=runner_cfg.algorithm.latent_command_threshold,
        )

        zero_action = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        total_steps = args_cli.switch_step + args_cli.steps_after_switch
        for _ in range(total_steps):
            obs_dict, _, _, _, _ = env.step(zero_action)

        post_obs_td = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs]).to(env.unwrapped.device)
        post_switch = _compute_batch_debug(
            obs_td=post_obs_td,
            student=student,
            teacher=teacher,
            latent_threshold=runner_cfg.algorithm.latent_command_threshold,
        )

        result = {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "switch_step": args_cli.switch_step,
            "steps_after_switch": args_cli.steps_after_switch,
            "obs_group_shapes": {key: list(value.shape) for key, value in obs_dict.items()},
            "latent_target_name": teacher.latent_target_name,
            "latent_target_dim": teacher.latent_target_dim,
            "student_latent_dim": runner_cfg.policy.latent_dim,
            "pre_switch": pre_switch,
            "post_switch": post_switch,
            "switch_reached_frac_final": float(env.unwrapped._switch_applied.float().mean().item()),
        }

        print("\n=== Adapt-V1 Debug ===")
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
