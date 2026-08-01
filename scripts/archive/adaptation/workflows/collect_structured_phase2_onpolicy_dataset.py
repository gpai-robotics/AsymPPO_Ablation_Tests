"""Collect on-policy structured Phase 2 adaptation targets.

This follows the core RMA Phase 2 idea more faithfully than the online PPO
recovery branch:

- keep the structured Phase 1 teacher frozen
- roll out the current student policy with its current ``phi(history)``
- log history observations paired with teacher latent / teacher action targets

Intended usage:

1. Start from the structured Phase 2 task with either:
   - no checkpoint (fresh Phase 2 init from the frozen Phase 1 root), or
   - a current student checkpoint
2. Collect on-policy rollouts into chunked ``.npz`` files
3. Train only ``phi`` offline against the frozen structured teacher
4. Optionally repeat with the improved student checkpoint

Run this script with IsaacLab Python, e.g. via ``isaaclab.sh -p``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect on-policy structured Phase 2 adaptation targets.")
parser.add_argument(
    "--task",
    type=str,
    default="RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch",
    help="Registered task name.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Optional student checkpoint used for rollout.")
parser.add_argument("--output-dir", type=str, required=True, help="Directory where dataset chunks are written.")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--chunk-steps", type=int, default=250)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--command-threshold", type=float, default=0.1)
parser.add_argument("--adaptation-bottleneck-dim", type=int, default=None)
parser.add_argument("--adaptation-residual", action="store_true", help="Instantiate the student as a frozen-base plus residual adaptation branch.")
parser.add_argument(
    "--store-rollout-latent",
    action="store_true",
    help="Store the student latent used during rollout for debugging / later analysis.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401
from rma_go2_lab.models.adaptation.frozen_adapt_v3_phase1 import FrozenAdaptV3Phase1


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


def _safe_load_runner(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    checkpoint = None
    try:
        runner.load(checkpoint_path)
    except (RuntimeError, ValueError, KeyError) as exc:
        message = str(exc)
        if not (
            "normalizer" in message
            or "optimizer_state_dict" in message
            or "parameter group" in message
            or "size mismatch" in message
        ):
            raise
        print(f"[WARN] Standard runner.load() failed: {message}", flush=True)
        print("[WARN] Retrying with model-state-only fallback load.", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=runner.device)
        model_state = checkpoint["model_state_dict"]
        filtered_state = {k: v for k, v in model_state.items() if "normalizer" not in k}
        compatible_state = _filter_compatible_state(runner.alg.policy, filtered_state)
        resumed = runner.alg.policy.load_state_dict(compatible_state, strict=False)
        if hasattr(resumed, "missing_keys") and hasattr(resumed, "unexpected_keys"):
            print(f"[INFO] Fallback policy load complete. Missing keys: {list(resumed.missing_keys)}", flush=True)
            print(f"[INFO] Fallback policy load complete. Unexpected keys: {list(resumed.unexpected_keys)}", flush=True)
        else:
            print(f"[INFO] Fallback policy load complete. Return type: {type(resumed).__name__}", flush=True)

    if getattr(runner.alg.policy, "adaptation_residual_mode", False):
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location=runner.device)
        model_state = checkpoint["model_state_dict"]
        has_residual_base = any(key.startswith("base_adaptation_module.") for key in model_state.keys())
        if not has_residual_base:
            runner.alg.policy.zero_trainable_adaptation_path()
            runner.alg.policy.load_residual_base_from_checkpoint(checkpoint_path)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _write_chunk(output_dir: Path, chunk_index: int, buffer: dict[str, list[np.ndarray]]) -> int:
    arrays = {key: np.concatenate(value, axis=0) for key, value in buffer.items() if value}
    if not arrays:
        return 0
    path = output_dir / f"chunk_{chunk_index:04d}.npz"
    np.savez_compressed(path, **arrays)
    sample_count = int(next(iter(arrays.values())).shape[0])
    print(f"[INFO] Wrote chunk {chunk_index:04d} with {sample_count} samples to {path}", flush=True)
    return sample_count


def _empty_buffer() -> dict[str, list[np.ndarray]]:
    keys = [
        "policy",
        "policy_history",
        "dynamics_privileged",
        "teacher_latent",
        "teacher_action",
        "rollout_action",
        "command_active",
        "switch_applied",
        "step_index",
    ]
    if args_cli.store_rollout_latent:
        keys.append("student_latent")
    return {key: [] for key in keys}


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg_obj = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    agent_cfg = agent_cfg_obj.to_dict()
    if "policy" in agent_cfg and isinstance(agent_cfg["policy"], dict):
        agent_cfg["policy"]["pretrained_path"] = None
        if args_cli.adaptation_bottleneck_dim is not None:
            agent_cfg["policy"]["adaptation_bottleneck_dim"] = int(args_cli.adaptation_bottleneck_dim)
            agent_cfg["policy"]["adaptation_decoder_hidden_dims"] = [64]
        if args_cli.adaptation_residual:
            agent_cfg["policy"]["adaptation_residual_mode"] = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_obj.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg_obj.device)

    if args_cli.checkpoint:
        print(f"[INFO] Loading rollout checkpoint: {args_cli.checkpoint}", flush=True)
        _safe_load_runner(runner, args_cli.checkpoint)
    else:
        print("[INFO] No rollout checkpoint provided. Using fresh structured Phase 2 initialization.", flush=True)

    policy = runner.alg.policy
    policy.eval()

    phase1_ref_path = agent_cfg_obj.algorithm.phase1_reference_path
    teacher = FrozenAdaptV3Phase1(
        checkpoint_path=phase1_ref_path,
        device=agent_cfg_obj.device,
        terrain_group_name=policy.terrain_group_name,
        dynamics_group_name=policy.dynamics_group_name,
        terrain_dim=policy.terrain_dim,
    ).to(agent_cfg_obj.device)

    obs, _ = env.reset()
    obs = env.get_observations()

    chunk_index = 0
    total_samples = 0
    total_steps = 0
    buffer = _empty_buffer()

    with torch.no_grad():
        for step_idx in range(args_cli.steps):
            current_obs = {
                "policy": obs["policy"],
                "policy_history": obs["policy_history"],
                policy.dynamics_group_name: obs[policy.dynamics_group_name],
            }
            if policy.terrain_group_name is not None:
                current_obs[policy.terrain_group_name] = obs[policy.terrain_group_name]

            teacher_latent = teacher.encode_extrinsics_latent(current_obs)
            teacher_action = teacher.policy.act_inference(current_obs)
            student_latent = policy.encode_history_latent(current_obs)
            rollout_action = policy.act_inference(current_obs)

            command = current_obs["policy"][:, 9:12]
            command_active = (torch.linalg.norm(command, dim=-1) > float(args_cli.command_threshold)).float().unsqueeze(-1)
            switch_applied = (
                env.unwrapped._switch_applied.float().unsqueeze(-1).clone()
                if hasattr(env.unwrapped, "_switch_applied")
                else torch.zeros((env.num_envs, 1), device=rollout_action.device)
            )
            step_column = torch.full((env.num_envs, 1), step_idx, dtype=torch.int64, device=rollout_action.device)

            buffer["policy"].append(_to_numpy(current_obs["policy"]))
            buffer["policy_history"].append(_to_numpy(current_obs["policy_history"]))
            buffer["dynamics_privileged"].append(_to_numpy(current_obs[policy.dynamics_group_name]))
            buffer["teacher_latent"].append(_to_numpy(teacher_latent))
            buffer["teacher_action"].append(_to_numpy(teacher_action))
            buffer["rollout_action"].append(_to_numpy(rollout_action))
            buffer["command_active"].append(_to_numpy(command_active))
            buffer["switch_applied"].append(_to_numpy(switch_applied))
            buffer["step_index"].append(_to_numpy(step_column))
            if args_cli.store_rollout_latent:
                buffer["student_latent"].append(_to_numpy(student_latent))

            obs, _, _, infos = env.step(rollout_action)
            total_steps += 1

            if (step_idx + 1) % args_cli.chunk_steps == 0:
                total_samples += _write_chunk(output_dir, chunk_index, buffer)
                chunk_index += 1
                buffer = _empty_buffer()

            if (step_idx + 1) % 100 == 0:
                switch_frac = 0.0
                if isinstance(infos, dict):
                    logs = infos.get("log", {})
                    switch_frac = float(logs.get("adaptation_switch_applied_frac", 0.0) or 0.0)
                print(
                    f"[INFO] Collected step {step_idx + 1}/{args_cli.steps} "
                    f"(latest switch_applied_frac={switch_frac:.4f})",
                    flush=True,
                )

    if any(buffer.values()):
        total_samples += _write_chunk(output_dir, chunk_index, buffer)
        chunk_index += 1

    manifest = {
        "task": args_cli.task,
        "rollout_checkpoint": args_cli.checkpoint,
        "phase1_reference_path": phase1_ref_path,
        "output_dir": str(output_dir),
        "num_envs": int(args_cli.num_envs),
        "steps": int(args_cli.steps),
        "chunk_steps": int(args_cli.chunk_steps),
        "command_threshold": float(args_cli.command_threshold),
        "adaptation_bottleneck_dim": args_cli.adaptation_bottleneck_dim,
        "adaptation_residual": bool(args_cli.adaptation_residual),
        "chunks_written": int(chunk_index),
        "samples_written": int(total_samples),
        "stored_keys": list(_empty_buffer().keys()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[INFO] Wrote manifest to {output_dir / 'manifest.json'}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
