"""Train only the structured Phase 2 adaptation module offline.

This is the RMA-style replacement for the unstable online PPO Phase 2 branch:

- freeze the structured Phase 1 teacher
- freeze the structured student actor / critic / privileged path
- optimize only ``phi(history)``
- regress to the teacher latent and optionally teacher action

Unlike the online training stack, this script intentionally avoids the IsaacLab
task registry so it can run in a lighter environment without pulling in the
full simulator / pxr dependency chain.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_REFERENCE_PATH = (
    REPO_ROOT / "rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt"
)


def _load_class_from_file(module_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(f"_offline_{module_path.stem}_{class_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


RmaV3ActorCritic = _load_class_from_file(
    REPO_ROOT / "rma_go2_lab/models/adaptation/rma_v3_actor_critic.py", "RmaV3ActorCritic"
)


parser = argparse.ArgumentParser(description="Offline structured Phase 2 adaptation training.")
parser.add_argument(
    "--task",
    type=str,
    default="RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch",
    help="Logical task label stored in outputs. No task-registry import is used.",
)
parser.add_argument("--dataset-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional student checkpoint to continue from. If unset, uses the task's default structured Phase 2 init.",
)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument("--latent-coef", type=float, default=1.0)
parser.add_argument("--action-coef", type=float, default=0.10)
parser.add_argument("--latent-l2-coef", type=float, default=1.0e-3)
parser.add_argument("--adaptation-bottleneck-dim", type=int, default=None)
parser.add_argument("--adaptation-residual", action="store_true", help="Train a compact residual correction branch on top of a frozen full-latent history adapter.")
parser.add_argument("--active-only", action="store_true")
parser.add_argument("--val-fraction", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument(
    "--low-friction-threshold",
    type=float,
    default=0.35,
    help="Samples with min(static_friction, dynamic_friction) below this are upweighted.",
)
parser.add_argument(
    "--low-friction-upweight",
    type=float,
    default=1.0,
    help="Multiplicative weight applied to low-friction samples.",
)
parser.add_argument(
    "--switch-upweight",
    type=float,
    default=1.0,
    help="Multiplicative weight applied to samples collected after a hidden switch fired.",
)
parser.add_argument(
    "--very-heavy-threshold",
    type=float,
    default=1.35,
    help="Samples with base_mass_ratio above this are treated as very heavy.",
)
parser.add_argument(
    "--very-heavy-upweight",
    type=float,
    default=1.0,
    help="Multiplicative weight applied to very-heavy samples.",
)
parser.add_argument(
    "--weak-motor-threshold",
    type=float,
    default=0.75,
    help="Samples with mean motor stiffness/damping below this are treated as very weak motor cases.",
)
parser.add_argument(
    "--weak-motor-upweight",
    type=float,
    default=1.0,
    help="Multiplicative weight applied to weak-motor samples.",
)
args_cli = parser.parse_args()


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


def _dummy_obs(policy_dim: int, history_dim: int, dynamics_dim: int) -> dict[str, torch.Tensor]:
    return {
        "policy": torch.zeros(1, policy_dim),
        "policy_history": torch.zeros(1, history_dim),
        "dynamics_privileged": torch.zeros(1, dynamics_dim),
    }


def _build_policy_from_structured_phase2_defaults(
    sample_chunk: Path,
    device: str,
    checkpoint_path: str | None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    chunk = np.load(sample_chunk)
    policy_dim = int(chunk["policy"].shape[1])
    history_dim = int(chunk["policy_history"].shape[1])
    dynamics_dim = int(chunk["dynamics_privileged"].shape[1])

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy_kwargs = dict(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=27,
        extrinsics_encoder_hidden_dims=[],
        extrinsics_encoder_mode="linear",
        extrinsics_identity_init=True,
        dynamics_decoder_hidden_dims=[],
        dynamics_decoder_mode="linear",
        dynamics_decoder_identity_init=True,
        adaptation_hidden_dims=[256, 128],
        adaptation_bottleneck_dim=args_cli.adaptation_bottleneck_dim,
        adaptation_decoder_hidden_dims=[64] if args_cli.adaptation_bottleneck_dim is not None else [],
        adaptation_residual_mode=bool(args_cli.adaptation_residual),
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        full_init_path=None,
        actor_init_path=str(PHASE1_REFERENCE_PATH),
        critic_init_path=None,
        actor_only_extrinsics_init_mode="identity",
        actor_only_adaptation_init_mode="zero" if args_cli.adaptation_residual else "small_xavier",
    )

    student = RmaV3ActorCritic(
        obs=_dummy_obs(policy_dim, history_dim, dynamics_dim),
        obs_groups=obs_groups,
        num_actions=12,
        **policy_kwargs,
    ).to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = checkpoint["model_state_dict"]
        filtered_state = {k: v for k, v in model_state.items() if "normalizer" not in k}
        compatible_state = _filter_compatible_state(student, filtered_state)
        resumed = student.load_state_dict(compatible_state, strict=False)
        if args_cli.adaptation_residual:
            student.load_residual_base_from_checkpoint(checkpoint_path)
        print(f"[INFO] Loaded student checkpoint: {checkpoint_path}", flush=True)
        if hasattr(resumed, "missing_keys") and hasattr(resumed, "unexpected_keys"):
            print(f"[INFO] Compatible-load missing keys: {list(resumed.missing_keys)}", flush=True)
            print(f"[INFO] Compatible-load unexpected keys: {list(resumed.unexpected_keys)}", flush=True)
    else:
        print("[INFO] No student checkpoint provided. Using structured Phase 2 default init.", flush=True)

    teacher = RmaV3ActorCritic(
        obs=_dummy_obs(policy_dim, history_dim, dynamics_dim),
        obs_groups={
            "policy": ["policy", "dynamics_privileged"],
            "critic": ["policy", "dynamics_privileged"],
        },
        num_actions=12,
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=27,
        extrinsics_encoder_hidden_dims=[],
        extrinsics_encoder_mode="linear",
        extrinsics_identity_init=True,
        dynamics_decoder_hidden_dims=[],
        dynamics_decoder_mode="linear",
        dynamics_decoder_identity_init=True,
        adaptation_hidden_dims=[256, 128],
        adaptation_bottleneck_dim=None,
        adaptation_decoder_hidden_dims=[],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        full_init_path=str(PHASE1_REFERENCE_PATH),
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    return student, teacher


def _freeze_for_offline_phi(student: RmaV3ActorCritic) -> list[torch.nn.Parameter]:
    modules_to_freeze = [student.actor, student.critic, student.extrinsics_encoder, student.dynamics_decoder]
    if student.terrain_summary_decoder is not None:
        modules_to_freeze.append(student.terrain_summary_decoder)
    for module in modules_to_freeze:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    if hasattr(student, "std"):
        student.std.requires_grad_(False)
    if hasattr(student, "log_std"):
        student.log_std.requires_grad_(False)

    trainable: list[torch.nn.Parameter] = list(student.adaptation_module.parameters())
    if student.temporal_encoder is not None:
        trainable += list(student.temporal_encoder.parameters())
    if student.history_projection is not None:
        trainable += list(student.history_projection.parameters())
    if student.adaptation_decoder is not None:
        trainable += list(student.adaptation_decoder.parameters())
    for param in trainable:
        param.requires_grad_(True)
    return trainable


def _split_indices(count: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(count)
    val_count = max(1, int(math.floor(count * val_fraction)))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    if train_idx.size == 0:
        train_idx = val_idx
    return train_idx, val_idx


def _iter_batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    rng = np.random.default_rng()
    shuffled = rng.permutation(indices)
    return [shuffled[start : start + batch_size] for start in range(0, shuffled.size, batch_size)]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = torch.clamp(weights, min=1.0e-8)
    return (values * weights).sum() / weights.sum()


def _sample_weights(arrays: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    dynamics = arrays["dynamics_privileged"][idx]
    weights = torch.ones((idx.shape[0],), device=idx.device, dtype=dynamics.dtype)

    min_friction = torch.minimum(dynamics[:, 0], dynamics[:, 1])
    low_friction_mask = min_friction < float(args_cli.low_friction_threshold)
    if float(args_cli.low_friction_upweight) != 1.0:
        weights = torch.where(
            low_friction_mask,
            weights * float(args_cli.low_friction_upweight),
            weights,
        )

    if "switch_applied" in arrays and float(args_cli.switch_upweight) != 1.0:
        switch_mask = arrays["switch_applied"][idx].squeeze(-1) > 0.5
        weights = torch.where(
            switch_mask,
            weights * float(args_cli.switch_upweight),
            weights,
        )

    base_mass_ratio = dynamics[:, 2]
    if float(args_cli.very_heavy_upweight) != 1.0:
        very_heavy_mask = base_mass_ratio > float(args_cli.very_heavy_threshold)
        weights = torch.where(
            very_heavy_mask,
            weights * float(args_cli.very_heavy_upweight),
            weights,
        )

    if float(args_cli.weak_motor_upweight) != 1.0:
        stiffness_mean = dynamics[:, 3:15].mean(dim=-1)
        damping_mean = dynamics[:, 15:27].mean(dim=-1)
        weak_motor_mask = torch.minimum(stiffness_mean, damping_mean) < float(args_cli.weak_motor_threshold)
        weights = torch.where(
            weak_motor_mask,
            weights * float(args_cli.weak_motor_upweight),
            weights,
        )

    return weights


def _compute_batch(
    student,
    teacher,
    arrays: dict[str, torch.Tensor],
    indices: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    policy_obs = arrays["policy"][idx]
    history_obs = arrays["policy_history"][idx]
    dynamics_obs = arrays["dynamics_privileged"][idx]
    teacher_latent = arrays["teacher_latent"][idx]
    teacher_action = arrays["teacher_action"][idx]

    obs = {
        "policy": policy_obs,
        "policy_history": history_obs,
        student.dynamics_group_name: dynamics_obs,
    }
    student_latent = student.encode_history_latent(obs)
    student_action = student.act_with_latent(policy_obs, student_latent)
    sample_weights = _sample_weights(arrays, idx)

    latent_loss_per_sample = (student_latent - teacher_latent).pow(2).mean(dim=-1)
    action_loss_per_sample = (student_action - teacher_action).pow(2).mean(dim=-1)
    latent_l2_per_sample = student_latent.pow(2).mean(dim=-1)

    latent_loss = _weighted_mean(latent_loss_per_sample, sample_weights)
    action_loss = _weighted_mean(action_loss_per_sample, sample_weights)
    latent_l2 = _weighted_mean(latent_l2_per_sample, sample_weights)
    cosine = F.cosine_similarity(student_latent, teacher_latent, dim=-1).mean()

    total = (
        float(args_cli.latent_coef) * latent_loss
        + float(args_cli.action_coef) * action_loss
        + float(args_cli.latent_l2_coef) * latent_l2
    )
    metrics = {
        "latent_loss": float(latent_loss.item()),
        "action_loss": float(action_loss.item()),
        "latent_l2": float(latent_l2.item()),
        "latent_cosine": float(cosine.item()),
    }
    return total, metrics


def _load_chunk_arrays(chunk_path: Path, device: str) -> dict[str, torch.Tensor]:
    raw = np.load(chunk_path)
    arrays = {}
    for key in [
        "policy",
        "policy_history",
        "dynamics_privileged",
        "teacher_latent",
        "teacher_action",
        "command_active",
        "switch_applied",
    ]:
        if key not in raw:
            continue
        arrays[key] = torch.from_numpy(raw[key]).float().to(device)
    return arrays


def _save_checkpoint(path: Path, student: RmaV3ActorCritic, summary: dict) -> None:
    payload = {
        "model_state_dict": student.state_dict(),
        "summary": summary,
    }
    torch.save(payload, path)
    print(f"[INFO] Wrote checkpoint: {path}", flush=True)


def main() -> None:
    dataset_dir = Path(args_cli.dataset_dir)
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = sorted(dataset_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.npz files found in {dataset_dir}")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    student, teacher = _build_policy_from_structured_phase2_defaults(
        chunk_paths[0],
        args_cli.device,
        args_cli.checkpoint,
    )
    trainable = _freeze_for_offline_phi(student)
    optimizer = torch.optim.Adam(trainable, lr=args_cli.lr)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, args_cli.epochs + 1):
        student.train()
        train_totals = {"total": 0.0, "latent_loss": 0.0, "action_loss": 0.0, "latent_l2": 0.0, "latent_cosine": 0.0}
        train_batches = 0

        val_totals = {"total": 0.0, "latent_loss": 0.0, "action_loss": 0.0, "latent_l2": 0.0, "latent_cosine": 0.0}
        val_batches = 0

        for chunk_idx, chunk_path in enumerate(chunk_paths):
            arrays = _load_chunk_arrays(chunk_path, args_cli.device)
            sample_count = int(arrays["policy"].shape[0])
            active_mask = arrays["command_active"].squeeze(-1) > 0.5
            indices = np.arange(sample_count)
            if args_cli.active_only:
                indices = indices[_to_numpy(active_mask)]
            if indices.size == 0:
                continue

            train_idx, val_idx = _split_indices(indices.size, args_cli.val_fraction, args_cli.seed + chunk_idx)
            mapped_train = indices[train_idx]
            mapped_val = indices[val_idx]

            for batch_indices in _iter_batches(mapped_train, args_cli.batch_size):
                optimizer.zero_grad(set_to_none=True)
                total, metrics = _compute_batch(student, teacher, arrays, batch_indices, args_cli.device)
                total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                train_totals["total"] += float(total.item())
                for key, value in metrics.items():
                    train_totals[key] += value
                train_batches += 1

            student.eval()
            with torch.no_grad():
                for batch_indices in _iter_batches(mapped_val, args_cli.batch_size):
                    total, metrics = _compute_batch(student, teacher, arrays, batch_indices, args_cli.device)
                    val_totals["total"] += float(total.item())
                    for key, value in metrics.items():
                        val_totals[key] += value
                    val_batches += 1
            student.train()

        if train_batches == 0:
            raise RuntimeError("No training batches were produced. Check dataset filters and command-active mask.")

        row = {
            "epoch": epoch,
            "train_total_loss": train_totals["total"] / train_batches,
            "train_latent_loss": train_totals["latent_loss"] / train_batches,
            "train_action_loss": train_totals["action_loss"] / train_batches,
            "train_latent_l2": train_totals["latent_l2"] / train_batches,
            "train_latent_cosine": train_totals["latent_cosine"] / train_batches,
            "val_total_loss": val_totals["total"] / max(val_batches, 1),
            "val_latent_loss": val_totals["latent_loss"] / max(val_batches, 1),
            "val_action_loss": val_totals["action_loss"] / max(val_batches, 1),
            "val_latent_l2": val_totals["latent_l2"] / max(val_batches, 1),
            "val_latent_cosine": val_totals["latent_cosine"] / max(val_batches, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        _save_checkpoint(output_dir / "last.pt", student, row)
        if row["val_total_loss"] < best_val:
            best_val = row["val_total_loss"]
            best_epoch = epoch
            _save_checkpoint(output_dir / "best.pt", student, row)

    summary = {
        "task": args_cli.task,
        "dataset_dir": str(dataset_dir),
        "student_checkpoint": args_cli.checkpoint,
        "phase1_reference_path": str(PHASE1_REFERENCE_PATH),
        "epochs": args_cli.epochs,
        "batch_size": args_cli.batch_size,
        "lr": args_cli.lr,
        "latent_coef": args_cli.latent_coef,
        "action_coef": args_cli.action_coef,
        "latent_l2_coef": args_cli.latent_l2_coef,
        "adaptation_bottleneck_dim": args_cli.adaptation_bottleneck_dim,
        "adaptation_residual": bool(args_cli.adaptation_residual),
        "active_only": bool(args_cli.active_only),
        "low_friction_threshold": args_cli.low_friction_threshold,
        "low_friction_upweight": args_cli.low_friction_upweight,
        "switch_upweight": args_cli.switch_upweight,
        "very_heavy_threshold": args_cli.very_heavy_threshold,
        "very_heavy_upweight": args_cli.very_heavy_upweight,
        "weak_motor_threshold": args_cli.weak_motor_threshold,
        "weak_motor_upweight": args_cli.weak_motor_upweight,
        "best_epoch": int(best_epoch),
        "best_val_total_loss": float(best_val),
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote summary to {output_dir / 'summary.json'}", flush=True)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


if __name__ == "__main__":
    main()
