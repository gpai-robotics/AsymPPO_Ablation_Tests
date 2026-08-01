"""Generate a runnable command script for the structured offline C2 pipeline.

This keeps the working collect/train/eval cycle reproducible without requiring
people to manually reassemble the command blocks from docs.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_SH = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")
DEFAULT_TASK = "RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt"
)


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _isaac_prefix(gpu: int) -> str:
    return f"CUDA_VISIBLE_DEVICES={gpu} env TERM=xterm {_q(ISAACLAB_SH)} -p"


def _append_optional_arg(parts: list[str], flag: str, value) -> None:
    if value is None:
        return
    parts.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a structured offline C2 cycle shell script.")
    parser.add_argument("--cycle-name", required=True, help="Short name for this cycle, e.g. structured_z27_phase2_phi_supervised_v4.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--student-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--chunk-steps", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--latent-coef", type=float, default=1.0)
    parser.add_argument("--action-coef", type=float, default=0.10)
    parser.add_argument("--latent-l2-coef", type=float, default=1.0e-3)
    parser.add_argument("--adaptation-bottleneck-dim", type=int, default=None)
    parser.add_argument("--adaptation-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--active-only", action="store_true", default=True)
    parser.add_argument("--low-friction-threshold", type=float, default=None)
    parser.add_argument("--low-friction-upweight", type=float, default=None)
    parser.add_argument("--switch-upweight", type=float, default=None)
    parser.add_argument("--very-heavy-threshold", type=float, default=None)
    parser.add_argument("--very-heavy-upweight", type=float, default=None)
    parser.add_argument("--weak-motor-threshold", type=float, default=None)
    parser.add_argument("--weak-motor-upweight", type=float, default=None)
    parser.add_argument(
        "--output-script",
        default=None,
        help="Optional explicit path for the generated shell script. Defaults to artifacts/pipeline_runs/<cycle-name>.sh.",
    )
    args = parser.parse_args()

    dataset_dir = REPO_ROOT / "artifacts/datasets" / f"{args.cycle_name}_dataset"
    model_dir = REPO_ROOT / "artifacts/models" / args.cycle_name
    eval_dir = REPO_ROOT / "artifacts/evaluations" / args.cycle_name
    ood_dir = REPO_ROOT / "artifacts/ood_evaluations" / args.cycle_name
    script_path = (
        Path(args.output_script)
        if args.output_script is not None
        else REPO_ROOT / "artifacts/pipeline_runs" / f"{args.cycle_name}.sh"
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = _isaac_prefix(args.gpu)

    collect_cmd = [
        prefix,
        _q(REPO_ROOT / "scripts/adaptation/collect_structured_phase2_onpolicy_dataset.py"),
        "--task",
        _q(args.task),
        "--checkpoint",
        _q(args.student_checkpoint),
        "--output-dir",
        _q(dataset_dir),
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--chunk-steps",
        str(args.chunk_steps),
        "--seed",
        str(args.seed),
        "--headless",
    ]
    _append_optional_arg(collect_cmd, "--adaptation-bottleneck-dim", args.adaptation_bottleneck_dim)
    if args.adaptation_residual:
        collect_cmd.append("--adaptation-residual")

    train_parts = [
        prefix,
        _q(REPO_ROOT / "scripts/adaptation/train_structured_phase2_phi_supervised.py"),
        "--task",
        _q(args.task),
        "--dataset-dir",
        _q(dataset_dir),
        "--output-dir",
        _q(model_dir),
        "--checkpoint",
        _q(args.student_checkpoint),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--latent-coef",
        str(args.latent_coef),
        "--action-coef",
        str(args.action_coef),
        "--latent-l2-coef",
        str(args.latent_l2_coef),
        "--seed",
        str(args.seed),
    ]
    _append_optional_arg(train_parts, "--adaptation-bottleneck-dim", args.adaptation_bottleneck_dim)
    if args.adaptation_residual:
        train_parts.append("--adaptation-residual")
    if args.active_only:
        train_parts.append("--active-only")
    _append_optional_arg(train_parts, "--low-friction-threshold", args.low_friction_threshold)
    _append_optional_arg(train_parts, "--low-friction-upweight", args.low_friction_upweight)
    _append_optional_arg(train_parts, "--switch-upweight", args.switch_upweight)
    _append_optional_arg(train_parts, "--very-heavy-threshold", args.very_heavy_threshold)
    _append_optional_arg(train_parts, "--very-heavy-upweight", args.very_heavy_upweight)
    _append_optional_arg(train_parts, "--weak-motor-threshold", args.weak_motor_threshold)
    _append_optional_arg(train_parts, "--weak-motor-upweight", args.weak_motor_upweight)

    best_pt = model_dir / "best.pt"

    gait_cmd = [
        prefix,
        _q(REPO_ROOT / "scripts/eval/gait.py"),
        "--task",
        _q(args.task),
        "--checkpoint",
        _q(best_pt),
        "--terrain-type",
        "random_rough",
        "--terrain-level",
        "5",
        "--command-profile",
        "forward",
        "--forced-lin-x",
        "0.55",
        "--forced-lin-y",
        "0.0",
        "--forced-ang-z",
        "0.0",
        "--steps",
        "1000",
        "--num_envs",
        "64",
        "--seed",
        str(args.seed),
        "--json-out",
        _q(eval_dir / "gait_best_random_rough_l5_forward.json"),
    ]
    _append_optional_arg(gait_cmd, "--adaptation-bottleneck-dim", args.adaptation_bottleneck_dim)
    if args.adaptation_residual:
        gait_cmd.append("--adaptation-residual")

    blind_cmd = [
        prefix,
        _q(REPO_ROOT / "scripts/eval/run_isolated_suite.py"),
        "--task",
        _q(args.task),
        "--checkpoint",
        _q(best_pt),
        "--suite",
        "blind_baseline_v1",
        "--output-dir",
        _q(eval_dir),
    ]
    _append_optional_arg(blind_cmd, "--adaptation-bottleneck-dim", args.adaptation_bottleneck_dim)
    if args.adaptation_residual:
        blind_cmd.append("--adaptation-residual")

    dyn_cmd = [
        prefix,
        _q(REPO_ROOT / "scripts/eval_ood/run_ood_suite.py"),
        "--task",
        _q(args.task),
        "--checkpoint",
        _q(best_pt),
        "--suite",
        "ood_dynamics_v1",
        "--output-dir",
        _q(ood_dir),
    ]
    _append_optional_arg(dyn_cmd, "--adaptation-bottleneck-dim", args.adaptation_bottleneck_dim)
    if args.adaptation_residual:
        dyn_cmd.append("--adaptation-residual")

    switch_cmd = [
        prefix,
        _q(REPO_ROOT / "scripts/eval_ood/run_ood_suite.py"),
        "--task",
        _q(args.task),
        "--checkpoint",
        _q(best_pt),
        "--suite",
        "ood_switch_v1",
        "--output-dir",
        _q(ood_dir),
    ]
    _append_optional_arg(switch_cmd, "--adaptation-bottleneck-dim", args.adaptation_bottleneck_dim)
    if args.adaptation_residual:
        switch_cmd.append("--adaptation-residual")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Auto-generated structured offline C2 cycle: {args.cycle_name}",
        f"# Student checkpoint: {args.student_checkpoint}",
        "",
        f"mkdir -p {_q(dataset_dir)} {_q(model_dir)} {_q(eval_dir)} {_q(ood_dir)}",
        "",
        "# 1. Collect on-policy adaptation dataset",
        " \\\n  ".join(collect_cmd),
        "",
        "# 2. Train phi offline",
        " \\\n  ".join(train_parts),
        "",
        "# 3. Evaluate gait",
        " \\\n  ".join(gait_cmd),
        "",
        "# 4. Evaluate blind nominal suite",
        " \\\n  ".join(blind_cmd),
        "",
        "# 5. Evaluate OOD dynamics suite",
        " \\\n  ".join(dyn_cmd),
        "",
        "# 6. Evaluate OOD switch suite",
        " \\\n  ".join(switch_cmd),
        "",
    ]

    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)

    print(f"[INFO] Wrote structured offline cycle script to: {script_path}")
    print(f"[INFO] Dataset dir: {dataset_dir}")
    print(f"[INFO] Model dir:   {model_dir}")
    print(f"[INFO] Eval dir:    {eval_dir}")
    print(f"[INFO] OOD dir:     {ood_dir}")


if __name__ == "__main__":
    main()
