#!/usr/bin/env python3
"""Generate a one-command hardening/validation script for the canonical C2 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = "RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt"
)
DEFAULT_SCRIPT_NAME = "structured_z27_phase2_phi_supervised_v1_final_baseline_validation"
DEFAULT_BUNDLE_DIR = (
    REPO_ROOT
    / "rma_go2_lab/policies/exported/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate"
)
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts/final_baseline_validation/structured_z27_phase2_phi_supervised_v1/evaluations"
DEFAULT_OOD_DIR = REPO_ROOT / "artifacts/final_baseline_validation/structured_z27_phase2_phi_supervised_v1/ood_evaluations"
DEFAULT_POLICY_NAME = "adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-name", default=DEFAULT_SCRIPT_NAME)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--ood-dir", default=str(DEFAULT_OOD_DIR))
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--policy-history-length", type=int, default=20)
    parser.add_argument("--phase", default="adapt-v3-structured-offline-c2")
    parser.add_argument(
        "--freeze-note",
        default=(
            "Canonical working structured offline C2 baseline. "
            "Frozen after offline phi training beat the online PPO recovery recipe for pipeline stability."
        ),
    )
    return parser.parse_args()


def _quote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def main() -> int:
    args = parse_args()

    script_path = REPO_ROOT / "artifacts/pipeline_runs" / f"{args.script_name}.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    bundle_dir = Path(args.bundle_dir)
    eval_dir = Path(args.eval_dir)
    ood_dir = Path(args.ood_dir)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"CHECKPOINT={_quote(checkpoint)}",
        f"BUNDLE_DIR={_quote(bundle_dir)}",
        f"EVAL_DIR={_quote(eval_dir)}",
        f"OOD_DIR={_quote(ood_dir)}",
        "",
        "mkdir -p \"$BUNDLE_DIR\" \"$EVAL_DIR\" \"$OOD_DIR\"",
        "",
        "# 1. Create/update bundle manifest.",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/package_candidate.py')} "
            f"--policy-name {_quote(args.policy_name)} "
            "--source-checkpoint \"$CHECKPOINT\" "
            f"--task {_quote(args.task)} "
            f"--phase {_quote(args.phase)} "
            "--policy-kind blind_adaptive_student "
            "--observation-groups policy,policy_history "
            "--control-rate-hz 50 "
            "--bundle-dir \"$BUNDLE_DIR\" "
            f"--freeze-note {_quote(args.freeze_note)} "
            "--latent-update 'per-step history update via phi(history) -> z_hat'"
        ),
        "",
        "# 2. Export deployable artifacts for the frozen adaptive student.",
        (
            "CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p "
            f"{_quote(REPO_ROOT / 'scripts/deploy/export_policy.py')} "
            f"--policy-name {_quote(args.policy_name)} "
            "--checkpoint \"$CHECKPOINT\" "
            f"--task {_quote(args.task)} "
            f"--phase {_quote(args.phase)} "
            "--bundle-dir \"$BUNDLE_DIR\" "
            "--policy-kind blind_adaptive_student "
            "--observation-groups policy,policy_history "
            "--format torchscript "
            "--format onnx "
            f"--policy-history-length {args.policy_history_length}"
        ),
        "",
        "# 3. Validate the bundle manifest and referenced artifacts.",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/validate_bundle.py')} "
            "--bundle-dir \"$BUNDLE_DIR\""
        ),
        "",
        "# 4. Re-run the four canonical Isaac-side gates against the frozen checkpoint.",
        (
            "CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p "
            f"{_quote(REPO_ROOT / 'scripts/eval/gait.py')} "
            f"--task {_quote(args.task)} "
            "--checkpoint \"$CHECKPOINT\" "
            "--terrain-type random_rough "
            "--terrain-level 5 "
            "--command-profile forward "
            "--forced-lin-x 0.55 "
            "--forced-lin-y 0.0 "
            "--forced-ang-z 0.0 "
            "--steps 1000 "
            "--num_envs 64 "
            "--seed 999 "
            "--json-out \"$EVAL_DIR/gait_best_random_rough_l5_forward.json\""
        ),
        (
            "CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p "
            f"{_quote(REPO_ROOT / 'scripts/eval/run_isolated_suite.py')} "
            f"--task {_quote(args.task)} "
            "--checkpoint \"$CHECKPOINT\" "
            "--suite blind_baseline_v1 "
            "--output-dir \"$EVAL_DIR\""
        ),
        (
            "CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p "
            f"{_quote(REPO_ROOT / 'scripts/eval_ood/run_ood_suite.py')} "
            f"--task {_quote(args.task)} "
            "--checkpoint \"$CHECKPOINT\" "
            "--suite ood_dynamics_v1 "
            "--output-dir \"$OOD_DIR\""
        ),
        (
            "CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p "
            f"{_quote(REPO_ROOT / 'scripts/eval_ood/run_ood_suite.py')} "
            f"--task {_quote(args.task)} "
            "--checkpoint \"$CHECKPOINT\" "
            "--suite ood_switch_v1 "
            "--output-dir \"$OOD_DIR\""
        ),
        "",
        "echo",
        "echo 'Final baseline validation complete.'",
        "echo \"Bundle dir: $BUNDLE_DIR\"",
        "echo \"Eval dir:   $EVAL_DIR\"",
        "echo \"OOD dir:    $OOD_DIR\"",
    ]

    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)

    print(f"[INFO] Wrote final baseline validation script to: {script_path}")
    print(f"[INFO] Bundle dir: {bundle_dir}")
    print(f"[INFO] Eval dir:   {eval_dir}")
    print(f"[INFO] OOD dir:    {ood_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
