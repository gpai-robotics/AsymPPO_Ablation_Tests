#!/usr/bin/env python3
"""Generate split Isaac-side and MuJoCo-side transfer scripts for C1/C2."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT_STEM = "c1_c2_transfer"
DEFAULT_C1_BUNDLE = REPO_ROOT / "rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate"
DEFAULT_C2_BUNDLE = (
    REPO_ROOT
    / "rma_go2_lab/policies/exported/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate"
)
DEFAULT_C1_TASK = "RMA-Go2-C1-ETHLike-V3-StageA"
DEFAULT_C2_TASK = "RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-stem", default=DEFAULT_SCRIPT_STEM)
    parser.add_argument("--c1-bundle-dir", default=str(DEFAULT_C1_BUNDLE))
    parser.add_argument("--c2-bundle-dir", default=str(DEFAULT_C2_BUNDLE))
    parser.add_argument("--c1-task", default=DEFAULT_C1_TASK)
    parser.add_argument("--c2-task", default=DEFAULT_C2_TASK)
    parser.add_argument("--c1-net-if", default="enp130s0")
    parser.add_argument("--c1-max-steps", type=int, default=1500)
    parser.add_argument("--c2-max-steps", type=int, default=1500)
    return parser.parse_args()


def _quote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def main() -> int:
    args = parse_args()
    output_dir = REPO_ROOT / "artifacts/pipeline_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    isaac_script_path = output_dir / f"{args.script_stem}_isaac.sh"
    mujoco_script_path = output_dir / f"{args.script_stem}_mujoco.sh"

    c1_bundle = Path(args.c1_bundle_dir)
    c2_bundle = Path(args.c2_bundle_dir)

    c1_deploy_eval = REPO_ROOT / "artifacts/deploy_eval" / c1_bundle.name
    c2_deploy_eval = REPO_ROOT / "artifacts/deploy_eval" / c2_bundle.name
    c1_mujoco_eval = REPO_ROOT / "artifacts/mujoco_eval" / c1_bundle.name
    c2_mujoco_eval = REPO_ROOT / "artifacts/mujoco_eval" / c2_bundle.name

    isaac_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"C1_BUNDLE={_quote(c1_bundle)}",
        f"C2_BUNDLE={_quote(c2_bundle)}",
        f"C1_DEPLOY_EVAL={_quote(c1_deploy_eval)}",
        f"C2_DEPLOY_EVAL={_quote(c2_deploy_eval)}",
        "",
        "mkdir -p \"$C1_DEPLOY_EVAL\" \"$C2_DEPLOY_EVAL\"",
        "",
        "echo '=== C1: Isaac deploy rehearsal ==='",
        (
            "CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p "
            f"{_quote(REPO_ROOT / 'scripts/deploy/play_deploy_policy.py')} "
            "--bundle-dir \"$C1_BUNDLE\" "
            f"--task {_quote(args.c1_task)} "
            "--num-envs 16 "
            "--max-steps 500 "
            "--seed 999 "
            "--trace-steps 100 "
            "--json-out \"$C1_DEPLOY_EVAL/isaac_deploy_rehearsal.json\""
        ),
        "",
        "echo '=== C1: hardware dry-run contract check ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_go2_hardware.py')} "
            "--bundle-dir \"$C1_BUNDLE\" "
            f"--net-if {_quote(args.c1_net_if)} "
            "--dry-run"
        ),
        "",
        "echo '=== C2: Isaac deploy rehearsal ==='",
        (
            "CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p "
            f"{_quote(REPO_ROOT / 'scripts/deploy/play_deploy_policy.py')} "
            "--bundle-dir \"$C2_BUNDLE\" "
            f"--task {_quote(args.c2_task)} "
            "--num-envs 16 "
            "--max-steps 500 "
            "--seed 999 "
            "--trace-steps 100 "
            "--compare-source "
            "--json-out \"$C2_DEPLOY_EVAL/isaac_deploy_rehearsal.json\""
        ),
        "",
        "echo",
        "echo 'Isaac-side transfer script completed.'",
        "echo 'Note: C2 sim2real is intentionally not launched here yet.'",
        "echo 'If C2 Isaac deploy did not emit a JSON, rerun that single command directly and inspect the structured exception line from play_deploy_policy.py.'",
    ]

    mujoco_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run this script only after activating the MuJoCo conda environment.",
        "",
        f"C1_BUNDLE={_quote(c1_bundle)}",
        f"C2_BUNDLE={_quote(c2_bundle)}",
        f"C1_DEPLOY_EVAL={_quote(c1_deploy_eval)}",
        f"C2_DEPLOY_EVAL={_quote(c2_deploy_eval)}",
        f"C1_MUJOCO_EVAL={_quote(c1_mujoco_eval)}",
        f"C2_MUJOCO_EVAL={_quote(c2_mujoco_eval)}",
        "",
        "mkdir -p \"$C1_DEPLOY_EVAL\" \"$C2_DEPLOY_EVAL\" \"$C1_MUJOCO_EVAL\" \"$C2_MUJOCO_EVAL\"",
        "",
        "echo '=== C1: MuJoCo nominal runtime ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_sim2sim.py')} "
            "--bundle-dir \"$C1_BUNDLE\" "
            "--execute-runtime "
            f"--max-steps {args.c1_max_steps} "
            "--trace-steps 100 "
            "--json-out \"$C1_DEPLOY_EVAL/mujoco_runtime.json\""
        ),
        "",
        "echo '=== C1: MuJoCo hidden-env suite ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_mujoco_ood_suite.py')} "
            "--bundle-dir \"$C1_BUNDLE\" "
            "--suite mujoco_hidden_env_v1 "
            "--output-dir \"$C1_MUJOCO_EVAL\""
        ),
        "",
        "echo '=== C1: MuJoCo moderate disturbance suite ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_mujoco_ood_suite.py')} "
            "--bundle-dir \"$C1_BUNDLE\" "
            "--suite mujoco_disturb_v2_moderate "
            "--output-dir \"$C1_MUJOCO_EVAL\""
        ),
        "",
        "echo '=== C1: MuJoCo rough terrain suite ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_mujoco_ood_suite.py')} "
            "--bundle-dir \"$C1_BUNDLE\" "
            "--suite mujoco_rough_v1 "
            "--output-dir \"$C1_MUJOCO_EVAL\""
        ),
        "",
        "echo '=== C2: MuJoCo nominal runtime ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_sim2sim.py')} "
            "--bundle-dir \"$C2_BUNDLE\" "
            "--execute-runtime "
            f"--max-steps {args.c2_max_steps} "
            "--trace-steps 100 "
            "--json-out \"$C2_DEPLOY_EVAL/mujoco_runtime.json\""
        ),
        "",
        "echo '=== C2: MuJoCo hidden-env suite ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_mujoco_ood_suite.py')} "
            "--bundle-dir \"$C2_BUNDLE\" "
            "--suite mujoco_hidden_env_v1 "
            "--output-dir \"$C2_MUJOCO_EVAL\""
        ),
        "",
        "echo '=== C2: MuJoCo moderate disturbance suite ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_mujoco_ood_suite.py')} "
            "--bundle-dir \"$C2_BUNDLE\" "
            "--suite mujoco_disturb_v2_moderate "
            "--output-dir \"$C2_MUJOCO_EVAL\""
        ),
        "",
        "echo '=== C2: MuJoCo rough terrain suite ==='",
        (
            "python "
            f"{_quote(REPO_ROOT / 'scripts/deploy/run_mujoco_ood_suite.py')} "
            "--bundle-dir \"$C2_BUNDLE\" "
            "--suite mujoco_rough_v1 "
            "--output-dir \"$C2_MUJOCO_EVAL\""
        ),
        "",
        "echo",
        "echo 'MuJoCo-side transfer script completed.'",
        "echo 'If you still see blocked_preflight with backend=mujoco available=False, the conda env is not the one actually running python for these commands.'",
        "echo 'The current hardware runner supports blind_history_policy only and must be extended for blind_adaptive_student first.'",
    ]

    isaac_script_path.write_text("\n".join(isaac_lines) + "\n")
    mujoco_script_path.write_text("\n".join(mujoco_lines) + "\n")
    isaac_script_path.chmod(0o755)
    mujoco_script_path.chmod(0o755)

    print(f"[INFO] Wrote Isaac transfer script to: {isaac_script_path}")
    print(f"[INFO] Wrote MuJoCo transfer script to: {mujoco_script_path}")
    print(f"[INFO] C1 bundle: {c1_bundle}")
    print(f"[INFO] C2 bundle: {c2_bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
