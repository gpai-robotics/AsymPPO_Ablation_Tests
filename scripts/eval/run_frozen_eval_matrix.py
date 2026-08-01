"""Run the full post-fix evaluation sweep for frozen policy candidates.

This script is a process-level orchestrator over the existing evaluation entry
points so the whole frozen candidate matrix can be re-run from one command.

It is intended for the "post-eval-bug-fix" pass where we want a clean, uniform
evaluation lineage for all still-valid candidates without manually pasting many
commands across multiple terminals.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")
GAIT_EVAL = REPO_ROOT / "scripts/eval/gait.py"
ISOLATED_SUITE = REPO_ROOT / "scripts/eval/run_isolated_suite.py"
OOD_SUITE = REPO_ROOT / "scripts/eval_ood/run_ood_suite.py"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/evaluations/frozen_eval_matrix_manifest.json"
DEFAULT_SECTIONS = ["gait", "blind_suite", "ood_geometry", "ood_dynamics", "ood_push", "ood_switch"]
SECTION_LABELS = {
    "gait": ["gait_standstill", "gait_forward"],
    "blind_suite": ["blind_suite"],
    "ood_geometry": ["ood_geometry_v1"],
    "ood_dynamics": ["ood_dynamics_v1"],
    "ood_push": ["ood_push_v1"],
    "ood_switch": ["ood_switch_v1"],
}


@dataclass(frozen=True)
class Candidate:
    name: str
    task: str
    checkpoint: Path
    eval_dir: Path
    ood_dir: Path
    gait_stem: str
    blind_suite_stem: str
    ood_stem_prefix: str
    has_height_scanner: bool = False


FROZEN_CANDIDATES: dict[str, Candidate] = {
    "b1": Candidate(
        name="baseline1",
        task="RMA-Go2-Blind-Baseline-Rough",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/blind_baseline1_scratch_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/baseline1",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/baseline1",
        gait_stem="gait_blind_scratch_model1999",
        blind_suite_stem="isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_blind_baseline1_scratch_final",
    ),
    "b2": Candidate(
        name="baseline2",
        task="RMA-Go2-Blind-Baseline-Rough-WarmStart",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/baseline2",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/baseline2",
        gait_stem="gait_blind_warmstart_model1500",
        blind_suite_stem="isolated_suite_model_1500_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_blind_baseline2_warmstart_final",
    ),
    "b3": Candidate(
        name="baseline3",
        task="RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/baseline3",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/baseline3",
        gait_stem="gait_blind_warmstart_imitation_model560",
        blind_suite_stem="isolated_suite_model_560_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_blind_baseline3_warmstart_imitation_final",
    ),
    "v3": Candidate(
        name="privileged_teacher_v3",
        task="RMA-Go2-Privileged-Teacher-Rough-V3",
        checkpoint=Path("/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt"),
        eval_dir=REPO_ROOT / "artifacts/evaluations/privileged_teacher_v3",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/privileged_teacher_v3",
        gait_stem="gait_privileged_teacher_v3_model1999",
        blind_suite_stem="isolated_suite_privileged_teacher_v3_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_privileged_teacher_v3",
        has_height_scanner=True,
    ),
    "na": Candidate(
        name="adaptation_student_na",
        task="RMA-Go2-Adaptation-Student-Rough-NoAdapt",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/adaptation_student_na_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/adaptation_student_na",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/adaptation_student_na",
        gait_stem="gait_student_na_model1999",
        blind_suite_stem="isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_model_1999",
    ),
    "v0": Candidate(
        name="adaptation_student_v0",
        task="RMA-Go2-Adaptation-Student-Rough-History",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/adaptation_student_v0_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/adaptation_student_v0",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/adaptation_student_v0",
        gait_stem="gait_student_adapt_v0_model1999",
        blind_suite_stem="isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_model_1999",
    ),
    # Reserved closeout candidates. These stay out of the default candidate list
    # until the corresponding canonical policy files are frozen into
    # rma_go2_lab/policies/.
    "v1": Candidate(
        name="adaptation_student_v1",
        task="RMA-Go2-Adaptation-Student-Rough-History-V1",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/adaptation_student_v1_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/adaptation_student_v1",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/adaptation_student_v1",
        gait_stem="gait_student_adapt_v1_final",
        blind_suite_stem="isolated_suite_adaptation_student_v1_final_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_adaptation_student_v1_final",
    ),
    "v2": Candidate(
        name="adaptation_student_v2",
        task="RMA-Go2-Adaptation-Student-Rough-History-V2",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/adaptation_student_v2_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/adaptation_student_v2",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/adaptation_student_v2",
        gait_stem="gait_student_adapt_v2_final",
        blind_suite_stem="isolated_suite_adaptation_student_v2_final_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_adaptation_student_v2_final",
    ),
    "v3_dyn_only": Candidate(
        name="adapt_v3_dyn_only",
        task="RMA-Go2-Adapt-V3-Phase2-StageA",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/adapt_v3_dyn_only",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/adapt_v3_dyn_only",
        gait_stem="gait_adapt_v3_dyn_only_phase2_stage_a_final",
        blind_suite_stem="isolated_suite_adapt_v3_dyn_only_phase2_stage_a_final_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_adapt_v3_dyn_only_phase2_stage_a_final",
    ),
    "v3_terrain_lite": Candidate(
        name="adapt_v3_terrain_lite",
        task="RMA-Go2-Adapt-V3-TerrainLite-Phase2-StageA",
        checkpoint=REPO_ROOT / "rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt",
        eval_dir=REPO_ROOT / "artifacts/evaluations/adapt_v3_terrain_lite",
        ood_dir=REPO_ROOT / "artifacts/ood_evaluations/adapt_v3_terrain_lite",
        gait_stem="gait_adapt_v3_terrain_lite_phase2_stage_a_final",
        blind_suite_stem="isolated_suite_adapt_v3_terrain_lite_phase2_stage_a_final_blind_baseline_v1_random_rough_levelspread_normal_seed999",
        ood_stem_prefix="ood_suite_adapt_v3_terrain_lite_phase2_stage_a_final",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full frozen evaluation matrix.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["b1", "b2", "b3", "v3", "na", "v0"],
        choices=sorted(FROZEN_CANDIDATES.keys()),
        help="Frozen candidates to evaluate. V1/V2 are reserved and should only be selected after their canonical final policy files exist.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=DEFAULT_SECTIONS,
        choices=["gait", "blind_suite", "ood_geometry", "ood_dynamics", "ood_push", "ood_switch"],
        help="Evaluation sections to run.",
    )
    parser.add_argument("--num-envs-gait", type=int, default=16)
    parser.add_argument("--steps-gait", type=int, default=200)
    parser.add_argument("--num-envs-suite", type=int, default=64)
    parser.add_argument("--steps-suite", type=int, default=1000)
    parser.add_argument("--num-envs-ood", type=int, default=64)
    parser.add_argument("--steps-ood", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _env() -> dict[str, str]:
    child_env = os.environ.copy()
    if child_env.get("TERM") in (None, "", "dumb"):
        child_env["TERM"] = "xterm"
    return child_env


def _run(cmd: list[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=_env())
    return proc.returncode


def _record(manifest: list[dict], candidate: Candidate, section: str, command: list[str], output: Path, returncode: int) -> None:
    manifest.append(
        {
            "candidate": candidate.name,
            "task": candidate.task,
            "checkpoint": str(candidate.checkpoint),
            "section": section,
            "output": str(output),
            "returncode": returncode,
            "command": command,
            "has_height_scanner": candidate.has_height_scanner,
        }
    )


def _ensure_candidate_valid(candidate: Candidate) -> None:
    if not candidate.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for candidate '{candidate.name}': {candidate.checkpoint}")
    if candidate.has_height_scanner and "Privileged-Teacher-Rough-V3" not in candidate.task:
        raise RuntimeError(
            f"Candidate '{candidate.name}' is marked as requiring height scanner but is not using the V3 teacher task."
        )


def run_gait(candidate: Candidate, args: argparse.Namespace, manifest: list[dict]) -> int:
    candidate.eval_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        ("gait_standstill", "standstill", candidate.eval_dir / f"{candidate.gait_stem}_standstill.json"),
        ("gait_forward", "forward", candidate.eval_dir / f"{candidate.gait_stem}_forward.json"),
    ]
    for section, profile, output in sections:
        if args.skip_existing and output.exists():
            print(f"[SKIP] {output}")
            _record(manifest, candidate, section, ["<skipped-existing>"], output, 0)
            continue
        cmd = [
            str(ISAACLAB),
            "-p",
            str(GAIT_EVAL),
            "--task",
            candidate.task,
            "--checkpoint",
            str(candidate.checkpoint),
            "--num_envs",
            str(args.num_envs_gait),
            "--steps",
            str(args.steps_gait),
            "--command-profile",
            profile,
            "--json-out",
            str(output),
            "--headless",
        ]
        rc = _run(cmd)
        _record(manifest, candidate, section, cmd, output, rc)
        if rc != 0 and not args.continue_on_error:
            return rc
    return 0


def run_blind_suite(candidate: Candidate, args: argparse.Namespace, manifest: list[dict]) -> int:
    candidate.eval_dir.mkdir(parents=True, exist_ok=True)
    output = candidate.eval_dir / f"{candidate.blind_suite_stem}.json"
    csv_out = candidate.eval_dir / f"{candidate.blind_suite_stem}.csv"
    cmd = [
        str(ISAACLAB),
        "-p",
        str(ISOLATED_SUITE),
        "--task",
        candidate.task,
        "--checkpoint",
        str(candidate.checkpoint),
        "--suite",
        "blind_baseline_v1",
        "--num_envs",
        str(args.num_envs_suite),
        "--steps",
        str(args.steps_suite),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(candidate.eval_dir),
        "--json-out",
        str(output),
        "--csv-out",
        str(csv_out),
    ]
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.skip_existing and output.exists():
        print(f"[SKIP] {output}")
        _record(manifest, candidate, "blind_suite", ["<skipped-existing>"], output, 0)
        return 0
    rc = _run(cmd)
    _record(manifest, candidate, "blind_suite", cmd, output, rc)
    return rc


def run_ood(candidate: Candidate, args: argparse.Namespace, manifest: list[dict], suite_name: str) -> int:
    candidate.ood_dir.mkdir(parents=True, exist_ok=True)
    output = candidate.ood_dir / f"{candidate.ood_stem_prefix}_{suite_name}_normal_seed{args.seed}.json"
    csv_out = candidate.ood_dir / f"{candidate.ood_stem_prefix}_{suite_name}_normal_seed{args.seed}.csv"
    cmd = [
        str(ISAACLAB),
        "-p",
        str(OOD_SUITE),
        "--task",
        candidate.task,
        "--checkpoint",
        str(candidate.checkpoint),
        "--suite",
        suite_name,
        "--num_envs",
        str(args.num_envs_ood),
        "--steps",
        str(args.steps_ood),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(candidate.ood_dir),
        "--json-out",
        str(output),
        "--csv-out",
        str(csv_out),
    ]
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.skip_existing and output.exists():
        print(f"[SKIP] {output}")
        _record(manifest, candidate, suite_name, ["<skipped-existing>"], output, 0)
        return 0
    rc = _run(cmd)
    _record(manifest, candidate, suite_name, cmd, output, rc)
    return rc


def _write_manifest(manifest: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _validate_matrix_completeness(manifest: list[dict], candidate_keys: list[str], sections: list[str]) -> None:
    expected = {
        (FROZEN_CANDIDATES[key].name, section_label)
        for key in candidate_keys
        for section in sections
        for section_label in SECTION_LABELS[section]
    }
    seen = {(entry["candidate"], entry["section"]) for entry in manifest}
    missing = sorted(expected - seen)
    if missing:
        rendered = ", ".join(f"{candidate}:{section}" for candidate, section in missing)
        raise RuntimeError(f"Frozen eval matrix is incomplete. Missing entries: {rendered}")

    allowed_section_labels = {label for section in sections for label in SECTION_LABELS[section]}
    bad = []
    for entry in manifest:
        if entry["candidate"] not in {FROZEN_CANDIDATES[key].name for key in candidate_keys}:
            continue
        if entry["section"] not in allowed_section_labels:
            continue
        if entry["returncode"] != 0:
            bad.append(f"{entry['candidate']}:{entry['section']} (rc={entry['returncode']})")
            continue
        if entry["command"] == ["<skipped-existing>"]:
            continue
        if not Path(entry["output"]).exists():
            bad.append(f"{entry['candidate']}:{entry['section']} (missing output)")
    if bad:
        raise RuntimeError("Frozen eval matrix finished with invalid entries: " + ", ".join(bad))


def main() -> int:
    args = parse_args()
    manifest: list[dict] = []
    candidate_name_set = {FROZEN_CANDIDATES[key].name for key in args.candidates}

    section_to_runner = {
        "gait": lambda candidate: run_gait(candidate, args, manifest),
        "blind_suite": lambda candidate: run_blind_suite(candidate, args, manifest),
        "ood_geometry": lambda candidate: run_ood(candidate, args, manifest, "ood_geometry_v1"),
        "ood_dynamics": lambda candidate: run_ood(candidate, args, manifest, "ood_dynamics_v1"),
        "ood_push": lambda candidate: run_ood(candidate, args, manifest, "ood_push_v1"),
        "ood_switch": lambda candidate: run_ood(candidate, args, manifest, "ood_switch_v1"),
    }

    try:
        for key in args.candidates:
            candidate = FROZEN_CANDIDATES[key]
            _ensure_candidate_valid(candidate)
            print(f"\n========== {candidate.name} ==========\n", flush=True)
            for section in args.sections:
                rc = section_to_runner[section](candidate)
                _write_manifest(manifest, args.manifest_out)
                if rc != 0 and not args.continue_on_error:
                    print(f"[ERROR] {candidate.name} failed in section '{section}' with return code {rc}", file=sys.stderr)
                    return rc
    finally:
        _write_manifest(manifest, args.manifest_out)

    _validate_matrix_completeness(manifest, args.candidates, args.sections)
    print(f"[INFO] Wrote manifest: {args.manifest_out}")
    print(f"[INFO] Completed a uniform matrix for candidates: {', '.join(sorted(candidate_name_set))}")
    print(f"[INFO] Sections run for every candidate: {', '.join(args.sections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
