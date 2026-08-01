#!/usr/bin/env python3
"""Watch a training log directory and run history-ablation suites on new checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAACLAB = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")
# Default to the frozen canonical C1 branch unless a historical task is
# explicitly requested.
DEFAULT_TASK = "RMA-Go2-C1-ETHLike-V3-StageA"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/evaluations/checkpoint_history_ablation"
DEFAULT_SUITES = ("c1_history_switch_v1",)


def _checkpoint_iter(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("model_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return -1


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"processed": {}}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {"processed": {}}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")


def _run_suite(
    isaaclab_sh: Path,
    checkpoint: Path,
    task: str,
    suite: str,
    num_envs: int,
    steps: int,
    seed: int,
    output_dir: Path,
) -> None:
    cmd = [
        "env",
        "TERM=xterm",
        str(isaaclab_sh),
        "-p",
        str(REPO_ROOT / "scripts/eval/run_isolated_suite.py"),
        "--checkpoint",
        str(checkpoint),
        "--task",
        task,
        "--suite",
        suite,
        "--num_envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--headless",
        "--output-dir",
        str(output_dir),
    ]
    print(f"[INFO] Running {suite} on {checkpoint.name}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _pending_checkpoints(log_dir: Path, min_iter: int) -> list[Path]:
    ckpts = []
    for path in log_dir.glob("model_*.pt"):
        itr = _checkpoint_iter(path)
        if itr >= min_iter:
            ckpts.append((itr, path))
    ckpts.sort(key=lambda item: item[0])
    return [path for _, path in ckpts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True, help="Training run directory containing model_*.pt.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--isaaclab-sh", type=Path, default=DEFAULT_ISAACLAB)
    parser.add_argument("--suite", action="append", dest="suites", help="Suite to run. Can be passed multiple times.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--min-checkpoint-iter", type=int, default=0)
    parser.add_argument("--poll-interval-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true", help="Process current checkpoints once and exit.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.exists():
        raise SystemExit(f"Missing log dir: {log_dir}")
    if not args.isaaclab_sh.exists():
        raise SystemExit(f"Missing isaaclab launcher: {args.isaaclab_sh}")

    suites = tuple(args.suites) if args.suites else DEFAULT_SUITES
    run_name = log_dir.name
    output_dir = args.output_dir.resolve() / run_name
    state_path = output_dir / ".history_ablation_state.json"

    print(f"[INFO] Watching checkpoints in: {log_dir}")
    print(f"[INFO] Suites: {', '.join(suites)}")
    print(f"[INFO] Output root: {output_dir}")

    while True:
        state = _load_state(state_path)
        processed = state.setdefault("processed", {})
        checkpoints = _pending_checkpoints(log_dir, args.min_checkpoint_iter)

        for checkpoint in checkpoints:
            ckpt_key = checkpoint.name
            done_suites = set(processed.get(ckpt_key, []))
            missing_suites = [suite for suite in suites if suite not in done_suites]
            if not missing_suites:
                continue
            for suite in missing_suites:
                try:
                    _run_suite(
                        isaaclab_sh=args.isaaclab_sh,
                        checkpoint=checkpoint,
                        task=args.task,
                        suite=suite,
                        num_envs=args.num_envs,
                        steps=args.steps,
                        seed=args.seed,
                        output_dir=output_dir,
                    )
                except subprocess.CalledProcessError as exc:
                    state["last_checkpoint"] = ckpt_key
                    state["last_failed_suite"] = suite
                    state["last_failure_returncode"] = exc.returncode
                    state["last_updated_unix_s"] = time.time()
                    _save_state(state_path, state)
                    raise
                done_suites.add(suite)
                processed[ckpt_key] = sorted(done_suites)
                state["last_checkpoint"] = ckpt_key
                state["last_failed_suite"] = None
                state["last_failure_returncode"] = None
                state["last_updated_unix_s"] = time.time()
                _save_state(state_path, state)

        if args.once:
            break
        time.sleep(args.poll_interval_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
