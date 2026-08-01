#!/usr/bin/env python3
"""Watch a teacher training run and audit dependency on selected checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAACLAB = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")
DEFAULT_TASK = "RMA-Go2-Privileged-Teacher-Rough-V4"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/evaluations/teacher_dependency_watch"
DEFAULT_TERRAINS = ("random_rough", "boxes", "pyramid_stairs", "pyramid_stairs_inv")


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


def _pending_checkpoints(log_dir: Path, min_iter: int) -> list[Path]:
    ckpts = []
    for path in log_dir.glob("model_*.pt"):
        itr = _checkpoint_iter(path)
        if itr >= min_iter:
            ckpts.append((itr, path))
    ckpts.sort(key=lambda item: item[0])
    return [path for _, path in ckpts]


def _run_probe(args, checkpoint: Path, terrain_type: str, output_dir: Path) -> None:
    out = output_dir / f"teacher_v3_dependency_audit_{terrain_type}_l{args.terrain_level}_{checkpoint.stem}.json"
    cmd = [
        "python",
        str(REPO_ROOT / "scripts/eval/run_teacher_v3_dependency_suite.py"),
        "--checkpoint",
        str(checkpoint),
        "--task",
        args.task,
        "--terrain-type",
        terrain_type,
        "--terrain-level",
        str(args.terrain_level),
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--modes",
        "normal",
        "zero_terrain",
        "zero_dynamics",
        "zero_both",
        "--trace-steps",
        str(args.trace_steps),
        "--progress-every",
        str(args.progress_every),
        "--output-dir",
        str(output_dir),
        "--json-out",
        str(out),
    ]
    if args.command_x is not None:
        cmd.extend(["--command-x", str(args.command_x)])
    if args.command_y is not None:
        cmd.extend(["--command-y", str(args.command_y)])
    if args.command_yaw is not None:
        cmd.extend(["--command-yaw", str(args.command_yaw)])
    if args.headless:
        cmd.append("--headless")
    print(f"[INFO] Auditing {checkpoint.name} on terrain={terrain_type}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--terrain", action="append", dest="terrains")
    parser.add_argument("--terrain-level", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--trace-steps", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--min-checkpoint-iter", type=int, default=0)
    parser.add_argument("--poll-interval-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    terrains = tuple(args.terrains) if args.terrains else DEFAULT_TERRAINS
    run_name = log_dir.name
    output_dir = args.output_dir.resolve() / run_name
    state_path = output_dir / ".teacher_dependency_state.json"

    print(f"[INFO] Watching teacher checkpoints in: {log_dir}")
    print(f"[INFO] Terrains: {', '.join(terrains)}")
    print(f"[INFO] Output root: {output_dir}")

    while True:
        state = _load_state(state_path)
        processed = state.setdefault("processed", {})
        checkpoints = _pending_checkpoints(log_dir, args.min_checkpoint_iter)
        for checkpoint in checkpoints:
            ckpt_key = checkpoint.name
            done = set(processed.get(ckpt_key, []))
            missing = [terrain for terrain in terrains if terrain not in done]
            if not missing:
                continue
            for terrain in missing:
                try:
                    _run_probe(args, checkpoint, terrain, output_dir)
                except subprocess.CalledProcessError as exc:
                    state["last_checkpoint"] = ckpt_key
                    state["last_failed_terrain"] = terrain
                    state["last_failure_returncode"] = exc.returncode
                    state["last_updated_unix_s"] = time.time()
                    _save_state(state_path, state)
                    raise
                done.add(terrain)
                processed[ckpt_key] = sorted(done)
                state["last_checkpoint"] = ckpt_key
                state["last_failed_terrain"] = None
                state["last_failure_returncode"] = None
                state["last_updated_unix_s"] = time.time()
                _save_state(state_path, state)
        if args.once:
            break
        time.sleep(args.poll_interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
