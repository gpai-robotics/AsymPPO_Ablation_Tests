#!/usr/bin/env python3
"""Run teacher dependency audit one mode per Isaac process and merge results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = REPO_ROOT / "scripts/eval/audit_teacher_v3_dependency.py"
ISAACLAB = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")


def _build_mode_cmd(args: argparse.Namespace, mode: str, json_out: Path) -> list[str]:
    cmd = [
        "env",
        "TERM=xterm",
        str(ISAACLAB),
        "-p",
        str(AUDIT_SCRIPT),
        "--checkpoint",
        args.checkpoint,
        "--task",
        args.task,
        "--terrain-type",
        args.terrain_type,
        "--terrain-level",
        str(args.terrain_level),
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--modes",
        mode,
        "--trace-steps",
        str(args.trace_steps),
        "--progress-every",
        str(args.progress_every),
        "--json-out",
        str(json_out),
    ]
    if args.command_x is not None:
        cmd.extend(["--command-x", str(args.command_x)])
    if args.command_y is not None:
        cmd.extend(["--command-y", str(args.command_y)])
    if args.command_yaw is not None:
        cmd.extend(["--command-yaw", str(args.command_yaw)])
    if args.headless:
        cmd.append("--headless")
    if args.device is not None:
        cmd.extend(["--device", args.device])
    return cmd


def _extract_single_mode_result(payload: dict, mode: str) -> dict:
    """Unwrap the single-mode audit payload into the actual mode result."""
    if "results_by_mode" in payload:
        nested = payload.get("results_by_mode", {})
        if mode not in nested:
            raise KeyError(f"Mode '{mode}' missing from payload results_by_mode keys={list(nested.keys())}")
        return nested[mode]
    return payload


def _assert_expected_terrain(payload: dict, expected_terrain: str) -> None:
    actual = payload.get("terrain_type")
    if actual != expected_terrain:
        raise RuntimeError(
            f"Terrain mismatch in audit payload: expected '{expected_terrain}', got '{actual}'. "
            "This run should not be trusted; rerun the audit after resolving the terrain-selection issue."
        )


def _pairwise_trace_diffs(results_by_mode: dict[str, dict]) -> dict[str, dict[str, float]]:
    normal = results_by_mode.get("normal")
    if not normal:
        return {}
    normal_trace = normal.get("trace", [])
    out: dict[str, dict[str, float]] = {}
    for mode, result in results_by_mode.items():
        if mode == "normal":
            continue
        other_trace = result.get("trace", [])
        count = min(len(normal_trace), len(other_trace))
        if count == 0:
            out[mode] = {"mean_action_abs_diff_vs_normal": 0.0}
            continue
        diffs = []
        for idx in range(count):
            a = normal_trace[idx]["action"]
            b = other_trace[idx]["action"]
            diffs.append(sum(abs(x - y) for x, y in zip(a, b)) / max(len(a), 1))
        out[mode] = {"mean_action_abs_diff_vs_normal": float(sum(diffs) / len(diffs))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="RMA-Go2-Privileged-Teacher-Rough-V3")
    parser.add_argument("--terrain-type", type=str, default="random_rough")
    parser.add_argument("--terrain-level", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["normal", "zero_terrain", "zero_dynamics", "zero_both"],
    )
    parser.add_argument("--trace-steps", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--command-x", type=float, default=None)
    parser.add_argument("--command-y", type=float, default=None)
    parser.add_argument("--command-yaw", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_slug = str(args.terrain_type).replace("/", "_")
    stem = f"teacher_v3_dependency_suite_{Path(args.checkpoint).stem}_{terrain_slug}_l{args.terrain_level}"

    results_by_mode: dict[str, dict] = {}
    for mode in args.modes:
        mode_json = output_dir / f"{stem}_{mode}.json"
        cmd = _build_mode_cmd(args, mode, mode_json)
        print(f"[INFO] Running mode in isolated process: {mode}", flush=True)
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        payload = json.loads(mode_json.read_text(encoding="utf-8"))
        _assert_expected_terrain(payload, args.terrain_type)
        results_by_mode[mode] = _extract_single_mode_result(payload, mode)

    summary = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "terrain_type": args.terrain_type,
        "terrain_level": args.terrain_level,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "modes": args.modes,
        "results_by_mode": results_by_mode,
        "pairwise_trace_diffs_vs_normal": _pairwise_trace_diffs(results_by_mode),
    }
    json_out = Path(args.json_out) if args.json_out else output_dir / f"{stem}_combined.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote combined JSON to: {json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
