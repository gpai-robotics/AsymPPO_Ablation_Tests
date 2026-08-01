#!/usr/bin/env python3
"""Compare flat omni priors on a fixed command schedule."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")
EVAL_SCRIPT = REPO_ROOT / "scripts/eval/eval_flat_omni_schedule.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-task", default="RMA-Go2-Flat-Omni-V1")
    parser.add_argument("--left-checkpoint", required=True)
    parser.add_argument("--left-name", default="flat_omni_v1")
    parser.add_argument("--right-task", default="RMA-Go2-Flat-Omni-Contact-V1")
    parser.add_argument("--right-checkpoint", required=True)
    parser.add_argument("--right-name", default="flat_omni_contact_v1")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--steps-per-segment", type=int, default=180)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts/evaluations/flat_omni_prior_compare"))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def _run_eval(task: str, checkpoint: str, name: str, output_json: Path, args: argparse.Namespace) -> None:
    cmd = [
        str(ISAACLAB),
        "-p",
        str(EVAL_SCRIPT),
        "--task",
        task,
        "--checkpoint",
        checkpoint,
        "--name",
        name,
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--warmup-steps",
        str(args.warmup_steps),
        "--json-out",
        str(output_json),
    ]
    if args.headless:
        cmd.append("--headless")
    env = dict(**__import__("os").environ)
    env.setdefault("TERM", "xterm")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _segment_map(blob: dict) -> dict[str, dict]:
    return {seg["segment"]: seg for seg in blob["segments"]}


def _overall(blob: dict) -> dict[str, float]:
    segments = blob["segments"]
    keys = [
        "vel_err_xy_mean",
        "vel_err_yaw_mean",
        "root_height_mean",
        "base_tilt_mean",
        "action_abs_mean",
        "foot_slide_proxy_mean",
        "done_count",
    ]
    out: dict[str, float] = {}
    for key in keys:
        values = [float(seg["metrics"][key]) for seg in segments]
        out[key] = float(sum(values) / len(values)) if values else 0.0
    return out


def _make_markdown(left: dict, right: dict, args: argparse.Namespace) -> str:
    left_name = left["name"]
    right_name = right["name"]
    left_overall = _overall(left)
    right_overall = _overall(right)
    left_segments = _segment_map(left)
    right_segments = _segment_map(right)

    lines = [
        "# Flat Omni Prior Comparison",
        "",
        f"- left: `{left_name}`",
        f"- right: `{right_name}`",
        "",
        "## Overall Means",
        "",
        f"| Metric | {left_name} | {right_name} |",
        "|---|---:|---:|",
    ]
    for key in ["vel_err_xy_mean", "vel_err_yaw_mean", "foot_slide_proxy_mean", "root_height_mean", "base_tilt_mean", "done_count"]:
        lines.append(f"| {key} | {left_overall[key]:.4f} | {right_overall[key]:.4f} |")

    lines += [
        "",
        "## Per-Segment Metrics",
        "",
        f"| Segment | Metric | {left_name} | {right_name} |",
        "|---|---|---:|---:|",
    ]
    metric_keys = ["vel_err_xy_mean", "vel_err_yaw_mean", "foot_slide_proxy_mean", "root_height_mean", "done_count"]
    for segment in [seg["segment"] for seg in left["segments"]]:
        for key in metric_keys:
            lines.append(
                f"| {segment} | {key} | {left_segments[segment]['metrics'][key]:.4f} | {right_segments[segment]['metrics'][key]:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_json = output_dir / f"{args.left_name}.json"
    right_json = output_dir / f"{args.right_name}.json"

    _run_eval(args.left_task, args.left_checkpoint, args.left_name, left_json, args)
    _run_eval(args.right_task, args.right_checkpoint, args.right_name, right_json, args)

    left = _load(left_json)
    right = _load(right_json)

    report = {
        "left": left,
        "right": right,
        "left_overall": _overall(left),
        "right_overall": _overall(right),
    }
    markdown = _make_markdown(left, right, args)
    print(markdown)

    json_out = Path(args.json_out) if args.json_out else output_dir / "comparison.json"
    md_out = Path(args.md_out) if args.md_out else output_dir / "comparison.md"
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(markdown, encoding="utf-8")
    print(f"[INFO] Wrote {json_out}")
    print(f"[INFO] Wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
