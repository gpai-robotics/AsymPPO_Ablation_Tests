#!/usr/bin/env python3
"""Print IsaacSim parity commands for combined AsymPPO model_5099.

This script does not launch IsaacSim. It emits copy-paste commands from the
validation manifest so GUI runs stay manual while their JSON outputs remain
standardized.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "validation" / "go2_combined_asymppo_steps_v1_model5099.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--num-envs", type=int, default=16)
    return parser.parse_args()


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _format_command(parts: list[str]) -> str:
    rendered = []
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        if part.startswith("--") and idx + 1 < len(parts) and not parts[idx + 1].startswith("--"):
            rendered.append(f"{_q(part)} {_q(parts[idx + 1])}")
            idx += 2
        else:
            rendered.append(_q(part))
            idx += 1
    return " \\\n  ".join(rendered)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = manifest["policy"]
    checkpoint = args.checkpoint or str(REPO_ROOT / policy["checkpoint"])
    task = policy["task"]
    out_root = REPO_ROOT / "artifacts" / "isaac_eval" / policy["name"]

    print("cd /home/bhuvan/projects/rma/rma_go2_lab")
    print("")
    for case in manifest["isaac_parity_cases"]:
        name = case["name"]
        command = case["command"]
        json_out = out_root / f"{name}.json"
        parts = [
            "bash",
            "scripts/isaaclab_user.sh",
            "-p",
            "scripts/eval/play_policy.py",
            "--task",
            task,
            "--checkpoint",
            checkpoint,
            "--num_envs",
            str(args.num_envs),
            "--nominal-env",
            "--terrain-type",
            case["terrain_type"],
            "--fixed-command",
            "--cmd-vx",
            str(command[0]),
            "--cmd-vy",
            str(command[1]),
            "--cmd-yaw",
            str(command[2]),
            "--max-steps",
            str(case["max_steps"]),
            "--print-env-info",
            "--eval-json-out",
            str(json_out),
        ]
        if int(case.get("terrain_level", -1)) >= 0:
            parts.extend(["--terrain-level", str(case["terrain_level"])])
        print(f"# {name}")
        print(_format_command(parts))
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
