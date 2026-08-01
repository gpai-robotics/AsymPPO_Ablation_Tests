#!/usr/bin/env python3
"""Apply a named base linear velocity source preset to a Unitree deploy.yaml."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_POLICY_DIR = Path(
    "reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/"
    "c1_blind_rough_omni_usable_v1_final/params"
)

SOURCE_LINE_RE = re.compile(r"^(?P<indent>\s*)source:\s*(?P<value>\S+)\s*$")


def _extract_source(text: str) -> str:
    in_base_lin_vel = False
    in_params = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "base_lin_vel:":
            in_base_lin_vel = True
            in_params = False
            continue
        if in_base_lin_vel and stripped and not line.startswith(" " * 6) and not line.startswith(" " * 4):
            in_base_lin_vel = False
            in_params = False
        if in_base_lin_vel and stripped == "params:":
            in_params = True
            continue
        if in_base_lin_vel and in_params:
            match = SOURCE_LINE_RE.match(line)
            if match:
                return match.group("value")
    return "<unset>"


def _replace_source(text: str, new_source: str) -> str:
    lines = text.splitlines()
    in_base_lin_vel = False
    in_params = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "base_lin_vel:":
            in_base_lin_vel = True
            in_params = False
            continue
        if in_base_lin_vel and stripped and not line.startswith(" " * 6) and not line.startswith(" " * 4):
            in_base_lin_vel = False
            in_params = False
        if in_base_lin_vel and stripped == "params:":
            in_params = True
            continue
        if in_base_lin_vel and in_params:
            match = SOURCE_LINE_RE.match(line)
            if match:
                indent = match.group("indent")
                lines[idx] = f"{indent}source: {new_source}"
                return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    raise SystemExit("Could not find observations.policy_obs.base_lin_vel.params.source in deploy.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=DEFAULT_POLICY_DIR,
        help="Path to the bundle params directory containing deploy.yaml.",
    )
    parser.add_argument(
        "--source",
        choices=("zero", "odometry"),
        required=True,
        help="Base linear velocity source to write into deploy.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would change without writing deploy.yaml.",
    )
    args = parser.parse_args()

    params_dir = args.params_dir.resolve()
    deploy_yaml = params_dir / "deploy.yaml"

    if not deploy_yaml.exists():
        raise SystemExit(f"deploy.yaml not found: {deploy_yaml}")

    original_text = deploy_yaml.read_text()
    before = _extract_source(original_text)
    updated_text = _replace_source(original_text, args.source)
    after = _extract_source(updated_text)

    print(f"[INFO] params_dir={params_dir}")
    print(f"[INFO] deploy_yaml={deploy_yaml}")
    print(f"[INFO] base_lin_vel source: {before} -> {after}")

    if args.dry_run:
        print("[INFO] Dry run only. deploy.yaml was not modified.")
        return 0

    deploy_yaml.write_text(updated_text)
    print("[INFO] deploy.yaml updated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
