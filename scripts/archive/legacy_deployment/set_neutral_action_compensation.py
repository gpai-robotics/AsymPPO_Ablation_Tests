#!/usr/bin/env python3
"""Toggle JointPositionAction neutral action compensation in a Unitree deploy.yaml."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_POLICY_DIR = Path(
    "reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/"
    "c1_blind_rough_omni_usable_v1_final/params"
)

ENABLED_RE = re.compile(r"^(?P<indent>\s*)enabled:\s*(?P<value>\S+)\s*$")
SCALE_RE = re.compile(r"^(?P<indent>\s*)scale:\s*(?P<value>\S+)\s*$")


def _extract_enabled(text: str) -> str:
    in_joint_position_action = False
    in_compensation = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "JointPositionAction:":
            in_joint_position_action = True
            in_compensation = False
            continue
        if in_joint_position_action and stripped and not line.startswith(" " * 4) and not line.startswith(" " * 2):
            in_joint_position_action = False
            in_compensation = False
        if in_joint_position_action and stripped == "neutral_action_compensation:":
            in_compensation = True
            continue
        if in_joint_position_action and in_compensation:
            match = ENABLED_RE.match(line)
            if match:
                return match.group("value")
    return "<unset>"


def _replace_compensation(text: str, enabled: str, scale: str | None) -> str:
    lines = text.splitlines()
    in_joint_position_action = False
    in_compensation = False
    replaced_enabled = False
    replaced_scale = scale is None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "JointPositionAction:":
            in_joint_position_action = True
            in_compensation = False
            continue
        if in_joint_position_action and stripped and not line.startswith(" " * 4) and not line.startswith(" " * 2):
            in_joint_position_action = False
            in_compensation = False
        if in_joint_position_action and stripped == "neutral_action_compensation:":
            in_compensation = True
            continue
        if in_joint_position_action and in_compensation:
            enabled_match = ENABLED_RE.match(line)
            if enabled_match:
                indent = enabled_match.group("indent")
                lines[idx] = f"{indent}enabled: {enabled}"
                replaced_enabled = True
                continue
            scale_match = SCALE_RE.match(line)
            if scale_match and scale is not None:
                indent = scale_match.group("indent")
                lines[idx] = f"{indent}scale: {scale}"
                replaced_scale = True
                continue
            if stripped and not line.startswith(" " * 6):
                in_compensation = False

    if not replaced_enabled or not replaced_scale:
        raise SystemExit("Could not fully update actions.JointPositionAction.neutral_action_compensation in deploy.yaml")

    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=DEFAULT_POLICY_DIR,
        help="Path to the bundle params directory containing deploy.yaml.",
    )
    parser.add_argument(
        "--enabled",
        choices=("true", "false"),
        required=True,
        help="Whether to enable neutral action compensation.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Optional compensation scale override.",
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
    before = _extract_enabled(original_text)
    scale_str = None if args.scale is None else f"{args.scale:g}"
    updated_text = _replace_compensation(original_text, args.enabled, scale_str)
    after = _extract_enabled(updated_text)

    print(f"[INFO] params_dir={params_dir}")
    print(f"[INFO] deploy_yaml={deploy_yaml}")
    print(f"[INFO] neutral_action_compensation.enabled: {before} -> {after}")
    if scale_str is not None:
        print(f"[INFO] neutral_action_compensation.scale := {scale_str}")

    if args.dry_run:
        print("[INFO] Dry run only. deploy.yaml was not modified.")
        return 0

    deploy_yaml.write_text(updated_text)
    print("[INFO] deploy.yaml updated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
