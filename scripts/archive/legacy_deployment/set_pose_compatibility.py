#!/usr/bin/env python3
"""Toggle pose compatibility mode and blend in a Unitree deploy.yaml."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_POLICY_DIR = Path(
    "reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/"
    "c1_blind_rough_omni_usable_v1_final/params"
)

ENABLED_RE = re.compile(r"^(?P<indent>\s*)enabled:\s*(?P<value>\S+)\s*$")
BLEND_RE = re.compile(r"^(?P<indent>\s*)blend:\s*(?P<value>\S+)\s*$")


def _extract_enabled(text: str) -> str:
    in_pose_compat = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "pose_compatibility:":
            in_pose_compat = True
            continue
        if in_pose_compat and stripped and not line.startswith(" " * 2):
            in_pose_compat = False
        if in_pose_compat:
            match = ENABLED_RE.match(line)
            if match:
                return match.group("value")
    return "<unset>"


def _extract_blend(text: str) -> str:
    in_pose_compat = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "pose_compatibility:":
            in_pose_compat = True
            continue
        if in_pose_compat and stripped and not line.startswith(" " * 2):
            in_pose_compat = False
        if in_pose_compat:
            match = BLEND_RE.match(line)
            if match:
                return match.group("value")
    return "<unset>"


def _replace_pose_compat(text: str, enabled: str, blend: str | None) -> str:
    lines = text.splitlines()
    in_pose_compat = False
    enabled_replaced = False
    blend_replaced = blend is None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "pose_compatibility:":
            in_pose_compat = True
            continue
        if in_pose_compat and stripped and not line.startswith(" " * 2):
            in_pose_compat = False
        if in_pose_compat:
            enabled_match = ENABLED_RE.match(line)
            if enabled_match:
                indent = enabled_match.group("indent")
                lines[idx] = f"{indent}enabled: {enabled}"
                enabled_replaced = True
                continue
            blend_match = BLEND_RE.match(line)
            if blend_match and blend is not None:
                indent = blend_match.group("indent")
                lines[idx] = f"{indent}blend: {blend}"
                blend_replaced = True
                continue

    if not enabled_replaced:
        raise SystemExit("Could not find pose_compatibility.enabled in deploy.yaml")
    if not blend_replaced:
        raise SystemExit("Could not find pose_compatibility.blend in deploy.yaml")

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
        help="Whether to enable pose compatibility mode.",
    )
    parser.add_argument(
        "--blend",
        type=float,
        help="Blend from current bundle pose (0.0) to compat pose (1.0).",
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
    before_blend = _extract_blend(original_text)
    blend_value = None
    if args.blend is not None:
        if not (0.0 <= args.blend <= 1.0):
            raise SystemExit("--blend must be within [0.0, 1.0]")
        blend_value = f"{args.blend:.3f}".rstrip("0").rstrip(".")
    updated_text = _replace_pose_compat(original_text, args.enabled, blend_value)
    after = _extract_enabled(updated_text)
    after_blend = _extract_blend(updated_text)

    print(f"[INFO] params_dir={params_dir}")
    print(f"[INFO] deploy_yaml={deploy_yaml}")
    print(f"[INFO] pose_compatibility.enabled: {before} -> {after}")
    if blend_value is not None:
        print(f"[INFO] pose_compatibility.blend: {before_blend} -> {after_blend}")

    if args.dry_run:
        print("[INFO] Dry run only. deploy.yaml was not modified.")
        return 0

    deploy_yaml.write_text(updated_text)
    print("[INFO] deploy.yaml updated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
