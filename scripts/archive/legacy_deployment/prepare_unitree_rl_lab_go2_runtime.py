#!/usr/bin/env python3
"""Stage a frozen Go2 bundle into the local unitree_rl_lab Go2 runtime tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from materialize_unitree_rl_lab_layout import materialize_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GO2_DEPLOY_DIR = REPO_ROOT / "reference_repos" / "unitree_rl_lab" / "deploy" / "robots" / "go2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--go2-deploy-dir", default=str(DEFAULT_GO2_DEPLOY_DIR))
    parser.add_argument(
        "--policy-slot",
        default="velocity",
        help="Policy slot under config/policy/ used by the Go2 deploy config.",
    )
    parser.add_argument(
        "--runtime-name",
        default=None,
        help="Optional runtime directory name. Defaults to bundle policy_name.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing staged runtime directory.")
    return parser.parse_args()


def _infer_runtime_name(bundle_dir: Path) -> str:
    manifest_path = bundle_dir / "bundle_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("policy_name"):
            return str(manifest["policy_name"])
    return bundle_dir.name


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    go2_deploy_dir = Path(args.go2_deploy_dir).resolve()
    runtime_name = args.runtime_name or _infer_runtime_name(bundle_dir)

    output_dir = go2_deploy_dir / "config" / "policy" / args.policy_slot / runtime_name
    materialize_bundle(bundle_dir, output_dir, robot="go2", force=args.force)

    print(f"Staged Go2 runtime at: {output_dir}")
    print("Recommended policy_dir for reference_repos/unitree_rl_lab/deploy/robots/go2/config/config.yaml:")
    print(f"  config/policy/{args.policy_slot}")
    print("Next steps:")
    print(f"  1. Ensure config.yaml points Velocity.policy_dir at config/policy/{args.policy_slot}")
    print("  2. Build go2_ctrl")
    print("  3. Validate FixStand before entering Velocity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
