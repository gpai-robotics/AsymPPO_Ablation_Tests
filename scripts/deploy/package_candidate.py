#!/usr/bin/env python3
"""Create or update a deployment bundle manifest for a frozen candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--policy-kind",
        required=True,
        choices=[
            "blind_fixed_policy",
            "blind_history_policy",
            "blind_adaptive_student",
            "privileged_base_only",
        ],
    )
    parser.add_argument(
        "--observation-groups",
        required=True,
        help="Comma-separated deployable observation groups.",
    )
    parser.add_argument("--control-rate-hz", required=True, type=float)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--freeze-note", default="")
    parser.add_argument(
        "--latent-update",
        default="",
        help="Short note for adaptive policies, for example per-step history update.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_name": args.policy_name,
        "source_checkpoint": str(Path(args.source_checkpoint)),
        "task": args.task,
        "phase": args.phase,
        "policy_kind": args.policy_kind,
        "deployable_observation_groups": parse_csv(args.observation_groups),
        "control_rate_hz": args.control_rate_hz,
        "latent_update_semantics": args.latent_update,
        "freeze_note": args.freeze_note,
        "exported_artifacts": [],
    }
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote bundle manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
