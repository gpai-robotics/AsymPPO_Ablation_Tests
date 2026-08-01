#!/usr/bin/env python3
"""Strictly verify that Isaac and MuJoCo reports used the same validation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MATCHED_FIELDS = (
    "control_rate_hz",
    "physics_dt_s",
    "control_dt_s",
    "max_steps",
    "warmup_steps",
    "seed_base",
    "rollouts_per_scenario",
    "reset_preset",
    "reset_pos_xy_jitter",
    "reset_yaw_jitter_deg",
    "reset_joint_pos_jitter",
    "reset_joint_vel_jitter",
    "obs_delay_steps",
    "action_delay_steps",
    "command_delay_steps",
    "actuator_model",
    "scenario_names",
    "terrain_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-report", type=Path, required=True)
    parser.add_argument("--mujoco-report", type=Path, required=True)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/diagnostics/go2_crosssim_report_parity.json"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1.0e-9
    return left == right


def main() -> int:
    args = parse_args()
    isaac = _load(args.isaac_report)
    mujoco = _load(args.mujoco_report)
    isaac_contract = isaac.get("crosssim_contract") or {}
    mujoco_contract = mujoco.get("crosssim_contract") or {}
    isaac_resolved = isaac.get("resolved_evaluation") or {}
    mujoco_resolved = mujoco.get("resolved_evaluation") or {}

    checks: list[dict[str, Any]] = []

    def check(name: str, left: Any, right: Any) -> None:
        checks.append(
            {
                "name": name,
                "ok": _equal(left, right),
                "isaac": left,
                "mujoco": right,
            }
        )

    check("profile_name", isaac_contract.get("profile_name"), mujoco_contract.get("profile_name"))
    check("profile_sha256", isaac_contract.get("profile_sha256"), mujoco_contract.get("profile_sha256"))
    check("scenario_sha256", isaac_contract.get("scenario_sha256"), mujoco_contract.get("scenario_sha256"))
    for field in MATCHED_FIELDS:
        check(f"resolved/{field}", isaac_resolved.get(field), mujoco_resolved.get(field))

    failures = [item for item in checks if not item["ok"]]
    report = {
        "status": "comparable" if not failures else "not_comparable",
        "isaac_report": str(args.isaac_report.resolve()),
        "mujoco_report": str(args.mujoco_report.resolve()),
        "failure_count": len(failures),
        "checks": checks,
        "important_conclusion": (
            "Comparable means the controllable experiment contract matched. "
            "It does not mean PhysX and MuJoCo physics are numerically identical."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
