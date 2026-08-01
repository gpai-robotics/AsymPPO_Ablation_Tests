#!/usr/bin/env python3
"""Check MuJoCo suite summary metrics against fixed behavior thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-summary", required=True)
    parser.add_argument("--max-vel-err", type=float, default=0.45)
    parser.add_argument("--max-yaw-err", type=float, default=0.45)
    parser.add_argument("--max-tilt-xy", type=float, default=0.45)
    parser.add_argument("--min-base-height", type=float, default=0.28)
    parser.add_argument("--max-ctrl-abs", type=float, default=8.0)
    parser.add_argument("--max-non-foot-terrain-contact-step-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.suite_summary)
    payload = json.loads(summary_path.read_text())
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add(
        "suite_complete",
        payload.get("status") == "complete",
        f"status={payload.get('status')} scenario_count={payload.get('scenario_count')} expected={payload.get('expected_scenario_count')}",
    )
    for row in payload.get("results", []):
        scenario = str(row.get("scenario"))
        successful = int(row.get("successful_rollouts") or 0)
        total = int(row.get("rollout_count") or 0)
        vel_err = float(row.get("vel_err_step_mean") or 0.0)
        yaw_err = float(row.get("yaw_err_step_mean") or 0.0)
        tilt = float(row.get("base_tilt_projected_gravity_xy_mean") or 0.0)
        base_height = float(row.get("base_height_mean") or 0.0)
        ctrl_abs = float(row.get("ctrl_abs_mean") or 0.0)
        non_foot_contact_frac = float(row.get("non_foot_terrain_contact_step_fraction") or 0.0)
        add(f"{scenario}:all_rollouts_successful", successful == total and total > 0, f"successful={successful}/{total}")
        add(f"{scenario}:vel_err", vel_err <= args.max_vel_err, f"{vel_err:.4f} <= {args.max_vel_err:.4f}")
        add(f"{scenario}:yaw_err", yaw_err <= args.max_yaw_err, f"{yaw_err:.4f} <= {args.max_yaw_err:.4f}")
        add(f"{scenario}:tilt_xy", tilt <= args.max_tilt_xy, f"{tilt:.4f} <= {args.max_tilt_xy:.4f}")
        add(f"{scenario}:base_height", base_height >= args.min_base_height, f"{base_height:.4f} >= {args.min_base_height:.4f}")
        add(f"{scenario}:ctrl_abs", ctrl_abs <= args.max_ctrl_abs, f"{ctrl_abs:.4f} <= {args.max_ctrl_abs:.4f}")
        add(
            f"{scenario}:non_foot_terrain_contact_step_fraction",
            non_foot_contact_frac <= args.max_non_foot_terrain_contact_step_fraction,
            f"{non_foot_contact_frac:.4f} <= {args.max_non_foot_terrain_contact_step_fraction:.4f}",
        )

    blockers = [check for check in checks if not check["ok"]]
    report = {
        "status": "pass" if not blockers else "blocked",
        "suite_summary": str(summary_path),
        "thresholds": {
            "max_vel_err": args.max_vel_err,
            "max_yaw_err": args.max_yaw_err,
            "max_tilt_xy": args.max_tilt_xy,
            "min_base_height": args.min_base_height,
            "max_ctrl_abs": args.max_ctrl_abs,
            "max_non_foot_terrain_contact_step_fraction": args.max_non_foot_terrain_contact_step_fraction,
        },
        "checks": checks,
        "blockers": blockers,
    }
    print(json.dumps(report, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
