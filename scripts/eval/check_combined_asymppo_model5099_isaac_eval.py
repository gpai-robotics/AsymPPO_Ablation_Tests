#!/usr/bin/env python3
"""Check IsaacSim parity JSON outputs for combined AsymPPO model_5099."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "validation" / "go2_combined_asymppo_steps_v1_model5099.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Directory containing per-case IsaacSim JSON files.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--fail-on-threshold", action="store_true", default=False)
    return parser.parse_args()


def _summary_value(data: dict[str, Any], metric: str, stat: str) -> float | None:
    value = ((data.get("metrics") or {}).get(metric) or {}).get(stat)
    return None if value is None else float(value)


def _check(name: str, observed: float | int | None, comparator: str, threshold: float | int) -> dict[str, Any]:
    if observed is None:
        passed = False
    elif comparator == "max":
        passed = float(observed) <= float(threshold)
    elif comparator == "min":
        passed = float(observed) >= float(threshold)
    elif comparator == "eq":
        passed = observed == threshold
    else:
        raise ValueError(f"unknown comparator: {comparator}")
    return {
        "metric": name,
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
        "passed": passed,
    }


def _case_thresholds(case: dict[str, Any]) -> dict[str, float | int]:
    terrain = str(case["terrain_type"])
    command = case["command"]
    is_command_authority = abs(float(command[1])) > 1.0e-6 or abs(float(command[2])) > 1.0e-6
    thresholds: dict[str, float | int] = {
        "steps_min": int(case["max_steps"]),
        "reset_count_max": 0,
        "tilt_mean_max": 0.20,
        "vel_err_mean_max": 0.25 if is_command_authority else 0.30,
        "yaw_err_mean_max": 0.25,
    }
    if terrain in {"plane", "random_rough"}:
        thresholds["base_height_min_min"] = 0.25
    # World-frame height is not a fair universal criterion on slope/stair cases;
    # those cases can legitimately move up/down relative to the initial origin.
    return thresholds


def _evaluate_case(case: dict[str, Any], eval_root: Path) -> dict[str, Any]:
    path = eval_root / f"{case['name']}.json"
    if not path.exists():
        return {
            "name": case["name"],
            "status": "missing",
            "path": str(path),
            "observed": {},
            "checks": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    observed = {
        "steps": int(data.get("steps") or 0),
        "reset_count": int(data.get("reset_count") or 0),
        "base_height_mean": _summary_value(data, "base_height", "mean"),
        "base_height_min": _summary_value(data, "base_height", "min"),
        "tilt_mean": _summary_value(data, "base_tilt_projected_gravity_xy", "mean"),
        "tilt_max": _summary_value(data, "base_tilt_projected_gravity_xy", "max"),
        "vel_err_mean": _summary_value(data, "vel_err_xy", "mean"),
        "vel_err_max": _summary_value(data, "vel_err_xy", "max"),
        "yaw_err_mean": _summary_value(data, "yaw_err", "mean"),
        "yaw_err_max": _summary_value(data, "yaw_err", "max"),
        "action_abs_mean": _summary_value(data, "action_abs_mean", "mean"),
    }
    thresholds = _case_thresholds(case)
    checks = [
        _check("steps", observed["steps"], "min", thresholds["steps_min"]),
        _check("reset_count", observed["reset_count"], "max", thresholds["reset_count_max"]),
        _check("tilt_mean", observed["tilt_mean"], "max", thresholds["tilt_mean_max"]),
        _check("vel_err_mean", observed["vel_err_mean"], "max", thresholds["vel_err_mean_max"]),
        _check("yaw_err_mean", observed["yaw_err_mean"], "max", thresholds["yaw_err_mean_max"]),
    ]
    if "base_height_min_min" in thresholds:
        checks.append(_check("base_height_min", observed["base_height_min"], "min", thresholds["base_height_min_min"]))
    status = "pass" if all(check["passed"] for check in checks) else "fail"
    return {
        "name": case["name"],
        "status": status,
        "path": str(path),
        "terrain_type": case["terrain_type"],
        "terrain_level": case["terrain_level"],
        "command": case["command"],
        "observed": observed,
        "checks": checks,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Combined AsymPPO Model 5099 IsaacSim Parity Report",
        "",
        f"- generated_utc: `{report['generated_utc']}`",
        f"- manifest: `{report['manifest']}`",
        f"- eval_root: `{report['eval_root']}`",
        f"- overall_status: `{report['overall_status']}`",
        "",
        "## Case Summary",
        "",
        "| Case | Status | Terrain | Command | Steps | Resets | Height Mean | Height Min | Tilt Mean | Vel Err | Yaw Err |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        obs = case.get("observed") or {}
        lines.append(
            "| {name} | {status} | {terrain} | {command} | {steps} | {resets} | {hmean} | {hmin} | {tilt} | {vel} | {yaw} |".format(
                name=case["name"],
                status=case["status"],
                terrain=case.get("terrain_type", "n/a"),
                command=case.get("command", "n/a"),
                steps=obs.get("steps", "n/a"),
                resets=obs.get("reset_count", "n/a"),
                hmean=_fmt(obs.get("base_height_mean")),
                hmin=_fmt(obs.get("base_height_min")),
                tilt=_fmt(obs.get("tilt_mean")),
                vel=_fmt(obs.get("vel_err_mean")),
                yaw=_fmt(obs.get("yaw_err_mean")),
            )
        )
    lines.extend(["", "## Failed Checks", ""])
    failed_any = False
    for case in report["cases"]:
        failed = [check for check in case.get("checks", []) if not check["passed"]]
        if not failed:
            continue
        failed_any = True
        lines.append(f"### {case['name']}")
        lines.append("")
        lines.append("| Metric | Observed | Requirement |")
        lines.append("| --- | ---: | ---: |")
        for check in failed:
            sign = {
                "min": ">=",
                "max": "<=",
                "eq": "==",
            }[check["comparator"]]
            lines.append(f"| `{check['metric']}` | {_fmt(check['observed'])} | {sign} {_fmt(check['threshold'])} |")
        lines.append("")
    if not failed_any:
        lines.append("No failed threshold checks.")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- Stair and slope height is reported in world frame, so it is used for context rather than universal pass/fail.",
            "- Plane and random-rough cases include a base-height floor because their world height remains comparable.",
            "- This report checks numeric parity evidence; visual inspection is still useful for gait quality and terrain sanity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy_name = manifest["policy"]["name"]
    eval_root = args.eval_root or (REPO_ROOT / "artifacts" / "isaac_eval" / policy_name)
    cases = [_evaluate_case(case, eval_root) for case in manifest["isaac_parity_cases"]]
    if any(case["status"] == "missing" for case in cases):
        overall_status = "missing_results"
    elif any(case["status"] != "pass" for case in cases):
        overall_status = "fail"
    else:
        overall_status = "pass"
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "eval_root": str(eval_root),
        "overall_status": overall_status,
        "cases": cases,
    }
    json_out = args.json_out or (eval_root / "model5099_isaac_parity_report.json")
    md_out = args.md_out or (eval_root / "model5099_isaac_parity_report.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_markdown(report), encoding="utf-8")
    print(f"[ISAAC-EVAL] overall_status={overall_status}")
    print(f"[ISAAC-EVAL] json={json_out}")
    print(f"[ISAAC-EVAL] markdown={md_out}")
    for case in cases:
        print(f"[ISAAC-EVAL] {case['name']}: {case['status']}")
    if args.fail_on_threshold and overall_status != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
