#!/usr/bin/env python3
"""Check combined AsymPPO model_5099 MuJoCo validation summaries.

This reducer converts raw ``run_mujoco_ood_suite.py`` outputs into a concise
pass/fail report against the acceptance thresholds recorded in the validation
manifest. It intentionally does not hide failures by retuning thresholds.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "validation" / "go2_combined_asymppo_steps_v1_model5099.json"
EXPECTED_METRIC_CONTRACT = "mujoco_runtime_named_obs_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        default=False,
        help="Exit non-zero when any thresholded suite fails.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _min(values: list[float]) -> float | None:
    if not values:
        return None
    return float(min(values))


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def _threshold_check(observed: float | None, comparator: str, threshold: float) -> bool:
    if observed is None:
        return False
    if comparator == "min":
        return observed >= threshold
    if comparator == "max":
        return observed <= threshold
    raise ValueError(f"unknown comparator: {comparator}")


def _suite_observations(summary: dict[str, Any]) -> dict[str, Any]:
    rows = list(summary.get("results") or [])
    rollout_counts = [int(row.get("rollout_count") or 0) for row in rows]
    successful_counts = [int(row.get("successful_rollouts") or 0) for row in rows]
    total_rollouts = sum(rollout_counts)
    successful_rollouts = sum(successful_counts)
    success_fraction = float(successful_rollouts / total_rollouts) if total_rollouts else None
    return {
        "suite_status": summary.get("status"),
        "scenario_count": len(rows),
        "metric_contract_versions": sorted(
            {
                str(row.get("metric_contract_version"))
                for row in rows
                if row.get("metric_contract_version") not in (None, "")
            }
        ),
        "total_rollouts": total_rollouts,
        "successful_rollouts": successful_rollouts,
        "successful_rollout_fraction": success_fraction,
        "base_height_mean_min": _min(_metric_values(rows, "base_height_mean")),
        "base_tilt_projected_gravity_xy_mean_max": _max(
            _metric_values(rows, "base_tilt_projected_gravity_xy_mean")
        ),
        "vel_err_step_mean_max": _max(_metric_values(rows, "vel_err_step_mean")),
        "yaw_err_step_mean_max": _max(_metric_values(rows, "yaw_err_step_mean")),
        "joint_vel_abs_mean_max": _max(_metric_values(rows, "joint_vel_abs_mean")),
        "ctrl_abs_mean_max": _max(_metric_values(rows, "ctrl_abs_mean")),
        "score_mean": _mean(_metric_values(rows, "score_mean")),
    }


def _evaluate_suite(suite: str, summary_path: Path, thresholds: dict[str, Any] | None) -> dict[str, Any]:
    if not summary_path.exists():
        return {
            "suite": suite,
            "summary_path": str(summary_path),
            "status": "missing_summary",
            "threshold_status": "missing",
            "observed": {},
            "checks": [],
        }

    summary = _load_json(summary_path)
    observed = _suite_observations(summary)
    checks: list[dict[str, Any]] = []
    metric_versions = set(observed.get("metric_contract_versions") or [])
    if metric_versions != {EXPECTED_METRIC_CONTRACT}:
        return {
            "suite": suite,
            "summary_path": str(summary_path),
            "status": "stale_metrics",
            "threshold_status": "stale_metrics",
            "observed": observed,
            "checks": checks,
            "expected_metric_contract": EXPECTED_METRIC_CONTRACT,
        }

    if not thresholds:
        return {
            "suite": suite,
            "summary_path": str(summary_path),
            "status": "observed_only",
            "threshold_status": "not_configured",
            "observed": observed,
            "checks": checks,
        }

    specs = [
        ("successful_rollout_fraction", "min", "successful_rollout_fraction_min"),
        ("base_height_mean_min", "min", "base_height_mean_min"),
        (
            "base_tilt_projected_gravity_xy_mean_max",
            "max",
            "base_tilt_projected_gravity_xy_mean_max",
        ),
        ("vel_err_step_mean_max", "max", "vel_err_step_mean_max"),
        ("yaw_err_step_mean_max", "max", "yaw_err_step_mean_max"),
    ]
    for observed_key, comparator, threshold_key in specs:
        if threshold_key not in thresholds:
            continue
        observed_value = observed.get(observed_key)
        threshold = float(thresholds[threshold_key])
        passed = _threshold_check(observed_value, comparator, threshold)
        checks.append(
            {
                "metric": observed_key,
                "observed": observed_value,
                "comparator": comparator,
                "threshold": threshold,
                "passed": passed,
            }
        )

    passed_all = all(check["passed"] for check in checks) if checks else False
    return {
        "suite": suite,
        "summary_path": str(summary_path),
        "status": "pass" if passed_all else "fail",
        "threshold_status": "configured",
        "observed": observed,
        "checks": checks,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Combined AsymPPO Model 5099 MuJoCo Validation Report",
        "",
        f"- generated_utc: `{report['generated_utc']}`",
        f"- manifest: `{report['manifest']}`",
        f"- eval_root: `{report['eval_root']}`",
        f"- overall_status: `{report['overall_status']}`",
        "",
        "## Suite Summary",
        "",
        "| Suite | Status | Rollouts | Success Frac | Height Min | Tilt Max | Vel Err Max | Yaw Err Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suite in report["suites"]:
        obs = suite["observed"]
        rollouts = f"{obs.get('successful_rollouts', 0)}/{obs.get('total_rollouts', 0)}"
        lines.append(
            "| {suite} | {status} | {rollouts} | {succ} | {height} | {tilt} | {vel} | {yaw} |".format(
                suite=suite["suite"],
                status=suite["status"],
                rollouts=rollouts,
                succ=_fmt(obs.get("successful_rollout_fraction")),
                height=_fmt(obs.get("base_height_mean_min")),
                tilt=_fmt(obs.get("base_tilt_projected_gravity_xy_mean_max")),
                vel=_fmt(obs.get("vel_err_step_mean_max")),
                yaw=_fmt(obs.get("yaw_err_step_mean_max")),
            )
        )
    lines.extend(["", "## Failed Checks", ""])
    failed_any = False
    stale_suites = [suite for suite in report["suites"] if suite["status"] == "stale_metrics"]
    if stale_suites:
        failed_any = True
        lines.append("### Stale Metric Summaries")
        lines.append("")
        lines.append(
            f"Expected metric contract: `{EXPECTED_METRIC_CONTRACT}`. "
            "Rerun the MuJoCo suites with the current runtime before interpreting tracking or tilt metrics."
        )
        lines.append("")
        lines.append("| Suite | Existing Metric Versions |")
        lines.append("| --- | --- |")
        for suite in stale_suites:
            versions = suite["observed"].get("metric_contract_versions") or []
            lines.append(f"| `{suite['suite']}` | `{versions}` |")
        lines.append("")
    for suite in report["suites"]:
        failed = [check for check in suite["checks"] if not check["passed"]]
        if not failed:
            continue
        failed_any = True
        lines.append(f"### {suite['suite']}")
        lines.append("")
        lines.append("| Metric | Observed | Requirement |")
        lines.append("| --- | ---: | ---: |")
        for check in failed:
            sign = ">=" if check["comparator"] == "min" else "<="
            lines.append(
                f"| `{check['metric']}` | {_fmt(check['observed'])} | {sign} {_fmt(check['threshold'])} |"
            )
        lines.append("")
    if not failed_any:
        lines.append("No failed threshold checks.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `pass` means every configured suite-level threshold passed.",
            "- `fail` means at least one configured threshold failed and must be investigated or the threshold must be deliberately revised.",
            "- `stale_metrics` means summaries were generated before the current named-observation metric contract and must be rerun.",
            "- `observed_only` means the suite ran but no acceptance thresholds are configured in the manifest.",
            "- This report is intentionally stricter than visual inspection; it is meant to block vague success claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    args = parse_args()
    manifest = _load_json(args.manifest)
    eval_root = args.eval_root
    if eval_root is None:
        eval_root = REPO_ROOT / str(manifest["outputs"]["mujoco_eval_root"])
    thresholds = manifest.get("acceptance_thresholds") or {}
    suites = [suite_cfg["suite"] for suite_cfg in manifest.get("mujoco_suites", [])]

    suite_reports = []
    for suite in suites:
        summary_path = eval_root / suite / "suite_summary.json"
        suite_reports.append(_evaluate_suite(suite, summary_path, thresholds.get(suite)))

    configured = [suite for suite in suite_reports if suite["threshold_status"] == "configured"]
    missing = [suite for suite in suite_reports if suite["status"] == "missing_summary"]
    stale = [suite for suite in suite_reports if suite["status"] == "stale_metrics"]
    failed = [suite for suite in configured if suite["status"] != "pass"]
    if missing:
        overall_status = "missing_results"
    elif stale:
        overall_status = "stale_metrics"
    elif failed:
        overall_status = "fail"
    else:
        overall_status = "pass"

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "eval_root": str(eval_root),
        "overall_status": overall_status,
        "suites": suite_reports,
    }

    json_out = args.json_out or (eval_root / "model5099_validation_report.json")
    md_out = args.md_out or (eval_root / "model5099_validation_report.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_markdown(report), encoding="utf-8")

    print(f"[VALIDATION] overall_status={overall_status}")
    print(f"[VALIDATION] json={json_out}")
    print(f"[VALIDATION] markdown={md_out}")
    for suite in suite_reports:
        print(f"[VALIDATION] {suite['suite']}: {suite['status']}")

    if args.fail_on_threshold and overall_status != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
