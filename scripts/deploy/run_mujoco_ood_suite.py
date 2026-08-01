#!/usr/bin/env python3
"""Run structured MuJoCo-side OOD suites for deployable bundles."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from mujoco_ood_scenarios import MujocoOODScenario, scenario_set


RUN_SIM2SIM = Path(__file__).resolve().parent / "run_sim2sim.py"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "mujoco_eval"
SUITE_CHOICES = [
    "mujoco_nominal_v1",
    "mujoco_disturb_v1",
    "mujoco_disturb_v2_moderate",
    "mujoco_rough_v1",
    "mujoco_rough_v2_hard",
    "mujoco_continuous_v1",
    "mujoco_hidden_env_v1",
    "mujoco_limit_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument(
        "--suite",
        default="mujoco_limit_v1",
        choices=SUITE_CHOICES,
    )
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument(
        "--history-ablation",
        default="normal",
        choices=["normal", "zero", "frozen"],
        help="Deploy-side history ablation mode forwarded to run_sim2sim.py.",
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output root. Results will be written to <output-dir>/<bundle>/<suite>/",
    )
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--num-rollouts", type=int, default=5, help="Number of seeded rollouts per scenario.")
    parser.add_argument("--seed-base", type=int, default=1000, help="Base seed for per-rollout scenario execution.")
    parser.add_argument(
        "--reset-preset",
        default="light",
        choices=["none", "light", "moderate", "strong"],
        help="Named reset diversity preset applied before any explicit jitter overrides.",
    )
    parser.add_argument("--reset-pos-xy-jitter", type=float, default=0.02)
    parser.add_argument("--reset-yaw-jitter-deg", type=float, default=6.0)
    parser.add_argument("--reset-joint-pos-jitter", type=float, default=0.03)
    parser.add_argument("--reset-joint-vel-jitter", type=float, default=0.1)
    parser.add_argument(
        "--keep-scenario-trace",
        action="store_true",
        default=False,
        help="Keep full per-scenario trace arrays inside scenario JSON outputs.",
    )
    return parser.parse_args()


def _apply_reset_preset(args: argparse.Namespace) -> None:
    presets = {
        "none": {
            "reset_pos_xy_jitter": 0.0,
            "reset_yaw_jitter_deg": 0.0,
            "reset_joint_pos_jitter": 0.0,
            "reset_joint_vel_jitter": 0.0,
        },
        "light": {
            "reset_pos_xy_jitter": 0.02,
            "reset_yaw_jitter_deg": 6.0,
            "reset_joint_pos_jitter": 0.03,
            "reset_joint_vel_jitter": 0.1,
        },
        "moderate": {
            "reset_pos_xy_jitter": 0.05,
            "reset_yaw_jitter_deg": 12.0,
            "reset_joint_pos_jitter": 0.05,
            "reset_joint_vel_jitter": 0.2,
        },
        "strong": {
            "reset_pos_xy_jitter": 0.08,
            "reset_yaw_jitter_deg": 18.0,
            "reset_joint_pos_jitter": 0.08,
            "reset_joint_vel_jitter": 0.3,
        },
    }
    preset = presets[args.reset_preset]
    default_values = {
        "reset_pos_xy_jitter": 0.02,
        "reset_yaw_jitter_deg": 6.0,
        "reset_joint_pos_jitter": 0.03,
        "reset_joint_vel_jitter": 0.1,
    }
    for field_name, default_value in default_values.items():
        if getattr(args, field_name) == default_value:
            setattr(args, field_name, preset[field_name])


def _score(result: dict[str, object]) -> float:
    summary = result.get("summary_metrics") or {}
    reward = float(summary.get("reward_proxy_mean", 0.0))
    vel_err = float(summary.get("vel_err_step_mean", 0.0))
    yaw_err = float(summary.get("yaw_err_step_mean", 0.0))
    tilt = float(summary.get("base_tilt_projected_gravity_xy_mean", 0.0))
    post = result.get("post_event_summary") or {}
    post_vel_err = float(post.get("vel_err_step_mean", vel_err))
    post_yaw_err = float(post.get("yaw_err_step_mean", yaw_err))
    event_penalty = 0.0 if not post else (5.0 * post_vel_err + 2.0 * post_yaw_err)
    return 20.0 * reward - 10.0 * vel_err - 5.0 * yaw_err - 3.0 * tilt - event_penalty


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mu = float(sum(values) / len(values))
    var = float(sum((v - mu) ** 2 for v in values) / len(values))
    return var**0.5


def _aggregate_rollouts(rollouts: list[dict[str, object]]) -> dict[str, object]:
    metric_names = [
        "reward_proxy_mean",
        "vel_err_step_mean",
        "yaw_err_step_mean",
        "base_height_mean",
        "base_tilt_projected_gravity_xy_mean",
        "joint_vel_abs_mean",
        "ctrl_abs_mean",
    ]
    post_metric_names = [
        "reward_proxy_mean",
        "vel_err_step_mean",
        "yaw_err_step_mean",
        "base_height_mean",
        "base_tilt_projected_gravity_xy_mean",
    ]
    summary_values = {name: [] for name in metric_names}
    post_values = {name: [] for name in post_metric_names}
    scores: list[float] = []
    first_event_steps: list[int] = []

    for rollout in rollouts:
        runtime = rollout.get("runtime_rehearsal") or {}
        summary = runtime.get("summary_metrics") or {}
        post = runtime.get("post_event_summary") or {}
        for name in metric_names:
            if name in summary and summary[name] is not None:
                summary_values[name].append(float(summary[name]))
        for name in post_metric_names:
            if name in post and post[name] is not None:
                post_values[name].append(float(post[name]))
        if rollout.get("score") is not None:
            scores.append(float(rollout["score"]))
        if runtime.get("first_event_step") is not None:
            first_event_steps.append(int(runtime["first_event_step"]))

    aggregate: dict[str, object] = {
        "score_mean": _mean(scores),
        "score_std": _std(scores),
        "first_event_step_mean": _mean([float(v) for v in first_event_steps]),
        "summary_metrics_mean": {},
        "summary_metrics_std": {},
        "post_event_summary_mean": {},
        "post_event_summary_std": {},
    }
    for name in metric_names:
        aggregate["summary_metrics_mean"][name] = _mean(summary_values[name])
        aggregate["summary_metrics_std"][name] = _std(summary_values[name])
    for name in post_metric_names:
        aggregate["post_event_summary_mean"][name] = _mean(post_values[name])
        aggregate["post_event_summary_std"][name] = _std(post_values[name])
    return aggregate


def _flatten_result(row: dict[str, object]) -> dict[str, object]:
    aggregate = row.get("aggregate") or {}
    summary = aggregate.get("summary_metrics_mean") or {}
    summary_std = aggregate.get("summary_metrics_std") or {}
    post = aggregate.get("post_event_summary_mean") or {}
    post_std = aggregate.get("post_event_summary_std") or {}
    representative = row.get("representative_runtime") or {}
    representative_summary = representative.get("summary_metrics") or {}
    overrides = representative.get("runtime_overrides") or {}
    model_path = overrides.get("model_path")
    terrain_variant = ""
    terrain_family = ""
    if model_path:
        model_name = Path(str(model_path)).name
        if model_name.startswith("scene_eval_forward_rough_") and model_name.endswith(".xml"):
            terrain_variant = model_name[len("scene_eval_") : -len(".xml")]
            terrain_family = "forward_rough_family"
        elif model_name.startswith("scene_eval_forward_technical_") and model_name.endswith(".xml"):
            terrain_variant = model_name[len("scene_eval_") : -len(".xml")]
            terrain_family = "forward_technical_family"
        elif model_name.startswith("scene_eval_rough_field_") and model_name.endswith(".xml"):
            terrain_variant = model_name[len("scene_eval_rough_") : -len(".xml")]
            terrain_family = "rough_field_family"
        elif model_name == "scene_eval_rough_hfield_track.xml":
            terrain_variant = "rough_hfield_track"
            terrain_family = "rough_track_single"
        elif model_name == "scene_eval_continuous_corridor.xml":
            terrain_variant = "continuous_corridor"
            terrain_family = "continuous_corridor"
    return {
        "rank": row.get("rank"),
        "scenario": row.get("scenario"),
        "terrain_family": terrain_family,
        "terrain_variant": terrain_variant,
        "history_ablation": row.get("history_ablation"),
        "metric_contract_version": representative_summary.get("metric_contract_version"),
        "metric_source": json.dumps(representative_summary.get("metric_source") or {}, sort_keys=True),
        "score_mean": row.get("score_mean"),
        "score_std": row.get("score_std"),
        "status": row.get("status"),
        "blocker_summary": row.get("blocker_summary"),
        "successful_rollouts": row.get("successful_rollouts"),
        "rollout_count": row.get("rollout_count"),
        "reward_proxy_mean": summary.get("reward_proxy_mean"),
        "reward_proxy_std": summary_std.get("reward_proxy_mean"),
        "vel_err_step_mean": summary.get("vel_err_step_mean"),
        "vel_err_step_std": summary_std.get("vel_err_step_mean"),
        "yaw_err_step_mean": summary.get("yaw_err_step_mean"),
        "yaw_err_step_std": summary_std.get("yaw_err_step_mean"),
        "base_height_mean": summary.get("base_height_mean"),
        "base_height_std": summary_std.get("base_height_mean"),
        "base_tilt_projected_gravity_xy_mean": summary.get("base_tilt_projected_gravity_xy_mean"),
        "base_tilt_projected_gravity_xy_std": summary_std.get("base_tilt_projected_gravity_xy_mean"),
        "joint_vel_abs_mean": summary.get("joint_vel_abs_mean"),
        "joint_vel_abs_std": summary_std.get("joint_vel_abs_mean"),
        "ctrl_abs_mean": summary.get("ctrl_abs_mean"),
        "ctrl_abs_std": summary_std.get("ctrl_abs_mean"),
        "post_event_reward_proxy_mean": post.get("reward_proxy_mean"),
        "post_event_reward_proxy_std": post_std.get("reward_proxy_mean"),
        "post_event_vel_err_step_mean": post.get("vel_err_step_mean"),
        "post_event_vel_err_step_std": post_std.get("vel_err_step_mean"),
        "post_event_yaw_err_step_mean": post.get("yaw_err_step_mean"),
        "post_event_yaw_err_step_std": post_std.get("yaw_err_step_mean"),
        "first_event_step_mean": aggregate.get("first_event_step_mean"),
        "model_path": model_path,
        "ground_friction": overrides.get("ground_friction"),
        "foot_friction": overrides.get("foot_friction"),
        "base_mass_scale": overrides.get("base_mass_scale"),
        "motor_strength_scale": overrides.get("motor_strength_scale"),
        "joint_damping_scale": overrides.get("joint_damping_scale"),
        "passive_joint_damping_scale": overrides.get("passive_joint_damping_scale"),
        "scenario_dir": row.get("scenario_dir"),
        "first_failed_stdout_tail": row.get("first_failed_stdout_tail", ""),
        "first_failed_stderr_tail": row.get("first_failed_stderr_tail", ""),
    }


def _compact_runtime(runtime: dict[str, object], *, keep_trace: bool) -> dict[str, object]:
    compact = dict(runtime)
    if not keep_trace:
        compact.pop("trace", None)
    return compact


def main() -> int:
    args = parse_args()
    _apply_reset_preset(args)
    bundle_name = Path(args.bundle_dir).name
    output_root = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir).resolve()
    suite_dir_name = (
        args.suite
        if args.history_ablation == "normal"
        else f"{args.suite}__history_{args.history_ablation}"
    )
    suite_dir = output_root / bundle_name / suite_dir_name
    scenario_dir = suite_dir / "scenario_defs"
    scenario_run_dir = suite_dir / "scenario_runs"
    suite_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_run_dir.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_set(args.suite)
    results: list[dict[str, object]] = []

    for scenario in scenarios:
        scenario_json_path = scenario_dir / f"{bundle_name}_{scenario.name}.json"
        per_scenario_run_dir = scenario_run_dir / scenario.name
        per_scenario_run_dir.mkdir(parents=True, exist_ok=True)
        scenario_json_path.write_text(json.dumps(scenario.to_json_dict(), indent=2) + "\n")
        print(
            json.dumps(
                {
                    "suite_progress": "scenario_begin",
                    "suite": args.suite,
                    "scenario": scenario.name,
                    "history_ablation": args.history_ablation,
                    "num_rollouts": args.num_rollouts,
                }
            ),
            flush=True,
        )
        rollout_records: list[dict[str, object]] = []
        stop_suite = False
        for rollout_idx in range(args.num_rollouts):
            runtime_json_path = per_scenario_run_dir / f"rollout_{rollout_idx:03d}.json"
            print(
                json.dumps(
                    {
                        "suite_progress": "rollout_begin",
                        "suite": args.suite,
                        "scenario": scenario.name,
                        "history_ablation": args.history_ablation,
                        "rollout_index": rollout_idx,
                        "rollout_count": args.num_rollouts,
                        "seed": args.seed_base + rollout_idx,
                    }
                ),
                flush=True,
            )
            cmd = [
                args.python_exe,
                str(RUN_SIM2SIM),
                "--bundle-dir",
                args.bundle_dir,
                "--control-rate-hz",
                str(args.control_rate_hz),
                "--history-mode",
                "runtime",
                "--history-ablation",
                args.history_ablation,
                "--max-steps",
                str(args.max_steps),
                "--trace-steps",
                str(args.trace_steps),
                "--execute-runtime",
                "--scenario-json",
                str(scenario_json_path),
                "--seed",
                str(args.seed_base + rollout_idx),
                "--reset-pos-xy-jitter",
                str(args.reset_pos_xy_jitter),
                "--reset-yaw-jitter-deg",
                str(args.reset_yaw_jitter_deg),
                "--reset-joint-pos-jitter",
                str(args.reset_joint_pos_jitter),
                "--reset-joint-vel-jitter",
                str(args.reset_joint_vel_jitter),
                "--json-out",
                str(runtime_json_path),
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            rollout_row: dict[str, object] = {
                "rollout_index": rollout_idx,
                "seed": args.seed_base + rollout_idx,
                "returncode": completed.returncode,
                "json_path": str(runtime_json_path),
                "stderr_tail": completed.stderr[-4000:] if completed.stderr else "",
                "stdout_tail": completed.stdout[-4000:] if completed.stdout else "",
            }
            if runtime_json_path.exists():
                data = json.loads(runtime_json_path.read_text())
                runtime = data.get("runtime_rehearsal") or {}
                rollout_row["status"] = data.get("status", "")
                rollout_row["blockers"] = list(data.get("blockers") or [])
                rollout_row["checks"] = list(data.get("checks") or [])
                rollout_row["runtime_rehearsal"] = _compact_runtime(runtime, keep_trace=args.keep_scenario_trace)
                rollout_row["score"] = _score(runtime)
                if not args.keep_scenario_trace and isinstance(data.get("runtime_rehearsal"), dict):
                    data["runtime_rehearsal"] = _compact_runtime(data["runtime_rehearsal"], keep_trace=False)
                    runtime_json_path.write_text(json.dumps(data, indent=2) + "\n")
            else:
                rollout_row["status"] = "missing_json"
                rollout_row["score"] = float("-inf")
            rollout_records.append(rollout_row)
            print(
                json.dumps(
                    {
                        "suite_progress": "rollout_end",
                        "suite": args.suite,
                        "scenario": scenario.name,
                        "history_ablation": args.history_ablation,
                        "rollout_index": rollout_idx,
                        "returncode": completed.returncode,
                        "status": rollout_row.get("status"),
                    }
                ),
                flush=True,
            )
            if completed.returncode != 0 and not args.continue_on_error:
                stop_suite = True
                break

        successful = [r for r in rollout_records if int(r.get("returncode", 1)) == 0 and r.get("runtime_rehearsal")]
        statuses = [str(r.get("status", "")) for r in rollout_records if r.get("status")]
        unique_statuses = sorted(set(statuses))
        blocker_counts: dict[str, int] = {}
        for rollout in rollout_records:
            for blocker in rollout.get("blockers", []) or []:
                blocker_counts[str(blocker)] = blocker_counts.get(str(blocker), 0) + 1
        blocker_summary = sorted(
            blocker_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        aggregate = _aggregate_rollouts(successful)
        representative_runtime = successful[0].get("runtime_rehearsal") if successful else {}
        if len(successful) == len(rollout_records) and len(rollout_records) == args.num_rollouts:
            row_status = "complete"
        elif unique_statuses and len(unique_statuses) == 1 and len(successful) == 0:
            row_status = unique_statuses[0]
        else:
            row_status = "partial"
        row = {
            "scenario": scenario.name,
            "history_ablation": args.history_ablation,
            "status": row_status,
            "blocker_summary": [f"{name} ({count}/{len(rollout_records)})" for name, count in blocker_summary],
            "rollout_count": len(rollout_records),
            "successful_rollouts": len(successful),
            "score_mean": aggregate.get("score_mean", float("-inf")),
            "score_std": aggregate.get("score_std"),
            "aggregate": aggregate,
            "representative_runtime": representative_runtime,
            "scenario_dir": str(per_scenario_run_dir),
            "first_failed_stdout_tail": next(
                (str(r.get("stdout_tail", "")) for r in rollout_records if int(r.get("returncode", 1)) != 0),
                "",
            ),
            "first_failed_stderr_tail": next(
                (str(r.get("stderr_tail", "")) for r in rollout_records if int(r.get("returncode", 1)) != 0),
                "",
            ),
            "rollouts": rollout_records,
        }
        results.append(row)
        print(
            json.dumps(
                {
                    "suite_progress": "scenario_end",
                    "suite": args.suite,
                    "scenario": scenario.name,
                    "history_ablation": args.history_ablation,
                    "status": row_status,
                    "successful_rollouts": len(successful),
                    "rollout_count": len(rollout_records),
                    "score_mean": aggregate.get("score_mean", float("-inf")),
                }
            ),
            flush=True,
        )
        if stop_suite:
            break

    results.sort(
        key=lambda item: float(item.get("score_mean") if item.get("score_mean") is not None else float("-inf")),
        reverse=True,
    )
    for idx, row in enumerate(results, start=1):
        row["rank"] = idx

    csv_path = suite_dir / "suite_summary.csv"
    json_path = suite_dir / "suite_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "scenario",
                "history_ablation",
                "terrain_family",
                "terrain_variant",
                "metric_contract_version",
                "metric_source",
                "score_mean",
        "score_std",
        "status",
                "blocker_summary",
                "successful_rollouts",
                "rollout_count",
                "reward_proxy_mean",
                "reward_proxy_std",
                "vel_err_step_mean",
                "vel_err_step_std",
                "yaw_err_step_mean",
                "yaw_err_step_std",
                "base_height_mean",
                "base_height_std",
                "base_tilt_projected_gravity_xy_mean",
                "base_tilt_projected_gravity_xy_std",
                "joint_vel_abs_mean",
                "joint_vel_abs_std",
                "ctrl_abs_mean",
                "ctrl_abs_std",
                "post_event_reward_proxy_mean",
                "post_event_reward_proxy_std",
                "post_event_vel_err_step_mean",
                "post_event_vel_err_step_std",
                "post_event_yaw_err_step_mean",
                "post_event_yaw_err_step_std",
                "first_event_step_mean",
                "model_path",
                "ground_friction",
                "foot_friction",
                "base_mass_scale",
                "motor_strength_scale",
                "joint_damping_scale",
                "passive_joint_damping_scale",
                "scenario_dir",
                "first_failed_stdout_tail",
                "first_failed_stderr_tail",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(_flatten_result(row))

    payload = {
        "status": "complete" if len(results) == len(scenarios) and all(int(r.get("successful_rollouts", 0)) == int(r.get("rollout_count", -1)) for r in results) else "partial",
        "suite": args.suite,
        "bundle_dir": str(Path(args.bundle_dir)),
        "bundle_name": bundle_name,
        "history_ablation": args.history_ablation,
        "suite_dir": str(suite_dir),
        "control_rate_hz": args.control_rate_hz,
        "max_steps": args.max_steps,
        "trace_steps": args.trace_steps,
        "python_exe": args.python_exe,
        "num_rollouts": args.num_rollouts,
        "seed_base": args.seed_base,
        "reset_preset": args.reset_preset,
        "reset_pos_xy_jitter": args.reset_pos_xy_jitter,
        "reset_yaw_jitter_deg": args.reset_yaw_jitter_deg,
        "reset_joint_pos_jitter": args.reset_joint_pos_jitter,
        "reset_joint_vel_jitter": args.reset_joint_vel_jitter,
        "scenario_count": len(results),
        "expected_scenario_count": len(scenarios),
        "results": [_flatten_result(row) | {"stderr_tail": row.get("stderr_tail", ""), "stdout_tail": row.get("stdout_tail", "")} for row in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
