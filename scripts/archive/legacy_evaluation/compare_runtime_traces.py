#!/usr/bin/env python3
"""Compare Isaac and MuJoCo runtime traces for the same policy.

This script is intentionally small and dependency-light:

- reads one Isaac trace JSON and one MuJoCo trace JSON
- normalizes both to a shared structure
- prints and optionally saves a compact comparison report
- optionally writes simple plots when matplotlib is available
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FOOT_LABELS = ("FL", "FR", "RL", "RR")
PAIR_NAMES = {
    "FLRR": "diag_a",
    "FRRL": "diag_b",
    "FLFRRLRR": "all4",
    "none": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-json", required=True)
    parser.add_argument("--mujoco-json", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--plot-dir", default=None)
    return parser.parse_args()


def _load_trace(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "runtime_rehearsal" in data:
        runtime_rehearsal = data["runtime_rehearsal"]
        if runtime_rehearsal is None:
            status = data.get("status", "unknown")
            blockers = data.get("blockers", [])
            raise SystemExit(
                f"MuJoCo trace did not contain a completed runtime rehearsal. "
                f"status={status} blockers={blockers}"
            )
        return runtime_rehearsal
    return data


def _contact_pattern(contact: dict[str, bool]) -> str:
    return "".join(k for k in FOOT_LABELS if contact[k]) or "none"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _summarize(trace_blob: dict) -> dict:
    summary_metrics = dict(trace_blob.get("summary_metrics", {}))
    trace = trace_blob.get("trace", [])

    pattern_counts: dict[str, int] = {}
    contact_counts = {k: 0 for k in FOOT_LABELS}
    contact_transitions = {k: 0 for k in FOOT_LABELS}
    foot_z = {k: [] for k in FOOT_LABELS}
    prev_contact = None

    for row in trace:
        contact = row["foot_contact"]
        pattern = _contact_pattern(contact)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        for foot in FOOT_LABELS:
            contact_counts[foot] += int(contact[foot])
            foot_z[foot].append(float(row["foot_pos_world"][foot][2]))
        if prev_contact is not None:
            for foot in FOOT_LABELS:
                if prev_contact[foot] != contact[foot]:
                    contact_transitions[foot] += 1
        prev_contact = contact

    top_patterns = sorted(pattern_counts.items(), key=lambda kv: kv[1], reverse=True)
    diag_a_count = pattern_counts.get("FLRR", 0)
    diag_b_count = pattern_counts.get("FRRL", 0)
    all4_count = pattern_counts.get("FLFRRLRR", 0)
    none_count = pattern_counts.get("none", 0)
    joint_vel_abs_series = []
    q_target_joint_err_series = []
    action_delta_series = []
    prev_action = None
    for row in trace:
        if "joint_vel" in row:
            joint_vel_abs_series.append(_mean([abs(float(v)) for v in row["joint_vel"]]))
        if "q_target" in row and "joint_pos" in row:
            q_target_joint_err_series.append(
                _mean([abs(float(qt) - float(qj)) for qt, qj in zip(row["q_target"], row["joint_pos"], strict=False)])
            )
        if "action" in row:
            if prev_action is None:
                action_delta_series.append(0.0)
            else:
                action_delta_series.append(
                    _mean([abs(float(a) - float(b)) for a, b in zip(row["action"], prev_action, strict=False)])
                )
            prev_action = row["action"]

    derived_summary_metrics = {
        "action_delta_mean": _mean(action_delta_series),
        "joint_vel_abs_mean": _mean(joint_vel_abs_series),
        "q_target_joint_err_mean": _mean(q_target_joint_err_series),
    }
    if trace and "applied_ctrl" in trace[0]:
        ctrl_abs_series = [_mean([abs(float(v)) for v in row["applied_ctrl"]]) for row in trace if "applied_ctrl" in row]
        ctrl_sat_series = []
        for row in trace:
            if "applied_ctrl" in row:
                ctrl = [abs(float(v)) for v in row["applied_ctrl"]]
                max_abs = max(ctrl) if ctrl else 0.0
                if max_abs <= 0.0:
                    ctrl_sat_series.append(0.0)
                else:
                    ctrl_sat_series.append(_mean([1.0 if abs(float(v)) >= (0.999 * max_abs) else 0.0 for v in row["applied_ctrl"]]))
        derived_summary_metrics["ctrl_abs_mean"] = _mean(ctrl_abs_series)
        derived_summary_metrics["ctrl_saturation_frac_mean"] = _mean(ctrl_sat_series)
    for key, value in derived_summary_metrics.items():
        summary_metrics.setdefault(key, value)

    return {
        "summary_metrics": summary_metrics,
        "trace_len": len(trace),
        "pattern_counts": pattern_counts,
        "top_patterns": top_patterns[:12],
        "diag_a_fraction": diag_a_count / max(len(trace), 1),
        "diag_b_fraction": diag_b_count / max(len(trace), 1),
        "all4_fraction": all4_count / max(len(trace), 1),
        "none_fraction": none_count / max(len(trace), 1),
        "contact_counts": contact_counts,
        "contact_transitions": contact_transitions,
        "foot_height_stats": {
            foot: {
                "mean": _mean(values),
                "min": float(min(values)) if values else 0.0,
                "max": float(max(values)) if values else 0.0,
                "range": (float(max(values)) - float(min(values))) if values else 0.0,
            }
            for foot, values in foot_z.items()
        },
        "latent_norm_series": [row["latent_norm"] for row in trace if row.get("latent_norm") is not None],
        "action_abs_series": [float(row["action_abs_mean"]) for row in trace],
        "action_delta_series": action_delta_series,
        "base_height_series": [float(row["root_height"]) for row in trace],
        "base_tilt_series": [float(row["base_tilt_xy_norm"]) for row in trace],
        "joint_vel_abs_series": joint_vel_abs_series,
        "q_target_joint_err_series": q_target_joint_err_series,
    }


def _pair(x: float) -> str:
    return f"{x:.3f}"


def _make_markdown(isaac: dict, mujoco: dict, isaac_path: Path, mujoco_path: Path) -> str:
    ism = isaac["summary_metrics"]
    msm = mujoco["summary_metrics"]
    lines = [
        "# Isaac vs MuJoCo Runtime Trace Comparison",
        "",
        f"- Isaac trace: `{isaac_path}`",
        f"- MuJoCo trace: `{mujoco_path}`",
        "",
        "## Scalar Summary",
        "",
        "| Metric | Isaac | MuJoCo |",
        "|---|---:|---:|",
    ]
    metrics = [
        "reward_proxy_mean",
        "vel_err_step_mean",
        "yaw_err_step_mean",
        "base_height_mean",
        "base_tilt_projected_gravity_xy_mean",
        "action_abs_mean",
        "action_delta_mean",
        "joint_vel_abs_mean",
        "q_target_joint_err_mean",
        "ctrl_abs_mean",
        "ctrl_saturation_frac_mean",
        "latent_norm_mean",
        "latent_norm_max",
        "latent_max_abs_mean",
        "latent_max_abs_max",
    ]
    for metric in metrics:
        lines.append(f"| {metric} | {_pair(float(ism.get(metric, 0.0)))} | {_pair(float(msm.get(metric, 0.0)))} |")

    lines += [
        "",
        "## Contact Pattern Fractions",
        "",
        "| Pattern | Isaac | MuJoCo |",
        "|---|---:|---:|",
        f"| diag_a (FL+RR) | {_pair(isaac['diag_a_fraction'])} | {_pair(mujoco['diag_a_fraction'])} |",
        f"| diag_b (FR+RL) | {_pair(isaac['diag_b_fraction'])} | {_pair(mujoco['diag_b_fraction'])} |",
        f"| all4 | {_pair(isaac['all4_fraction'])} | {_pair(mujoco['all4_fraction'])} |",
        f"| none | {_pair(isaac['none_fraction'])} | {_pair(mujoco['none_fraction'])} |",
        "",
        "## Per-Foot Contact Fraction",
        "",
        "| Foot | Isaac | MuJoCo |",
        "|---|---:|---:|",
    ]
    isaac_contact = ism.get("foot_contact_fraction", {})
    mujoco_contact = msm.get("foot_contact_fraction", {})
    for foot in FOOT_LABELS:
        lines.append(
            f"| {foot} | {_pair(float(isaac_contact.get(foot, 0.0)))} | {_pair(float(mujoco_contact.get(foot, 0.0)))} |"
        )

    lines += [
        "",
        "## Per-Foot Mean Height",
        "",
        "| Foot | Isaac | MuJoCo |",
        "|---|---:|---:|",
    ]
    isaac_h = ism.get("foot_height_mean", {})
    mujoco_h = msm.get("foot_height_mean", {})
    for foot in FOOT_LABELS:
        lines.append(
            f"| {foot} | {_pair(float(isaac_h.get(foot, 0.0)))} | {_pair(float(mujoco_h.get(foot, 0.0)))} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Higher diagonal fractions and lower all-4 fraction indicate a cleaner alternating gait.",
        "- Lower velocity/yaw error and higher base height indicate better tracking and posture.",
        "- Lower latent norms generally indicate a calmer adaptive inference path.",
    ]
    return "\n".join(lines) + "\n"


def _write_plots(plot_dir: Path, isaac: dict, mujoco: dict) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def line_plot(name: str, y1: list[float], y2: list[float], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y1, label="Isaac")
        ax.plot(y2, label="MuJoCo")
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        out = plot_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(str(out))

    line_plot("latent_norm", isaac["latent_norm_series"], mujoco["latent_norm_series"], "latent norm")
    line_plot("action_abs_mean", isaac["action_abs_series"], mujoco["action_abs_series"], "action abs mean")
    line_plot("action_delta_mean", isaac["action_delta_series"], mujoco["action_delta_series"], "action delta mean")
    line_plot("base_height", isaac["base_height_series"], mujoco["base_height_series"], "root height")
    line_plot("base_tilt_xy", isaac["base_tilt_series"], mujoco["base_tilt_series"], "tilt")
    line_plot("joint_vel_abs_mean", isaac["joint_vel_abs_series"], mujoco["joint_vel_abs_series"], "joint vel abs mean")
    line_plot(
        "q_target_joint_err_mean",
        isaac["q_target_joint_err_series"],
        mujoco["q_target_joint_err_series"],
        "q_target error mean",
    )
    return written


def main() -> int:
    args = parse_args()
    isaac_path = Path(args.isaac_json)
    mujoco_path = Path(args.mujoco_json)

    isaac_trace = _load_trace(isaac_path)
    mujoco_trace = _load_trace(mujoco_path)

    isaac_summary = _summarize(isaac_trace)
    mujoco_summary = _summarize(mujoco_trace)

    report = {
        "isaac": isaac_summary,
        "mujoco": mujoco_summary,
    }

    md = _make_markdown(isaac_summary, mujoco_summary, isaac_path, mujoco_path)
    print(md)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")

    if args.plot_dir:
        written = _write_plots(Path(args.plot_dir), isaac_summary, mujoco_summary)
        if written:
            print("Wrote plots:")
            for path in written:
                print(path)
        else:
            print("Plot generation skipped: matplotlib unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
