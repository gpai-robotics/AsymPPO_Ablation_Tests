#!/usr/bin/env python3
"""Create presentation-ready IsaacLab vs MuJoCo velocity-tracking plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-json", required=True)
    parser.add_argument("--mujoco-json", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-svg", default="")
    parser.add_argument("--title", default="C1 Velocity Tracking")
    parser.add_argument("--left-label", default="IsaacLab")
    parser.add_argument("--right-label", default="Deployment")
    parser.add_argument(
        "--forward-only",
        action="store_true",
        default=False,
        help="Plot only commanded vs realized forward velocity.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "C1 blind-reactive candidate: commanded vs realized base velocity "
            "on best nominal IsaacLab and deployment-side MuJoCo traces"
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=20.0)
    return parser.parse_args()


def _load_trace(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    if "series" in blob:
        series = blob["series"]
        time_s = series["time_s"]
        return [
            {
                "step": int(round(float(t) / 0.02)),
                "command": [float(series["cmd_vx"][idx]), 0.0, 0.0],
                "base_lin_vel_local": [
                    float(series["vx"][idx]),
                    float(series.get("vy", [0.0] * len(time_s))[idx]),
                    0.0,
                ],
                "base_ang_vel_local": [0.0, 0.0, float(series.get("yaw_rate", [0.0] * len(time_s))[idx])],
            }
            for idx, t in enumerate(time_s)
        ]
    if "trace" in blob:
        return blob["trace"]
    runtime = blob.get("runtime_rehearsal", {})
    return runtime.get("trace", [])


def _series(trace: list[dict], max_seconds: float) -> dict[str, np.ndarray]:
    if not trace:
        raise SystemExit("Trace is empty; cannot plot.")
    steps = np.array([item["step"] for item in trace], dtype=np.float32)
    control_dt = 0.02
    time_s = steps * control_dt
    if max_seconds > 0:
        keep = time_s <= max_seconds
    else:
        keep = np.ones_like(time_s, dtype=bool)
    time_s = time_s[keep]

    cmd_vx = np.array([item["command"][0] for item in trace], dtype=np.float32)[keep]
    cmd_yaw = np.array([item["command"][2] for item in trace], dtype=np.float32)[keep]
    vx = np.array([item["base_lin_vel_local"][0] for item in trace], dtype=np.float32)[keep]
    vy = np.array([item["base_lin_vel_local"][1] for item in trace], dtype=np.float32)[keep]
    yaw = np.array([item["base_ang_vel_local"][2] for item in trace], dtype=np.float32)[keep]
    return {
        "time_s": time_s,
        "cmd_vx": cmd_vx,
        "cmd_yaw": cmd_yaw,
        "vx": vx,
        "vy": vy,
        "yaw": yaw,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (13, 7.5),
            "axes.facecolor": "#fbfaf7",
            "figure.facecolor": "#fbfaf7",
            "axes.edgecolor": "#d5d0c5",
            "axes.labelcolor": "#2b2a28",
            "xtick.color": "#4c4a45",
            "ytick.color": "#4c4a45",
            "text.color": "#1e1d1a",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.color": "#e7e1d6",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
        }
    )


def main() -> int:
    args = parse_args()
    _style()

    isaac = _series(_load_trace(Path(args.isaac_json)), args.max_seconds)
    mujoco = _series(_load_trace(Path(args.mujoco_json)), args.max_seconds)

    if args.forward_only:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(2, 2, sharex="col")
    fig.suptitle(args.title, y=0.98, fontweight="bold")

    colors = {
        "cmd": "#c46b2d",
        "vx": "#1f6aa5",
        "yaw": "#1f6aa5",
        "vy": "#2e8b57",
    }

    def plot_column(col_axes, data: dict[str, np.ndarray], title: str) -> None:
        ax_top, ax_bottom = col_axes
        ax_top.set_title(title, loc="left", fontweight="bold")
        ax_top.plot(data["time_s"], data["cmd_vx"], color=colors["cmd"], linestyle="--", linewidth=2.2, label="Command vx")
        ax_top.plot(data["time_s"], data["vx"], color=colors["vx"], linewidth=2.4, label="Actual vx")
        ax_top.plot(data["time_s"], data["vy"], color=colors["vy"], linewidth=1.4, alpha=0.85, label="Actual vy")
        ax_top.set_ylabel("Linear Velocity (m/s)")
        ax_top.legend(frameon=False, loc="upper right")

        ax_bottom.plot(data["time_s"], data["cmd_yaw"], color=colors["cmd"], linestyle="--", linewidth=2.2, label="Command yaw")
        ax_bottom.plot(data["time_s"], data["yaw"], color=colors["yaw"], linewidth=2.4, label="Actual yaw")
        ax_bottom.set_ylabel("Yaw Rate (rad/s)")
        ax_bottom.set_xlabel("Time (s)")
        ax_bottom.legend(frameon=False, loc="upper right")

    if args.forward_only:
        def plot_forward(ax, data: dict[str, np.ndarray], title: str) -> None:
            ax.set_title(title, loc="left", fontweight="bold")
            ax.plot(data["time_s"], data["cmd_vx"], color=colors["cmd"], linestyle="--", linewidth=2.4, label="Command vx")
            ax.plot(data["time_s"], data["vx"], color=colors["vx"], linewidth=2.7, label="Actual vx")
            ax.set_ylabel("Forward Velocity (m/s)")
            ax.set_xlabel("Time (s)")
            ax.legend(frameon=False, loc="upper right")

        plot_forward(axes[0], isaac, args.left_label)
        plot_forward(axes[1], mujoco, args.right_label)
        axes_iter = axes.ravel()
    else:
        plot_column((axes[0, 0], axes[1, 0]), isaac, args.left_label)
        plot_column((axes[0, 1], axes[1, 1]), mujoco, args.right_label)
        axes_iter = axes.ravel()

    for ax in axes_iter:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0)

    fig.text(
        0.5,
        0.015,
        args.caption,
        ha="center",
        fontsize=10,
        color="#5c5850",
    )

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_png, dpi=220)
    if args.out_svg:
        out_svg = Path(args.out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    print(f"[INFO] Wrote {out_png}")
    if args.out_svg:
        print(f"[INFO] Wrote {args.out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
