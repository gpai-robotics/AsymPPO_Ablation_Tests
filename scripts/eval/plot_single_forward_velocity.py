#!/usr/bin/env python3
"""Create a single presentation-ready forward-velocity tracking plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-svg", default="")
    parser.add_argument("--title", default="Velocity Tracking")
    parser.add_argument("--label", default="IsaacLab")
    parser.add_argument(
        "--caption",
        default="Commanded vs realized forward velocity",
    )
    parser.add_argument("--max-seconds", type=float, default=5.0)
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
    time_s = steps * 0.02
    keep = time_s <= max_seconds if max_seconds > 0 else np.ones_like(time_s, dtype=bool)
    return {
        "time_s": time_s[keep],
        "cmd_vx": np.array([item["command"][0] for item in trace], dtype=np.float32)[keep],
        "vx": np.array([item["base_lin_vel_local"][0] for item in trace], dtype=np.float32)[keep],
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.8),
            "axes.facecolor": "#fbfaf7",
            "figure.facecolor": "#fbfaf7",
            "axes.edgecolor": "#d5d0c5",
            "axes.labelcolor": "#2b2a28",
            "xtick.color": "#4c4a45",
            "ytick.color": "#4c4a45",
            "text.color": "#1e1d1a",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.color": "#e7e1d6",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
        }
    )


def main() -> int:
    args = parse_args()
    _style()
    data = _series(_load_trace(Path(args.trace_json)), args.max_seconds)

    fig, ax = plt.subplots()
    ax.set_title(args.title, loc="left", fontweight="bold")
    ax.plot(data["time_s"], data["cmd_vx"], color="#c46b2d", linestyle="--", linewidth=2.5, label="Command vx")
    ax.plot(data["time_s"], data["vx"], color="#1f6aa5", linewidth=2.8, label=f"{args.label} actual vx")
    ax.set_ylabel("Forward Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0)
    y_min = float(min(np.min(data["cmd_vx"]), np.min(data["vx"])))
    y_max = float(max(np.max(data["cmd_vx"]), np.max(data["vx"])))
    pad = max(0.05, 0.12 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(y_min - pad, y_max + pad)

    fig.text(0.5, 0.02, args.caption, ha="center", fontsize=10, color="#5c5850")
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
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
