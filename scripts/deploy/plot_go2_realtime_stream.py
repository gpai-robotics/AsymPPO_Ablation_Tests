#!/usr/bin/env python3
"""Plot a live Go2 telemetry JSONL stream written by monitor_go2_realtime.py."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from go2_monitor_schema import normalize_payload_joint_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--history-sec", type=float, default=20.0)
    return parser.parse_args()


class TailBuffer:
    def __init__(self, path: Path, history_sec: float) -> None:
        self.path = path
        self.history_sec = history_sec
        self.maxlen = 2000
        self.t = deque(maxlen=self.maxlen)
        self.q_err = deque(maxlen=self.maxlen)
        self.tau_est = deque(maxlen=self.maxlen)
        self.joint_vel = deque(maxlen=self.maxlen)
        self.temperature = deque(maxlen=self.maxlen)
        self.foot_force = deque(maxlen=self.maxlen)
        self.imu_gyro = deque(maxlen=self.maxlen)
        self.sport_vel = deque(maxlen=self.maxlen)
        self.sport_yaw = deque(maxlen=self.maxlen)
        self.offset = 0

    def poll(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload, _ = normalize_payload_joint_order(json.loads(line))
                t = float(payload["wall_time"])
                latest = payload["latest"]
                self.t.append(t)
                self.q_err.append(latest["q_err"])
                self.tau_est.append(latest["tau_est"])
                self.joint_vel.append(latest["joint_vel"])
                self.temperature.append(latest["temperature"])
                self.foot_force.append(latest["foot_force"])
                self.imu_gyro.append(latest["imu_gyro"])
                self.sport_vel.append(latest["sport_vel"])
                self.sport_yaw.append(latest["sport_yaw"])
            self.offset = handle.tell()

    def arrays(self, series: deque) -> tuple[np.ndarray, np.ndarray]:
        if not self.t or not series:
            return np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
        t = np.asarray(self.t, dtype=np.float32)
        values = np.asarray(list(series), dtype=np.float32)
        mask = t >= (t[-1] - self.history_sec)
        return t[mask] - t[-1], values[mask]


def main() -> int:
    args = parse_args()
    path = Path(args.jsonl)
    buf = TailBuffer(path, args.history_sec)

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
    fig.suptitle(f"Go2 Live Stream Plot: {path.name}")
    fig.text(
        0.5,
        0.985,
        "Joint traces ordered: FL_hip FR_hip RL_hip RR_hip FL_thigh FR_thigh RL_thigh RR_thigh FL_calf FR_calf RL_calf RR_calf",
        ha="center",
        va="top",
        fontsize=8,
    )

    line_sets = {}
    colors12 = plt.cm.tab20(np.linspace(0, 1, 12))
    for key, ax, title in (
        ("q_err", axes[0, 0], "Joint Position Error"),
        ("tau_est", axes[0, 1], "Estimated Joint Torque"),
        ("joint_vel", axes[1, 0], "Joint Velocity"),
        ("temperature", axes[1, 1], "Motor Temperature"),
    ):
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        lines = [ax.plot([], [], lw=1.0, color=colors12[i])[0] for i in range(12)]
        line_sets[key] = lines

    axes[2, 0].set_title("Base Motion")
    axes[2, 1].set_title("Foot Force")
    axes[2, 0].grid(True, alpha=0.25)
    axes[2, 1].grid(True, alpha=0.25)
    motion_labels = ["vx", "vy", "vz", "imu_gx", "imu_gy", "imu_gz", "yaw_speed"]
    motion_lines = [axes[2, 0].plot([], [], lw=1.2, label=motion_labels[i])[0] for i in range(7)]
    axes[2, 0].legend(loc="upper left", ncol=4, fontsize=8, framealpha=0.8)
    foot_lines = [axes[2, 1].plot([], [], lw=1.2, label=f"foot_{i}")[0] for i in range(4)]
    axes[2, 1].legend(loc="upper left", ncol=4, fontsize=8, framealpha=0.8)
    line_sets["motion"] = motion_lines
    line_sets["foot_force"] = foot_lines

    for ax in axes.ravel():
        ax.set_xlim(-args.history_sec, 0.0)

    def set_axis_ylim(ax, values):
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return
        lower = float(np.percentile(finite_values, 2))
        upper = float(np.percentile(finite_values, 98))
        vmin = min(lower, float(np.min(finite_values)))
        vmax = max(upper, float(np.max(finite_values)))
        if abs(vmax - vmin) < 1e-6:
            pad = max(0.1, abs(vmax) * 0.1 + 0.05)
        else:
            pad = max(0.05, 0.12 * (vmax - vmin))
        ax.set_ylim(vmin - pad, vmax + pad)

    def set_lines(ax, lines, t_rel, values):
        if t_rel.size == 0 or values.size == 0:
            return
        for i, line in enumerate(lines):
            if i < values.shape[1]:
                line.set_data(t_rel, values[:, i])
        set_axis_ylim(ax, values.reshape(-1))

    def update(_):
        buf.poll()
        for key in ("q_err", "tau_est", "joint_vel", "temperature"):
            t_rel, values = buf.arrays(getattr(buf, key))
            set_lines(axes[0, 0] if key == "q_err" else axes[0, 1] if key == "tau_est" else axes[1, 0] if key == "joint_vel" else axes[1, 1], line_sets[key], t_rel, values)

        t_vel, sport_vel = buf.arrays(buf.sport_vel)
        t_gyro, gyro = buf.arrays(buf.imu_gyro)
        t_yaw, yaw = buf.arrays(buf.sport_yaw)
        if t_vel.size and sport_vel.size:
            for i in range(min(3, sport_vel.shape[1])):
                motion_lines[i].set_data(t_vel, sport_vel[:, i])
        if t_gyro.size and gyro.size:
            for i in range(min(3, gyro.shape[1])):
                motion_lines[3 + i].set_data(t_gyro, gyro[:, i])
        if t_yaw.size and yaw.size:
            motion_lines[6].set_data(t_yaw, yaw[:, 0] if yaw.ndim > 1 else yaw)
        motion_chunks = []
        for value in (sport_vel, gyro, yaw):
            if value.size:
                finite = value[np.isfinite(value)]
                if finite.size:
                    motion_chunks.append(finite)
        if motion_chunks:
            set_axis_ylim(axes[2, 0], np.concatenate(motion_chunks))

        t_foot, foot = buf.arrays(buf.foot_force)
        set_lines(axes[2, 1], foot_lines, t_foot, foot)

        artists = []
        for lines in line_sets.values():
            artists.extend(lines)
        return artists

    while not path.exists():
        print(f"Waiting for stream file: {path}")
        time.sleep(0.5)

    stop_requested = False

    def on_close(_event):
        nonlocal stop_requested
        stop_requested = True

    fig.canvas.mpl_connect("close_event", on_close)
    anim = FuncAnimation(fig, update, interval=250, blit=False, cache_frame_data=False)
    fig._go2_anim = anim
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.show(block=False)
    try:
        while not stop_requested and plt.fignum_exists(fig.number):
            plt.pause(0.2)
    except KeyboardInterrupt:
        stop_requested = True
    finally:
        try:
            anim.event_source.stop()
        except Exception:
            pass
        if plt.fignum_exists(fig.number):
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
