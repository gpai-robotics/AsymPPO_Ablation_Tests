#!/usr/bin/env python3
"""Inspect go2_ctrl release windows where commands are near zero but motion persists."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
TS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]")

VELOCITY_CMD_RE = re.compile(
    rf"VelocityCmd vx/vy/wz raw=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"target=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"filtered=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"lin_vel=\[{FLOAT_RE}, {FLOAT_RE}, {FLOAT_RE}\] "
    rf"imu_wz={FLOAT_RE} blend_alpha={FLOAT_RE}"
)

JOINT_GENERIC_RE = re.compile(
    rf"JointDiag (?P<label>raw_action|rel_cmd|rel_pos) "
    rf"FL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"FR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\]"
)

JOINT_TAU_RE = re.compile(
    rf"JointTauDiag tau_est "
    rf"FL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"FR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"RR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\]"
)

LAST_ACTION_RE = re.compile(
    rf"ObsDiag policy_obs last_action_FL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"last_action_FR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"last_action_RL=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\] "
    rf"last_action_RR=\[{FLOAT_RE},{FLOAT_RE},{FLOAT_RE}\]"
)

LEG_ORDER = ("FL", "FR", "RL", "RR")


@dataclass
class Sample:
    ts: str
    filtered: list[float]
    lin_vel: list[float]
    imu_wz: float
    raw_action: dict[str, list[float]] | None = None
    rel_cmd: dict[str, list[float]] | None = None
    rel_pos: dict[str, list[float]] | None = None
    tau_est: dict[str, list[float]] | None = None
    last_action: dict[str, list[float]] | None = None

    @property
    def lin_vel_xy_norm(self) -> float:
        return math.hypot(self.lin_vel[0], self.lin_vel[1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, help="Path to raw go2_ctrl log.")
    parser.add_argument("--cmd-threshold-xy", type=float, default=0.05)
    parser.add_argument("--cmd-threshold-wz", type=float, default=0.05)
    parser.add_argument("--min-imu-wz", type=float, default=0.30)
    parser.add_argument("--min-lin-vel-xy", type=float, default=0.15)
    parser.add_argument(
        "--pre-active-window",
        type=int,
        default=6,
        help="Require at least one non-neutral command sample within this many prior samples.",
    )
    parser.add_argument(
        "--show-context",
        type=int,
        default=2,
        help="How many samples before/after each episode to print.",
    )
    return parser.parse_args()


def _extract_floats(match: re.Match[str], start_group: int = 1) -> list[float]:
    return [float(group) for group in match.groups()[start_group - 1 :]]


def _extract_leg_vectors(values: list[float]) -> dict[str, list[float]]:
    return {
        LEG_ORDER[0]: values[0:3],
        LEG_ORDER[1]: values[3:6],
        LEG_ORDER[2]: values[6:9],
        LEG_ORDER[3]: values[9:12],
    }


def _leg_norms(leg_vectors: dict[str, list[float]] | None) -> dict[str, float]:
    if leg_vectors is None:
        return {}
    return {
        leg: math.sqrt(sum(component * component for component in leg_vectors[leg]))
        for leg in LEG_ORDER
    }


def _lr_asymmetry(leg_vectors: dict[str, list[float]] | None) -> float:
    if leg_vectors is None:
        return 0.0
    diffs = []
    for left, right in (("FL", "FR"), ("RL", "RR")):
        for axis in range(3):
            diffs.append(abs(leg_vectors[left][axis] - leg_vectors[right][axis]))
    return sum(diffs) / len(diffs)


def _format_vec(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:+.3f}" for value in values) + "]"


def _parse_samples(log_path: Path) -> list[Sample]:
    samples: list[Sample] = []
    for line in log_path.read_text().splitlines():
        ts_match = TS_RE.match(line)
        ts = ts_match.group("ts") if ts_match else ""

        velocity_match = VELOCITY_CMD_RE.search(line)
        if velocity_match:
            values = _extract_floats(velocity_match)
            samples.append(
                Sample(
                    ts=ts,
                    filtered=values[6:9],
                    lin_vel=values[9:12],
                    imu_wz=values[12],
                )
            )
            continue

        if not samples:
            continue

        generic_match = JOINT_GENERIC_RE.search(line)
        if generic_match:
            label = generic_match.group("label")
            values = [float(group) for group in generic_match.groups()[1:]]
            setattr(samples[-1], label, _extract_leg_vectors(values))
            continue

        tau_match = JOINT_TAU_RE.search(line)
        if tau_match:
            samples[-1].tau_est = _extract_leg_vectors(_extract_floats(tau_match))
            continue

        last_action_match = LAST_ACTION_RE.search(line)
        if last_action_match:
            samples[-1].last_action = _extract_leg_vectors(_extract_floats(last_action_match))
            continue

    return samples


def _is_neutral(sample: Sample, xy_threshold: float, wz_threshold: float) -> bool:
    return (
        abs(sample.filtered[0]) <= xy_threshold
        and abs(sample.filtered[1]) <= xy_threshold
        and abs(sample.filtered[2]) <= wz_threshold
    )


def _is_active(sample: Sample, xy_threshold: float, wz_threshold: float) -> bool:
    return not _is_neutral(sample, xy_threshold, wz_threshold)


def main() -> int:
    args = _parse_args()
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise SystemExit(f"Missing log file: {log_path}")

    samples = _parse_samples(log_path)
    if not samples:
        raise SystemExit(f"No velocity samples found in {log_path}")

    candidate_indices: list[int] = []
    for index, sample in enumerate(samples):
        if not _is_neutral(sample, args.cmd_threshold_xy, args.cmd_threshold_wz):
            continue
        if (
            abs(sample.imu_wz) < args.min_imu_wz
            and sample.lin_vel_xy_norm < args.min_lin_vel_xy
        ):
            continue
        start = max(0, index - args.pre_active_window)
        if not any(
            _is_active(prior, args.cmd_threshold_xy, args.cmd_threshold_wz)
            for prior in samples[start:index]
        ):
            continue
        candidate_indices.append(index)

    episodes: list[tuple[int, int]] = []
    for index in candidate_indices:
        if not episodes or index != episodes[-1][1] + 1:
            episodes.append((index, index))
        else:
            episodes[-1] = (episodes[-1][0], index)

    print(f"log_file: {log_path}")
    print(f"samples: {len(samples)}")
    print(f"release_episodes: {len(episodes)}")

    if not episodes:
        return 0

    for episode_number, (start_index, end_index) in enumerate(episodes, start=1):
        peak_imu_sample = max(samples[start_index : end_index + 1], key=lambda sample: abs(sample.imu_wz))
        peak_lin_sample = max(
            samples[start_index : end_index + 1],
            key=lambda sample: sample.lin_vel_xy_norm,
        )
        context_start = max(0, start_index - args.show_context)
        context_end = min(len(samples), end_index + args.show_context + 1)
        print()
        print(
            f"Episode {episode_number}: samples {start_index}-{end_index} | "
            f"peak |imu_wz|={abs(peak_imu_sample.imu_wz):.3f} at {peak_imu_sample.ts} | "
            f"peak |lin_vel_xy|={peak_lin_sample.lin_vel_xy_norm:.3f} at {peak_lin_sample.ts}"
        )
        for index in range(context_start, context_end):
            sample = samples[index]
            marker = ">>" if start_index <= index <= end_index else "  "
            raw_norms = _leg_norms(sample.raw_action)
            rel_norms = _leg_norms(sample.rel_cmd)
            tau_norms = _leg_norms(sample.tau_est)
            print(
                f"{marker} #{index:03d} {sample.ts} "
                f"filtered={_format_vec(sample.filtered)} "
                f"lin_vel={_format_vec(sample.lin_vel)} "
                f"imu_wz={sample.imu_wz:+.3f} "
                f"raw_lr_asym={_lr_asymmetry(sample.raw_action):.3f} "
                f"rel_lr_asym={_lr_asymmetry(sample.rel_cmd):.3f}"
            )
            if raw_norms:
                print(
                    "   raw_norms="
                    + ", ".join(f"{leg}:{raw_norms[leg]:.3f}" for leg in LEG_ORDER)
                )
            if rel_norms:
                print(
                    "   rel_norms="
                    + ", ".join(f"{leg}:{rel_norms[leg]:.3f}" for leg in LEG_ORDER)
                )
            if tau_norms:
                print(
                    "   tau_norms="
                    + ", ".join(f"{leg}:{tau_norms[leg]:.3f}" for leg in LEG_ORDER)
                )
            if sample.last_action:
                last_norms = _leg_norms(sample.last_action)
                print(
                    "   last_action_norms="
                    + ", ".join(f"{leg}:{last_norms[leg]:.3f}" for leg in LEG_ORDER)
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
