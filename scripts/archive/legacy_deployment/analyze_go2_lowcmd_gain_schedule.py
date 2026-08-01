#!/usr/bin/env python3
"""Summarize gain schedules from a full-rate Go2 rt/lowcmd JSONL capture."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


LEG_NAMES = ("FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf")


@dataclass
class GainSample:
    wall_time: float
    count: int
    kp: list[float]
    kd: list[float]
    q_des: list[float]


@dataclass
class GainSegment:
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    kp: tuple[float, ...]
    kd: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def count(self) -> int:
        return self.end_index - self.start_index + 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to *_lowcmd_stream.jsonl")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=20,
        help="Maximum number of gain segments to print.",
    )
    return parser.parse_args()


def _load_samples(path: Path) -> list[GainSample]:
    samples: list[GainSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            lowcmd = payload.get("lowcmd", {})
            snap = lowcmd.get("snapshot")
            if not snap:
                continue
            samples.append(
                GainSample(
                    wall_time=float(payload["wall_time"]),
                    count=int(lowcmd.get("count", 0)),
                    kp=[float(x) for x in snap["joint_kp_12"]],
                    kd=[float(x) for x in snap["joint_kd_12"]],
                    q_des=[float(x) for x in snap["joint_q_des_12"]],
                )
            )
    return samples


def _segment_samples(samples: list[GainSample]) -> list[GainSegment]:
    if not samples:
        return []
    segments: list[GainSegment] = []
    start = 0
    current_kp = tuple(samples[0].kp)
    current_kd = tuple(samples[0].kd)
    for index in range(1, len(samples)):
        kp = tuple(samples[index].kp)
        kd = tuple(samples[index].kd)
        if kp != current_kp or kd != current_kd:
            segments.append(
                GainSegment(
                    start_index=start,
                    end_index=index - 1,
                    start_time=samples[start].wall_time,
                    end_time=samples[index - 1].wall_time,
                    kp=current_kp,
                    kd=current_kd,
                )
            )
            start = index
            current_kp = kp
            current_kd = kd
    segments.append(
        GainSegment(
            start_index=start,
            end_index=len(samples) - 1,
            start_time=samples[start].wall_time,
            end_time=samples[-1].wall_time,
            kp=current_kp,
            kd=current_kd,
        )
    )
    return segments


def _format_gain_vector(values: tuple[float, ...]) -> str:
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return f"{unique_values[0]:.3f} (all joints)"
    items = [f"{LEG_NAMES[index]}={values[index]:.3f}" for index in range(len(values))]
    return ", ".join(items)


def _mean_rate_hz(samples: list[GainSample]) -> float | None:
    if len(samples) < 2:
        return None
    duration = samples[-1].wall_time - samples[0].wall_time
    if duration <= 0:
        return None
    return float(len(samples) - 1) / duration


def main() -> int:
    args = _parse_args()
    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    samples = _load_samples(path)
    if not samples:
        raise SystemExit(f"No lowcmd samples found in {path}")

    segments = _segment_samples(samples)
    hz = _mean_rate_hz(samples)

    print(f"jsonl: {path}")
    print(f"samples: {len(samples)}")
    print(f"stream_hz_estimate: {hz:.2f}" if hz is not None else "stream_hz_estimate: unknown")
    print(f"gain_segments: {len(segments)}")

    unique_regimes: dict[tuple[tuple[float, ...], tuple[float, ...]], float] = {}
    for segment in segments:
        key = (segment.kp, segment.kd)
        unique_regimes[key] = unique_regimes.get(key, 0.0) + segment.duration_s

    print()
    print("Unique gain regimes:")
    for idx, ((kp, kd), total_duration) in enumerate(
        sorted(unique_regimes.items(), key=lambda item: item[1], reverse=True),
        start=1,
    ):
        print(
            f"  {idx}. duration={total_duration:.3f}s | "
            f"kp={_format_gain_vector(kp)} | kd={_format_gain_vector(kd)}"
        )

    print()
    print("Gain timeline:")
    for idx, segment in enumerate(segments[: args.max_segments], start=1):
        print(
            f"  {idx}. samples={segment.start_index}-{segment.end_index} "
            f"count={segment.count} duration={segment.duration_s:.3f}s "
            f"kp={_format_gain_vector(segment.kp)} kd={_format_gain_vector(segment.kd)}"
        )
    if len(segments) > args.max_segments:
        print(f"  ... truncated {len(segments) - args.max_segments} additional segments")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
