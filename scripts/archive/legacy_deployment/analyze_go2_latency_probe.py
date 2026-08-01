#!/usr/bin/env python3
"""Estimate deploy-side latency from high-rate Go2 lowstate/lowcmd probe captures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
]


@dataclass
class LowStateStream:
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    gyro: np.ndarray
    remote_cmd: np.ndarray


@dataclass
class LowCmdStream:
    t: np.ndarray
    q_des: np.ndarray


def _median_dt(t: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    return float(np.median(np.diff(t)))


def _load_lowstate_stream(path: Path) -> LowStateStream:
    t = []
    q = []
    dq = []
    gyro = []
    remote_cmd = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            snap = payload.get("lowstate", {}).get("snapshot")
            if snap is None:
                continue
            t.append(float(payload["monotonic_ns"]) * 1e-9)
            q.append([float(x) for x in snap["joint_q_12"]])
            dq.append([float(x) for x in snap["joint_dq_12"]])
            gyro.append([float(x) for x in snap["imu_gyro_xyz"]])
            remote = snap["remote"]
            remote_cmd.append([float(remote["ly"]), -float(remote["lx"]), -float(remote["rx"])])
    return LowStateStream(
        t=np.asarray(t, dtype=np.float64),
        q=np.asarray(q, dtype=np.float32),
        dq=np.asarray(dq, dtype=np.float32),
        gyro=np.asarray(gyro, dtype=np.float32),
        remote_cmd=np.asarray(remote_cmd, dtype=np.float32),
    )


def _load_lowcmd_stream(path: Path) -> LowCmdStream:
    t = []
    q_des = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            snap = payload.get("lowcmd", {}).get("snapshot")
            if snap is None:
                continue
            t.append(float(payload["monotonic_ns"]) * 1e-9)
            q_des.append([float(x) for x in snap["joint_q_des_12"]])
    return LowCmdStream(
        t=np.asarray(t, dtype=np.float64),
        q_des=np.asarray(q_des, dtype=np.float32),
    )


def _rising_edges(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.empty((0,), dtype=np.int64)
    prev = np.concatenate([[False], mask[:-1]])
    return np.flatnonzero(mask & ~prev)


def _merge_nearby_events(times: np.ndarray, min_gap_s: float) -> np.ndarray:
    if times.size == 0:
        return times
    keep = [times[0]]
    for item in times[1:]:
        if item - keep[-1] >= min_gap_s:
            keep.append(item)
    return np.asarray(keep, dtype=np.float64)


def _onset_times_from_signal(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    active_threshold: float,
    min_gap_s: float,
) -> np.ndarray:
    if t.size == 0:
        return np.empty((0,), dtype=np.float64)
    mask = signal >= active_threshold
    edges = _rising_edges(mask)
    return _merge_nearby_events(t[edges], min_gap_s=min_gap_s)


def _find_first_onset_after(
    t: np.ndarray,
    signal: np.ndarray,
    event_time: float,
    *,
    baseline_window_s: float,
    search_window_s: float,
    delta_threshold: float,
) -> float | None:
    base_mask = (t >= event_time - baseline_window_s) & (t < event_time)
    if not np.any(base_mask):
        return None
    baseline = float(np.median(signal[base_mask]))
    search_mask = (t >= event_time) & (t <= event_time + search_window_s)
    if not np.any(search_mask):
        return None
    search_t = t[search_mask]
    search_signal = signal[search_mask]
    hit = np.flatnonzero(np.abs(search_signal - baseline) >= delta_threshold)
    if hit.size == 0:
        return None
    return float(search_t[int(hit[0])])


def _estimate_event_delays(
    trigger_t: np.ndarray,
    trigger_signal: np.ndarray,
    response_t: np.ndarray,
    response_signal: np.ndarray,
    *,
    trigger_active_threshold: float,
    response_delta_threshold: float,
    min_gap_s: float,
    baseline_window_s: float,
    search_window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_times = _onset_times_from_signal(
        trigger_t,
        trigger_signal,
        active_threshold=trigger_active_threshold,
        min_gap_s=min_gap_s,
    )
    delays_ms = []
    matched_events = []
    for event_time in event_times:
        response_time = _find_first_onset_after(
            response_t,
            response_signal,
            event_time,
            baseline_window_s=baseline_window_s,
            search_window_s=search_window_s,
            delta_threshold=response_delta_threshold,
        )
        if response_time is None:
            continue
        matched_events.append(event_time)
        delays_ms.append((response_time - event_time) * 1e3)
    return np.asarray(matched_events, dtype=np.float64), np.asarray(delays_ms, dtype=np.float64)


def _estimate_joint_qdes_to_dq(
    lowcmd: LowCmdStream,
    lowstate: LowStateStream,
    *,
    joint_index: int,
    qdes_delta_threshold: float,
    dq_delta_threshold: float,
    min_gap_s: float,
    baseline_window_s: float,
    search_window_s: float,
) -> np.ndarray:
    _, delays = _estimate_event_delays(
        lowcmd.t,
        np.abs(lowcmd.q_des[:, joint_index]),
        lowstate.t,
        np.abs(lowstate.dq[:, joint_index]),
        trigger_active_threshold=qdes_delta_threshold,
        response_delta_threshold=dq_delta_threshold,
        min_gap_s=min_gap_s,
        baseline_window_s=baseline_window_s,
        search_window_s=search_window_s,
    )
    return delays


def _fmt_stats(values_ms: np.ndarray) -> str:
    if values_ms.size == 0:
        return "no matched events"
    return (
        f"count={values_ms.size} mean={np.mean(values_ms):.1f}ms "
        f"p50={np.median(values_ms):.1f}ms p95={np.percentile(values_ms, 95):.1f}ms "
        f"min={np.min(values_ms):.1f}ms max={np.max(values_ms):.1f}ms"
    )


def _joint_activity_summary(q_des: np.ndarray) -> list[tuple[int, float, float, float]]:
    if q_des.size == 0:
        return []
    q_range = np.max(q_des, axis=0) - np.min(q_des, axis=0)
    q_std = np.std(q_des, axis=0)
    q_peak = np.max(np.abs(q_des), axis=0)
    rows = []
    for joint_index in range(q_des.shape[1]):
        rows.append((joint_index, float(q_range[joint_index]), float(q_std[joint_index]), float(q_peak[joint_index])))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def _resample_series(
    t_src: np.ndarray,
    x_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    if t_src.size < 2 or t_dst.size == 0:
        return np.zeros_like(t_dst, dtype=np.float64)
    return np.interp(t_dst, t_src, x_src)


def _crosscorr_lag_ms(
    trigger_t: np.ndarray,
    trigger_signal: np.ndarray,
    response_t: np.ndarray,
    response_signal: np.ndarray,
    *,
    max_lag_s: float,
) -> tuple[float | None, float | None]:
    if trigger_t.size < 4 or response_t.size < 4:
        return None, None

    overlap_start = max(float(trigger_t[0]), float(response_t[0]))
    overlap_end = min(float(trigger_t[-1]), float(response_t[-1]))
    if overlap_end <= overlap_start:
        return None, None

    dt = max(_median_dt(trigger_t), _median_dt(response_t))
    if dt <= 0.0:
        return None, None

    t_uniform = np.arange(overlap_start, overlap_end, dt, dtype=np.float64)
    if t_uniform.size < 8:
        return None, None

    trig = _resample_series(trigger_t, trigger_signal, t_uniform)
    resp = _resample_series(response_t, response_signal, t_uniform)

    trig = trig - np.mean(trig)
    resp = resp - np.mean(resp)
    trig_std = np.std(trig)
    resp_std = np.std(resp)
    if trig_std <= 1e-8 or resp_std <= 1e-8:
        return None, None

    max_lag_samples = max(1, int(round(max_lag_s / dt)))
    best_lag = None
    best_corr = None
    for lag in range(max_lag_samples + 1):
        if lag == 0:
            left = trig
            right = resp
        else:
            left = trig[:-lag]
            right = resp[lag:]
        if left.size < 8:
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if not np.isfinite(corr):
            continue
        if best_corr is None or abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    if best_lag is None or best_corr is None:
        return None, None
    return best_lag * dt * 1e3, best_corr


def _signal_derivative_norm(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    if t.size < 2:
        return np.zeros((0,), dtype=np.float64)
    dx = np.diff(x, axis=0)
    dt = np.diff(t)
    dt = np.clip(dt, 1e-6, None)
    if dx.ndim == 1:
        deriv = dx / dt
        return np.concatenate([[0.0], np.abs(deriv)])
    deriv = dx / dt[:, None]
    return np.concatenate([np.zeros((1, deriv.shape[1]), dtype=np.float64), np.abs(deriv)], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowstate-jsonl", required=True, help="Full-rate lowstate stream JSONL.")
    parser.add_argument("--lowcmd-jsonl", required=True, help="Full-rate lowcmd stream JSONL.")
    parser.add_argument("--command-active-threshold", type=float, default=0.15)
    parser.add_argument("--qdes-active-threshold", type=float, default=0.12)
    parser.add_argument("--qdes-delta-threshold", type=float, default=0.05)
    parser.add_argument("--dq-delta-threshold", type=float, default=0.25)
    parser.add_argument("--gyro-delta-threshold", type=float, default=0.12)
    parser.add_argument("--min-gap-s", type=float, default=0.5)
    parser.add_argument("--baseline-window-s", type=float, default=0.1)
    parser.add_argument("--search-window-s", type=float, default=0.5)
    args = parser.parse_args()

    lowstate_path = Path(args.lowstate_jsonl)
    lowcmd_path = Path(args.lowcmd_jsonl)
    lowstate = _load_lowstate_stream(lowstate_path)
    lowcmd = _load_lowcmd_stream(lowcmd_path)

    if lowstate.t.size == 0:
        raise SystemExit(f"No lowstate samples found in {lowstate_path}")
    if lowcmd.t.size == 0:
        raise SystemExit(f"No lowcmd samples found in {lowcmd_path}")

    remote_norm = np.linalg.norm(lowstate.remote_cmd, axis=1)
    qdes_norm = np.linalg.norm(lowcmd.q_des, axis=1)
    dq_norm = np.linalg.norm(lowstate.dq, axis=1)
    gyro_norm = np.linalg.norm(lowstate.gyro, axis=1)
    qdes_rate_norm = _signal_derivative_norm(lowcmd.t, lowcmd.q_des)
    if qdes_rate_norm.ndim > 1:
        qdes_rate_norm = np.linalg.norm(qdes_rate_norm, axis=1)

    _, remote_to_qdes_ms = _estimate_event_delays(
        lowstate.t,
        remote_norm,
        lowcmd.t,
        qdes_norm,
        trigger_active_threshold=args.command_active_threshold,
        response_delta_threshold=args.qdes_delta_threshold,
        min_gap_s=args.min_gap_s,
        baseline_window_s=args.baseline_window_s,
        search_window_s=args.search_window_s,
    )
    matched_qdes_times, qdes_to_dq_ms = _estimate_event_delays(
        lowcmd.t,
        qdes_norm,
        lowstate.t,
        dq_norm,
        trigger_active_threshold=args.qdes_active_threshold,
        response_delta_threshold=args.dq_delta_threshold,
        min_gap_s=args.min_gap_s,
        baseline_window_s=args.baseline_window_s,
        search_window_s=args.search_window_s,
    )
    _, qdes_to_gyro_ms = _estimate_event_delays(
        lowcmd.t,
        qdes_norm,
        lowstate.t,
        gyro_norm,
        trigger_active_threshold=args.qdes_active_threshold,
        response_delta_threshold=args.gyro_delta_threshold,
        min_gap_s=args.min_gap_s,
        baseline_window_s=args.baseline_window_s,
        search_window_s=args.search_window_s,
    )
    cross_remote_to_qdes_ms, cross_remote_to_qdes_corr = _crosscorr_lag_ms(
        lowstate.t,
        remote_norm,
        lowcmd.t,
        qdes_rate_norm,
        max_lag_s=args.search_window_s,
    )
    cross_qdes_to_dq_ms, cross_qdes_to_dq_corr = _crosscorr_lag_ms(
        lowcmd.t,
        qdes_rate_norm,
        lowstate.t,
        dq_norm,
        max_lag_s=args.search_window_s,
    )
    cross_qdes_to_imu_ms, cross_qdes_to_imu_corr = _crosscorr_lag_ms(
        lowcmd.t,
        qdes_rate_norm,
        lowstate.t,
        gyro_norm,
        max_lag_s=args.search_window_s,
    )
    joint_activity = _joint_activity_summary(lowcmd.q_des)

    print(f"lowstate_jsonl: {lowstate_path}")
    print(f"lowcmd_jsonl: {lowcmd_path}")
    print(f"lowstate_samples: {lowstate.t.size}")
    print(f"lowcmd_samples: {lowcmd.t.size}")
    if lowstate.t.size >= 2:
        print(f"lowstate_hz_estimate: {(lowstate.t.size - 1) / (lowstate.t[-1] - lowstate.t[0]):.1f}")
    if lowcmd.t.size >= 2:
        print(f"lowcmd_hz_estimate: {(lowcmd.t.size - 1) / (lowcmd.t[-1] - lowcmd.t[0]):.1f}")

    print("\nDelay estimates:")
    print(f"  remote_to_qdes: {_fmt_stats(remote_to_qdes_ms)}")
    print(f"  qdes_to_joint_motion: {_fmt_stats(qdes_to_dq_ms)}")
    print(f"  qdes_to_imu: {_fmt_stats(qdes_to_gyro_ms)}")
    print("\nCross-correlation estimates:")
    if cross_remote_to_qdes_ms is None:
        print("  remote_to_qdes: unavailable")
    else:
        print(f"  remote_to_qdes: lag={cross_remote_to_qdes_ms:.1f}ms corr={cross_remote_to_qdes_corr:+.3f}")
    if cross_qdes_to_dq_ms is None:
        print("  qdes_to_joint_motion: unavailable")
    else:
        print(f"  qdes_to_joint_motion: lag={cross_qdes_to_dq_ms:.1f}ms corr={cross_qdes_to_dq_corr:+.3f}")
    if cross_qdes_to_imu_ms is None:
        print("  qdes_to_imu: unavailable")
    else:
        print(f"  qdes_to_imu: lag={cross_qdes_to_imu_ms:.1f}ms corr={cross_qdes_to_imu_corr:+.3f}")

    print("\nTop joints by q_des activity:")
    for joint_index, q_range, q_std, q_peak in joint_activity[:6]:
        print(
            f"  {JOINT_NAMES[joint_index]}: "
            f"range={q_range:.3f} std={q_std:.3f} peak_abs={q_peak:.3f}"
        )

    print("\nPer-joint qdes_to_dq:")
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        delays_ms = _estimate_joint_qdes_to_dq(
            lowcmd,
            lowstate,
            joint_index=joint_index,
            qdes_delta_threshold=args.qdes_delta_threshold,
            dq_delta_threshold=args.dq_delta_threshold,
            min_gap_s=args.min_gap_s,
            baseline_window_s=args.baseline_window_s,
            search_window_s=args.search_window_s,
        )
        qdes_rate_joint = _signal_derivative_norm(lowcmd.t, lowcmd.q_des[:, joint_index])
        dq_joint = np.abs(lowstate.dq[:, joint_index])
        cross_ms, cross_corr = _crosscorr_lag_ms(
            lowcmd.t,
            qdes_rate_joint,
            lowstate.t,
            dq_joint,
            max_lag_s=args.search_window_s,
        )
        extra = ""
        if cross_ms is not None and cross_corr is not None:
            extra = f" | xcorr lag={cross_ms:.1f}ms corr={cross_corr:+.3f}"
        print(f"  {joint_name}: {_fmt_stats(delays_ms)}{extra}")

    if matched_qdes_times.size == 0:
        print("\nNote: no q_des events crossed the configured threshold. Use deliberate step/pulse tests.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
