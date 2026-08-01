#!/usr/bin/env python3
"""Read-only mapping diagnostic for the newer Go2.

This subscribes directly to raw Unitree DDS lowstate and prints:
- raw motor indices 0..11 with q/dq/tau_est/temp
- the current deploy bundle's assumed semantic joint mapping

Use this without entering Velocity. It is intended to help verify whether the
newer robot matches the same motor ordering / nominal pose contract as the old
robot.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK2PY_ROOT = REPO_ROOT / "reference_repos" / "unitree_sdk2_python"
if str(SDK2PY_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK2PY_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


DEPLOY_CFG = (
    REPO_ROOT
    / "reference_repos"
    / "unitree_rl_lab"
    / "deploy"
    / "robots"
    / "go2"
    / "config"
    / "policy"
    / "velocity"
    / "c1_blind_rough_omni_usable_v1_final_new_robot"
    / "params"
    / "deploy.yaml"
)

SEMANTIC_NAMES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iface",
        default=os.environ.get("GO2_DDS_IFACE", "enp0s31f6"),
        help="Network interface for ChannelFactoryInitialize().",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=[],
        help="Topic to try. Can be passed multiple times.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the first sample on each topic.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of summary frames to print after the first sample arrives.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=0.5,
        help="Seconds between printed frames.",
    )
    return parser.parse_args()


def load_mapping() -> tuple[list[int], list[float]]:
    text = DEPLOY_CFG.read_text(encoding="utf-8")
    joint_ids_map = _parse_top_level_scalar_list(text, "joint_ids_map", cast=int)
    default_joint_pos = _parse_top_level_scalar_list(text, "default_joint_pos", cast=float)
    if len(joint_ids_map) != 12 or len(default_joint_pos) != 12:
        raise RuntimeError("Expected 12-entry joint_ids_map and default_joint_pos.")
    return joint_ids_map, default_joint_pos


def _parse_top_level_scalar_list(text: str, key: str, cast):
    pattern = re.compile(rf"^{re.escape(key)}:\s*\n((?:^[ \t]+-\s*.+\n)+)", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Could not find top-level list for '{key}' in {DEPLOY_CFG}")
    block = match.group(1)
    values = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        values.append(cast(line[1:].strip()))
    return values


def fmt_float(val: float) -> str:
    return f"{val:+.3f}"


def joint_line(label: str, q: float, dq: float, tau_est: float, temp: float, q_rel: float | None = None) -> str:
    rel_suffix = "" if q_rel is None else f" q_rel={fmt_float(q_rel)}"
    return (
        f"{label:<10} q={fmt_float(q)} dq={fmt_float(dq)} "
        f"tau={fmt_float(tau_est)} temp={temp:>4.1f}{rel_suffix}"
    )


def print_frame(msg: LowState_, joint_ids_map: list[int], default_joint_pos: list[float]) -> None:
    print(
        f"\n[FRAME] tick={msg.tick} power_v={float(msg.power_v):.2f} "
        f"foot_force={[int(x) for x in msg.foot_force]} "
        f"gyro_z={fmt_float(float(msg.imu_state.gyroscope[2]))}",
        flush=True,
    )
    print("[RAW 0..11]", flush=True)
    for idx in range(12):
        motor = msg.motor_state[idx]
        print(
            "  " + joint_line(
                f"motor[{idx}]",
                float(motor.q),
                float(motor.dq),
                float(motor.tau_est),
                float(motor.temperature),
            ),
            flush=True,
        )

    print("[ASSUMED SEMANTIC MAP]", flush=True)
    for semantic_idx, name in enumerate(SEMANTIC_NAMES):
        motor_idx = joint_ids_map[semantic_idx]
        motor = msg.motor_state[motor_idx]
        q = float(motor.q)
        print(
            "  "
            + joint_line(
                f"{name}->{motor_idx}",
                q,
                float(motor.dq),
                float(motor.tau_est),
                float(motor.temperature),
                q - default_joint_pos[semantic_idx],
            ),
            flush=True,
        )


def wait_for_topic(topic: str, timeout_s: float) -> tuple[bool, dict[str, LowState_ | None]]:
    latest: dict[str, LowState_ | None] = {"msg": None}

    def cb(msg: LowState_) -> None:
        latest["msg"] = msg

    print(f"[INFO] Subscribing to {topic} ...", flush=True)
    sub = ChannelSubscriber(topic, LowState_)
    sub.Init(cb, 10)

    t_end = time.time() + timeout_s
    while time.time() < t_end:
        if latest["msg"] is not None:
            print(f"[OK] First sample received on {topic}", flush=True)
            return True, latest
        time.sleep(0.05)

    print(f"[TIMEOUT] No sample on {topic} within {timeout_s:.1f}s", flush=True)
    return False, latest


def main() -> int:
    args = parse_args()
    joint_ids_map, default_joint_pos = load_mapping()
    topics = args.topics or ["rt/lowstate", "rt/lf/lowstate"]

    print(f"[INFO] Initializing sdk2py DDS on iface={args.iface}", flush=True)
    print(f"[INFO] Using deploy mapping from {DEPLOY_CFG}", flush=True)
    print(f"[INFO] joint_ids_map={joint_ids_map}", flush=True)
    print(f"[INFO] default_joint_pos={default_joint_pos}", flush=True)
    print(
        "[INFO] Suggested procedure: keep controller out of Velocity, stay in Passive or FixStand, "
        "and observe which raw motor indices move when a visible leg/joint moves.",
        flush=True,
    )
    ChannelFactoryInitialize(0, args.iface)

    latest: dict[str, LowState_ | None] | None = None
    for topic in topics:
        ok, latest_candidate = wait_for_topic(topic, args.timeout)
        if ok:
            latest = latest_candidate
            break
    if latest is None:
        return 1

    next_print = 0.0
    printed = 0
    last_tick: int | None = None
    while printed < args.samples:
        msg = latest["msg"]
        if msg is None:
            time.sleep(0.05)
            continue
        now = time.time()
        tick = int(msg.tick)
        if tick == last_tick:
            time.sleep(0.02)
            continue
        if now < next_print:
            time.sleep(0.02)
            continue
        print_frame(msg, joint_ids_map, default_joint_pos)
        last_tick = tick
        printed += 1
        next_print = now + max(args.period, 0.05)

    print("\n[INFO] Mapping diagnostic complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
