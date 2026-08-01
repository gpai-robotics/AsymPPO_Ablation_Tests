#!/usr/bin/env python3
"""Read-only verifier for raw Unitree DDS lowstate via sdk2py.

This bypasses ROS2 topic bridging and subscribes the same way Unitree's
own sdk2py examples do. It is intended as a quick truth test for whether
`rt/lowstate` or `rt/lf/lowstate` samples are actually reachable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK2PY_ROOT = REPO_ROOT / "reference_repos" / "unitree_sdk2_python"
if str(SDK2PY_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK2PY_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


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
        help="Seconds to wait per topic before declaring timeout.",
    )
    return parser.parse_args()


def summarize(msg: LowState_) -> str:
    fr_hip = msg.motor_state[0]
    mode_like = getattr(msg, "mode_pr", None)
    if mode_like is None:
        mode_like = getattr(msg, "level_flag", None)
    return (
        f"tick={msg.tick} mode={mode_like} "
        f"imu_gyro_z={float(msg.imu_state.gyroscope[2]):+.3f} "
        f"power_v={float(msg.power_v):.2f} "
        f"foot_force={[float(x) for x in msg.foot_force]} "
        f"fr0_q={float(fr_hip.q):+.3f}"
    )


def wait_once(topic: str, timeout_s: float) -> bool:
    latest: dict[str, LowState_ | None] = {"msg": None}

    def cb(msg: LowState_) -> None:
        latest["msg"] = msg

    print(f"[INFO] Subscribing to {topic} ...", flush=True)
    sub = ChannelSubscriber(topic, LowState_)
    sub.Init(cb, 10)

    t_end = time.time() + timeout_s
    while time.time() < t_end:
        if latest["msg"] is not None:
            print(f"[OK] First sample on {topic}: {summarize(latest['msg'])}", flush=True)
            return True
        time.sleep(0.05)

    print(f"[TIMEOUT] No sample received on {topic} within {timeout_s:.1f}s", flush=True)
    return False


def main() -> int:
    args = parse_args()
    topics = args.topics or ["rt/lowstate", "rt/lf/lowstate"]

    print(f"[INFO] Initializing sdk2py DDS on iface={args.iface}", flush=True)
    ChannelFactoryInitialize(0, args.iface)

    success = False
    for topic in topics:
        if wait_once(topic, args.timeout):
            success = True
            break

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
