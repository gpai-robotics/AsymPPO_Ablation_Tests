#!/usr/bin/env python3
"""Bridge ROS2 /odometry/filtered twist.linear to localhost UDP."""

from __future__ import annotations

import argparse
import socket
import struct

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--odom-topic",
        default="/odometry/filtered",
        help="ROS2 odometry topic to subscribe to.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="UDP host for the local bridge.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5560,
        help="UDP port for the local bridge.",
    )
    return parser.parse_args()


class OdomBridge(Node):
    def __init__(self, odom_topic: str, host: str, port: int) -> None:
        super().__init__("odometry_to_udp_bridge")
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        self.create_subscription(Odometry, odom_topic, self._callback, qos)
        self.get_logger().info(f"Bridging {odom_topic} -> udp://{host}:{port}")

    def _callback(self, msg: Odometry) -> None:
        linear = msg.twist.twist.linear
        payload = struct.pack("fff", float(linear.x), float(linear.y), float(linear.z))
        self._sock.sendto(payload, self._addr)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = OdomBridge(args.odom_topic, args.host, args.port)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
