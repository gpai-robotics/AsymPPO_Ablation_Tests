#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/scripts/deploy/source_local_odom_env.sh"

ros2 launch go2_odometry go2_odometry_switch.launch.py odom_type:=fake base_height:=0.30
