#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS_DIR="${ROOT_DIR}/reference_repos/odom_ws"
GO2_HW_PREFIX="${CONDA_PREFIX:-/home/bhuvan/miniconda3/envs/go2-hw}"

"${ROOT_DIR}/scripts/deploy/setup_local_odom_ws.sh"
source /opt/ros/humble/setup.bash

export CMAKE_PREFIX_PATH="${GO2_HW_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${GO2_HW_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"

cd "${WS_DIR}"

# Build only the packages needed by the Go2 odometry stack.
colcon build \
  --symlink-install \
  --packages-select unitree_go unitree_description inekf go2_odometry \
  --cmake-args \
    -DBUILD_TESTING=OFF \
    -DPython3_EXECUTABLE="${GO2_HW_PREFIX}/bin/python3" \
    -DPython_EXECUTABLE="${GO2_HW_PREFIX}/bin/python3" \
    -DPYTHON_EXECUTABLE="${GO2_HW_PREFIX}/bin/python3" \
    -DPython_FIND_STRATEGY=LOCATION
