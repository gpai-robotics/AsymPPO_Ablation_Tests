#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS_DIR="${ROOT_DIR}/reference_repos/odom_ws"
GO2_HW_PREFIX="${CONDA_PREFIX:-/home/bhuvan/miniconda3/envs/go2-hw}"
UNITREE_ROS2_WS_DIR="${ROOT_DIR}/reference_repos/unitree_ros2/cyclonedds_ws"

# ROS setup files are not consistently safe under `set -u`, so temporarily
# relax nounset while sourcing them.
_had_nounset=0
case $- in
  *u*) _had_nounset=1 ;;
esac
set +u
source /opt/ros/humble/setup.bash
if [ -f "${UNITREE_ROS2_WS_DIR}/install/setup.bash" ]; then
  source "${UNITREE_ROS2_WS_DIR}/install/setup.bash"
fi
if [ "${_had_nounset}" -eq 1 ]; then
  set -u
fi
unset _had_nounset

# Match the Unitree robot DDS transport explicitly so ROS2 nodes in this
# workspace can see low-level Go2 topics like /lowstate on the robot NIC.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
GO2_DDS_IFACE="${GO2_DDS_IFACE:-enp0s31f6}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${GO2_DDS_IFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>}"

# Keep runtime discovery explicit so we don't depend on generated setup files
# for partially-built packages.
export AMENT_PREFIX_PATH="${WS_DIR}/install/go2_odometry:${WS_DIR}/install/unitree_go:${WS_DIR}/install/unitree_description:${WS_DIR}/install/inekf:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="${GO2_HW_PREFIX}:${WS_DIR}/install/go2_odometry:${WS_DIR}/install/unitree_go:${WS_DIR}/install/unitree_description:${WS_DIR}/install/inekf:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${GO2_HW_PREFIX}/lib:${WS_DIR}/install/unitree_go/lib:${WS_DIR}/install/inekf/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:${WS_DIR}/install/go2_odometry/lib/go2_odometry:${WS_DIR}/install/unitree_go/lib/python3.10/site-packages:${WS_DIR}/install/unitree_description/lib/python3.10/site-packages:${WS_DIR}/install/inekf/lib/python3.10/site-packages:${PYTHONPATH:-}"
