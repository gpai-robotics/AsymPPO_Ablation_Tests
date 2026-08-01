#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS_DIR="${ROOT_DIR}/reference_repos/odom_ws"
SRC_DIR="${WS_DIR}/src"

mkdir -p "${SRC_DIR}"

link_repo() {
  local name="$1"
  local target="$2"
  local link_path="${SRC_DIR}/${name}"
  if [[ -L "${link_path}" || -e "${link_path}" ]]; then
    rm -rf "${link_path}"
  fi
  ln -s "${target}" "${link_path}"
}

# Core odometry stack.
link_repo "go2_odometry" "${ROOT_DIR}/reference_repos/go2_odometry"
link_repo "inekf" "${ROOT_DIR}/reference_repos/invariant-ekf"
link_repo "unitree_description" "${ROOT_DIR}/reference_repos/unitree_description"

# Unitree ROS2 message packages.
link_repo "unitree_go" "${ROOT_DIR}/reference_repos/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go"
link_repo "unitree_api" "${ROOT_DIR}/reference_repos/unitree_ros2/cyclonedds_ws/src/unitree/unitree_api"
link_repo "unitree_hg" "${ROOT_DIR}/reference_repos/unitree_ros2/cyclonedds_ws/src/unitree/unitree_hg"

cat <<EOF
Local odometry workspace prepared:
  ${WS_DIR}

Packages linked into src/:
  go2_odometry
  inekf
  unitree_description
  unitree_go
  unitree_api
  unitree_hg

Next steps:
  source /opt/ros/humble/setup.bash
  cd ${WS_DIR}
  colcon build --symlink-install
EOF
