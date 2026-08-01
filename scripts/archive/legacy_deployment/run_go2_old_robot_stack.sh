#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_IFACE="${GO2_DDS_IFACE:-enp0s31f6}"
ROLE="${1:-help}"
NET_IFACE="${2:-${DEFAULT_IFACE}}"

usage() {
  cat <<EOF
Usage:
  scripts/deploy/run_go2_old_robot_stack.sh odom [iface]
  scripts/deploy/run_go2_old_robot_stack.sh bridge [iface]
  scripts/deploy/run_go2_old_robot_stack.sh ctrl [iface]
  scripts/deploy/run_go2_old_robot_stack.sh summary [latest|logfile]
  scripts/deploy/run_go2_old_robot_stack.sh help

Roles:
  odom     Launch the ROS2 odometry stack for the old robot.
  bridge   Bridge /odometry/filtered to udp://127.0.0.1:5560.
  ctrl     Force old-robot deploy.yaml to odometry mode, then launch go2_ctrl with logging.
  summary  Summarize the latest old-robot go2_ctrl log or a specific logfile.
EOF
}

run_odom() {
  cd "${ROOT_DIR}"
  export GO2_DDS_IFACE="${NET_IFACE}"
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/deploy/source_local_odom_env.sh"
  ros2 launch go2_odometry go2_odometry_switch.launch.py odom_type:=use_full_odom
}

run_bridge() {
  cd "${ROOT_DIR}"
  export GO2_DDS_IFACE="${NET_IFACE}"
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/deploy/source_local_odom_env.sh"
  python "${ROOT_DIR}/scripts/deploy/bridge_odometry_to_udp.py"
}

run_ctrl() {
  cd "${ROOT_DIR}"
  python "${ROOT_DIR}/scripts/deploy/set_base_lin_vel_source.py" --source odometry
  "${ROOT_DIR}/scripts/deploy/run_go2_ctrl_logged.sh" "${NET_IFACE}"
}

run_summary() {
  cd "${ROOT_DIR}"
  local target="${1:-latest}"
  if [ "${target}" = "latest" ]; then
    target="$(ls -t "${ROOT_DIR}"/logs/go2_ctrl/*.log 2>/dev/null | head -n 1 || true)"
    if [ -z "${target}" ]; then
      echo "[ERROR] No go2_ctrl logs found under ${ROOT_DIR}/logs/go2_ctrl" >&2
      exit 1
    fi
  fi
  python "${ROOT_DIR}/scripts/deploy/summarize_go2_ctrl_log.py" --log-file "${target}"
}

case "${ROLE}" in
  odom)
    run_odom
    ;;
  bridge)
    run_bridge
    ;;
  ctrl)
    run_ctrl
    ;;
  summary)
    run_summary "${2:-latest}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[ERROR] Unknown role: ${ROLE}" >&2
    usage >&2
    exit 1
    ;;
esac
