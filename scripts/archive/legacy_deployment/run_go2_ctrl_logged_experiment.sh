#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${ROOT_DIR}/reference_repos/unitree_rl_lab_go2_old_robot_experiments"
BUILD_DIR="${RUNTIME_ROOT}/deploy/robots/go2/build"
BIN="${BUILD_DIR}/go2_ctrl"
NET_IFACE="${1:-enp0s31f6}"
LOG_DIR="${ROOT_DIR}/logs/go2_ctrl_experiment"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/go2_ctrl_experiment_${STAMP}.log"

if [ ! -x "${BIN}" ]; then
  echo "[ERROR] Missing experimental executable: ${BIN}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

cd "${BUILD_DIR}"

echo "[INFO] Writing experimental go2_ctrl log to ${LOG_FILE}"

env -i \
  HOME="${HOME}" \
  USER="${USER:-bhuvan}" \
  LOGNAME="${LOGNAME:-${USER:-bhuvan}}" \
  SHELL=/bin/bash \
  TERM="${TERM:-xterm}" \
  PATH="/usr/local/bin:/usr/bin:/bin" \
  stdbuf -oL -eL \
  "${BIN}" --network "${NET_IFACE}" 2>&1 | tee "${LOG_FILE}"
