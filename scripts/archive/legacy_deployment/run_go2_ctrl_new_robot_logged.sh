#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEW_RUNTIME_DIR="${ROOT_DIR}/reference_repos/unitree_rl_lab/deploy/robots/go2_new_robot_runtime"
RUNTIME_BUILD_DIR="${NEW_RUNTIME_DIR}/build"
RUNTIME_CONFIG_DIR="${NEW_RUNTIME_DIR}/config"
RUNTIME_BIN="${RUNTIME_BUILD_DIR}/go2_ctrl"

SRC_CONFIG="${ROOT_DIR}/reference_repos/unitree_rl_lab/deploy/robots/go2/config/config_new_robot.yaml"
SRC_POLICY_DIR="${ROOT_DIR}/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final_new_robot"
RUNTIME_POLICY_DIR="${RUNTIME_CONFIG_DIR}/policy/velocity/c1_blind_rough_omni_usable_v1_final_new_robot"

NET_IFACE="${1:-enp0s31f6}"
LOG_DIR="${ROOT_DIR}/logs/go2_ctrl_new_robot"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/go2_ctrl_new_robot_${STAMP}.log"

if [ ! -f "${NEW_RUNTIME_DIR}/CMakeLists.txt" ]; then
  echo "[ERROR] Missing new-robot runtime source tree: ${NEW_RUNTIME_DIR}" >&2
  exit 1
fi

if [ ! -f "${SRC_CONFIG}" ]; then
  echo "[ERROR] Missing config: ${SRC_CONFIG}" >&2
  exit 1
fi

if [ ! -d "${SRC_POLICY_DIR}" ]; then
  echo "[ERROR] Missing policy bundle: ${SRC_POLICY_DIR}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_BUILD_DIR}"
mkdir -p "${RUNTIME_CONFIG_DIR}/policy/velocity"
mkdir -p "${LOG_DIR}"

cp "${SRC_CONFIG}" "${RUNTIME_CONFIG_DIR}/config.yaml"
rm -rf "${RUNTIME_POLICY_DIR}"
cp -a "${SRC_POLICY_DIR}" "${RUNTIME_POLICY_DIR}"

cmake -S "${NEW_RUNTIME_DIR}" -B "${RUNTIME_BUILD_DIR}"
cd "${RUNTIME_BUILD_DIR}"
cmake --build . --target go2_ctrl -j

echo "[INFO] Writing new-robot go2_ctrl log to ${LOG_FILE}"

env -i \
  HOME="${HOME}" \
  USER="${USER:-bhuvan}" \
  LOGNAME="${LOGNAME:-${USER:-bhuvan}}" \
  SHELL=/bin/bash \
  TERM="${TERM:-xterm}" \
  PATH="/usr/local/bin:/usr/bin:/bin" \
  stdbuf -oL -eL \
  "${RUNTIME_BIN}" --network "${NET_IFACE}" 2>&1 | tee "${LOG_FILE}"
