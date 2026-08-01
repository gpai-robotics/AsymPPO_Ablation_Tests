#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NET_IFACE="${1:-enp0s31f6}"
if [ "$#" -gt 0 ]; then
  shift
fi
LOG_DIR="${ROOT_DIR}/logs/go2_new_robot_mapping"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/go2_new_robot_mapping_${STAMP}.log"

mkdir -p "${LOG_DIR}"

echo "[INFO] Writing new-robot mapping diagnostic log to ${LOG_FILE}"

cd "${ROOT_DIR}"

env GO2_DDS_IFACE="${NET_IFACE}" \
  stdbuf -oL -eL \
  python scripts/deploy/check_go2_new_robot_mapping.py --iface "${NET_IFACE}" "$@" 2>&1 | tee "${LOG_FILE}"
