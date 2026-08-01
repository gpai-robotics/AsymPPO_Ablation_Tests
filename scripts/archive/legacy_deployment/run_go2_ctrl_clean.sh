#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/reference_repos/unitree_rl_lab/deploy/robots/go2/build"
BIN="${BUILD_DIR}/go2_ctrl"
NET_IFACE="${1:-enp0s31f6}"

if [ ! -x "${BIN}" ]; then
  echo "[ERROR] Missing executable: ${BIN}" >&2
  exit 1
fi

cd "${BUILD_DIR}"

exec env -i \
  HOME="${HOME}" \
  USER="${USER:-bhuvan}" \
  LOGNAME="${LOGNAME:-${USER:-bhuvan}}" \
  SHELL=/bin/bash \
  TERM="${TERM:-xterm}" \
  PATH="/usr/local/bin:/usr/bin:/bin" \
  "${BIN}" --network "${NET_IFACE}"
