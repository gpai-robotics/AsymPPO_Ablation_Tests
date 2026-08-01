#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NET_IF="${1:-enp0s31f6}"
DURATION_S="${2:-20}"
PRINT_EVERY_S="${3:-0.5}"
LABEL="${4:-latency_probe}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/artifacts/go2_latency_probe"
SUMMARY_OUT="${OUT_DIR}/${STAMP}_${LABEL}_summary.json"
SERIES_OUT="${OUT_DIR}/${STAMP}_${LABEL}_series.jsonl"
LOWSTATE_OUT="${OUT_DIR}/${STAMP}_${LABEL}_lowstate_stream.jsonl"
LOWCMD_OUT="${OUT_DIR}/${STAMP}_${LABEL}_lowcmd_stream.jsonl"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"
echo "[INFO] Capturing Go2 latency probe"
echo "[INFO] summary=${SUMMARY_OUT}"
echo "[INFO] series=${SERIES_OUT}"
echo "[INFO] lowstate=${LOWSTATE_OUT}"
echo "[INFO] lowcmd=${LOWCMD_OUT}"

python scripts/deploy/probe_go2_readonly.py \
  --net-if "${NET_IF}" \
  --duration "${DURATION_S}" \
  --print-every "${PRINT_EVERY_S}" \
  --subscribe-lowcmd \
  --json-out "${SUMMARY_OUT}" \
  --series-jsonl-out "${SERIES_OUT}" \
  --lowstate-stream-jsonl-out "${LOWSTATE_OUT}" \
  --lowcmd-stream-jsonl-out "${LOWCMD_OUT}"

echo
echo "[INFO] Analyze with:"
echo "python scripts/deploy/analyze_go2_latency_probe.py --lowstate-jsonl ${LOWSTATE_OUT} --lowcmd-jsonl ${LOWCMD_OUT}"
