#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${ROOT_DIR}/artifacts/go2_readonly_signatures"
NET_IF_DEFAULT="enp0s31f6"
DURATION_DEFAULT="5"
PRINT_EVERY_DEFAULT="1"

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy/run_go2_readonly_signature_check.sh capture <label> [net_if] [duration_s]
  scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic <label> [net_if] [duration_s] [print_every_s]
  scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic-lowcmd <label> [net_if] [duration_s] [print_every_s]
  scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint <label> [net_if] [duration_s]
  scripts/deploy/run_go2_readonly_signature_check.sh compare <snapshot_a.json> <snapshot_b.json>
  scripts/deploy/run_go2_readonly_signature_check.sh compare-dynamic <a_series.jsonl> <a_lowcmd.jsonl> <b_series.jsonl> <b_lowcmd.jsonl>
  scripts/deploy/run_go2_readonly_signature_check.sh summarize <set_label> <snapshot1.json> <snapshot2.json> [...]
  scripts/deploy/run_go2_readonly_signature_check.sh prep

Examples:
  scripts/deploy/run_go2_readonly_signature_check.sh prep
  scripts/deploy/run_go2_readonly_signature_check.sh capture old_robot_stand
  scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint old_robot_stand enp0s31f6 8
  scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic old_robot_lateral_release enp0s31f6 20 0.25
  scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic-lowcmd old_robot_lateral_release enp0s31f6 20 0.25
  scripts/deploy/run_go2_readonly_signature_check.sh capture old_robot_sit enp0s31f6 8
  scripts/deploy/run_go2_readonly_signature_check.sh compare /tmp/a.json /tmp/b.json
  scripts/deploy/run_go2_readonly_signature_check.sh compare-dynamic /tmp/a_series.jsonl /tmp/a_lowcmd.jsonl /tmp/b_series.jsonl /tmp/b_lowcmd.jsonl
  scripts/deploy/run_go2_readonly_signature_check.sh summarize old_robot_stand /tmp/old1.json /tmp/old2.json /tmp/old3.json
EOF
}

ensure_root() {
  mkdir -p "${OUT_ROOT}"
}

timestamp_dir() {
  date +"%Y%m%d_%H%M%S"
}

cmd_prep() {
  ensure_root
  local stamp
  stamp="$(timestamp_dir)"
  local run_dir="${OUT_ROOT}/${stamp}"
  mkdir -p "${run_dir}"
  cat <<EOF
Prepared read-only signature workspace:
  ${run_dir}

Suggested captures:
  ${run_dir}/old_robot_sit.json
  ${run_dir}/old_robot_stand.json
  ${run_dir}/new_robot_sit.json
  ${run_dir}/new_robot_stand.json

Example:
  scripts/deploy/run_go2_readonly_signature_check.sh capture old_robot_stand ${NET_IF_DEFAULT} ${DURATION_DEFAULT}
EOF
}

cmd_capture() {
  local label="${1:-}"
  local net_if="${2:-${NET_IF_DEFAULT}}"
  local duration="${3:-${DURATION_DEFAULT}}"
  local include_blueprint="${4:-false}"
  if [[ -z "${label}" ]]; then
    usage
    exit 1
  fi

  ensure_root
  local stamp
  stamp="$(timestamp_dir)"
  local out_file="${OUT_ROOT}/${stamp}_${label}.json"

  echo "[INFO] Capturing read-only snapshot"
  echo "[INFO] label=${label}"
  echo "[INFO] net_if=${net_if}"
  echo "[INFO] duration=${duration}"
  echo "[INFO] include_blueprint=${include_blueprint}"
  echo "[INFO] out_file=${out_file}"

  (
    cd "${ROOT_DIR}"
    if [[ "${include_blueprint}" == "true" ]]; then
      python scripts/deploy/probe_go2_readonly.py \
        --net-if "${net_if}" \
        --duration "${duration}" \
        --print-every "${PRINT_EVERY_DEFAULT}" \
        --include-blueprint \
        --json-out "${out_file}"
    else
      python scripts/deploy/probe_go2_readonly.py \
        --net-if "${net_if}" \
        --duration "${duration}" \
        --print-every "${PRINT_EVERY_DEFAULT}" \
        --json-out "${out_file}"
    fi
  )

  echo
  echo "[INFO] Snapshot written:"
  echo "  ${out_file}"
}

cmd_capture_dynamic() {
  local subscribe_lowcmd="${1:-false}"
  shift || true
  local label="${1:-}"
  local net_if="${2:-${NET_IF_DEFAULT}}"
  local duration="${3:-20}"
  local print_every="${4:-0.25}"
  if [[ -z "${label}" ]]; then
    usage
    exit 1
  fi

  ensure_root
  local stamp
  stamp="$(timestamp_dir)"
  local summary_file="${OUT_ROOT}/${stamp}_${label}_summary.json"
  local series_file="${OUT_ROOT}/${stamp}_${label}_series.jsonl"
  local lowcmd_stream_file="${OUT_ROOT}/${stamp}_${label}_lowcmd_stream.jsonl"

  echo "[INFO] Capturing dynamic read-only series"
  echo "[INFO] label=${label}"
  echo "[INFO] net_if=${net_if}"
  echo "[INFO] duration=${duration}"
  echo "[INFO] print_every=${print_every}"
  echo "[INFO] subscribe_lowcmd=${subscribe_lowcmd}"
  echo "[INFO] summary_file=${summary_file}"
  echo "[INFO] series_file=${series_file}"
  if [[ "${subscribe_lowcmd}" == "true" ]]; then
    echo "[INFO] lowcmd_stream_file=${lowcmd_stream_file}"
  fi

  (
    cd "${ROOT_DIR}"
    if [[ "${subscribe_lowcmd}" == "true" ]]; then
      python scripts/deploy/probe_go2_readonly.py \
        --net-if "${net_if}" \
        --duration "${duration}" \
        --print-every "${print_every}" \
        --subscribe-lowcmd \
        --json-out "${summary_file}" \
        --series-jsonl-out "${series_file}" \
        --lowcmd-stream-jsonl-out "${lowcmd_stream_file}"
    else
      python scripts/deploy/probe_go2_readonly.py \
        --net-if "${net_if}" \
        --duration "${duration}" \
        --print-every "${print_every}" \
        --json-out "${summary_file}" \
        --series-jsonl-out "${series_file}"
    fi
  )

  echo
  echo "[INFO] Dynamic capture written:"
  echo "  summary: ${summary_file}"
  echo "  series : ${series_file}"
  if [[ "${subscribe_lowcmd}" == "true" ]]; then
    echo "  lowcmd : ${lowcmd_stream_file}"
  fi
}

cmd_compare() {
  local a="${1:-}"
  local b="${2:-}"
  if [[ -z "${a}" || -z "${b}" ]]; then
    usage
    exit 1
  fi

  (
    cd "${ROOT_DIR}"
    python scripts/deploy/compare_go2_readonly_snapshots.py --a "${a}" --b "${b}"
  )
}

cmd_compare_dynamic() {
  local a_series="${1:-}"
  local a_lowcmd="${2:-}"
  local b_series="${3:-}"
  local b_lowcmd="${4:-}"
  if [[ -z "${a_series}" || -z "${a_lowcmd}" || -z "${b_series}" || -z "${b_lowcmd}" ]]; then
    usage
    exit 1
  fi

  (
    cd "${ROOT_DIR}"
    python scripts/deploy/compare_go2_dynamic_signatures.py \
      --a-series "${a_series}" \
      --a-lowcmd "${a_lowcmd}" \
      --b-series "${b_series}" \
      --b-lowcmd "${b_lowcmd}"
  )
}

cmd_summarize() {
  local label="${1:-}"
  shift || true
  if [[ -z "${label}" || "$#" -lt 1 ]]; then
    usage
    exit 1
  fi

  (
    cd "${ROOT_DIR}"
    python scripts/deploy/summarize_go2_readonly_snapshot_set.py --label "${label}" "$@"
  )
}

main() {
  local subcommand="${1:-}"
  shift || true

  case "${subcommand}" in
    capture)
      cmd_capture "$@"
      ;;
    capture-blueprint)
      cmd_capture "${1:-}" "${2:-${NET_IF_DEFAULT}}" "${3:-${DURATION_DEFAULT}}" true
      ;;
    capture-dynamic)
      cmd_capture_dynamic false "$@"
      ;;
    capture-dynamic-lowcmd)
      cmd_capture_dynamic true "$@"
      ;;
    compare)
      cmd_compare "$@"
      ;;
    compare-dynamic)
      cmd_compare_dynamic "$@"
      ;;
    summarize)
      cmd_summarize "$@"
      ;;
    prep)
      cmd_prep
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
