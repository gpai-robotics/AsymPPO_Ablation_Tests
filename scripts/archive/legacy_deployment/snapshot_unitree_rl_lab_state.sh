#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${ROOT_DIR}/reference_repos/unitree_rl_lab"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${ROOT_DIR}/artifacts/unitree_rl_lab_snapshots/${STAMP}"

mkdir -p "${OUT_DIR}"

git -C "${REPO_DIR}" rev-parse HEAD > "${OUT_DIR}/base_commit.txt"
git -C "${REPO_DIR}" status --short > "${OUT_DIR}/status_short.txt"
git -C "${REPO_DIR}" status > "${OUT_DIR}/status.txt"
git -C "${REPO_DIR}" diff > "${OUT_DIR}/tracked_changes.patch"
git -C "${REPO_DIR}" diff --stat > "${OUT_DIR}/tracked_changes.stat"
git -C "${REPO_DIR}" ls-files --others --exclude-standard > "${OUT_DIR}/untracked_files.txt"

if [ -s "${OUT_DIR}/untracked_files.txt" ]; then
  while IFS= read -r relpath; do
    src="${REPO_DIR}/${relpath}"
    dst="${OUT_DIR}/untracked/${relpath}"
    if [ -d "${src}" ]; then
      mkdir -p "$(dirname "${dst}")"
      cp -a "${src}" "${dst}"
    elif [ -f "${src}" ]; then
      mkdir -p "$(dirname "${dst}")"
      cp -a "${src}" "${dst}"
    fi
  done < "${OUT_DIR}/untracked_files.txt"
fi

printf '[INFO] Snapshot written to %s\n' "${OUT_DIR}"
