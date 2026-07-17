#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

TARGET_GROUP="${1:-100}"
CHUNK_SIZE="${2:-10}"
OUT_PATH="${3:-${DPS_ROOT}/preference_head/cluster_head_manifest.tsv}"
SPLIT="${SPLIT:-dev}"
TASKS="${TASKS:-}"

EXTRA_ARGS=()
if [[ -n "${TASKS}" ]]; then
  EXTRA_ARGS+=(--tasks "${TASKS}")
fi

python "${DPS_ROOT}/preference_head/build_cluster_heads_manifest.py" \
  --split "${SPLIT}" \
  --target_group "${TARGET_GROUP}" \
  --chunk_size "${CHUNK_SIZE}" \
  --output "${OUT_PATH}" \
  "${EXTRA_ARGS[@]}"
