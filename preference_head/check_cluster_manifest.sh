#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

TARGET_GROUP="${1:-100}"
CHUNK_SIZE="${2:-10}"
MANIFEST="${3:-${DPS_ROOT}/preference_head/cluster_head_manifest.tsv}"
SPLIT="${SPLIT:-dev}"

if [ ! -f "${MANIFEST}" ]; then
  echo "Missing manifest: ${MANIFEST}"
  exit 1
fi

TASKS=(
  "LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7"
  "LongLaMP-2" "LongLaMP-3" "LongLaMP-4"
)

total=0
echo "Task K chunks"
for task in "${TASKS[@]}"; do
  k=$(python "${DPS_ROOT}/preference_head/compute_k.py" \
    --task "${task}" \
    --split "${SPLIT}" \
    --target_group "${TARGET_GROUP}")
  if [ -z "${k}" ]; then
    echo "Failed to compute K for ${task}"
    exit 1
  fi
  chunks=$(( (k + CHUNK_SIZE - 1) / CHUNK_SIZE ))
  total=$(( total + chunks ))
  printf "%s\t%s\t%s\n" "${task}" "${k}" "${chunks}"
done

lines=$(wc -l < "${MANIFEST}")
echo "Manifest lines: ${lines} | Expected lines: ${total}"

if [ "${lines}" -ne "${total}" ]; then
  echo "Mismatch: rebuild manifest or check target group/chunk size."
  exit 1
fi
