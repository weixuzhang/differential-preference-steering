#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

TARGET_GROUP="${TARGET_GROUP:-100}"
SPLIT="${SPLIT:-dev}"
EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
BASE_OUT="${BASE_OUT:-${DPS_ROOT}/preference_head/cluster_runs}"

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=(
    "LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7"
    "LongLaMP-2" "LongLaMP-3" "LongLaMP-4"
  )
fi

for task in "${TASKS[@]}"; do
  K=$(python "${DPS_ROOT}/preference_head/compute_k.py" \
    --task "${task}" \
    --split "${SPLIT}" \
    --target_group "${TARGET_GROUP}")
  if [ -z "${K}" ]; then
    echo "Failed to compute K for ${task}"
    continue
  fi

  task_slug=$(echo "${task}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  out_dir="${BASE_OUT}/${task_slug}_k${K}"

  echo "========================================="
  echo "Task: ${task} | split=${SPLIT} | k=${K} (target group ${TARGET_GROUP})"
  echo "Output: ${out_dir}"
  echo "========================================="
  python "${DPS_ROOT}/preference_head/cluster_profiles.py" \
    --task "${task}" \
    --split "${SPLIT}" \
    --k "${K}" \
    --output_dir "${out_dir}" \
    --save_embeddings \
    --embedding_model "${EMB_MODEL}"
done
