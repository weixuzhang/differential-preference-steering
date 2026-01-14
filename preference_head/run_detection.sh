#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-400}"
TOP_PERCENT="${TOP_PERCENT:-0.04}"
SAVE_DIR="${SAVE_DIR:-${DPS_ROOT}/preference_head/preference_scores}"

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=("LaMP-1")
fi

mkdir -p "${SAVE_DIR}"

echo "========================================="
echo "Preference Head Detection"
echo "Model: ${MODEL_PATH}"
echo "Split: ${SPLIT}"
echo "Samples: ${NUM_SAMPLES}"
echo "Top %: ${TOP_PERCENT}"
echo "Save dir: ${SAVE_DIR}"
echo "========================================="

for task in "${TASKS[@]}"; do
  echo "Detecting preference heads for ${task}..."
  python "${DPS_ROOT}/preference_head/preference_head_detection.py" \
    --model_path "${MODEL_PATH}" \
    --task "${task}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --top_percent "${TOP_PERCENT}" \
    --save_dir "${SAVE_DIR}"
done

echo "========================================="
echo "Detection complete!"
echo "========================================="
