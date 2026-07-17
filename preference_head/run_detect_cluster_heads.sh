#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

INDEX="${1:-${ARRAY_ID:-0}}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
BASE_CLUSTER="${BASE_CLUSTER:-${DPS_ROOT}/results/preference_head/cluster_runs}"
BASE_OUT="${BASE_OUT:-${DPS_ROOT}/preference_head/cluster_heads}"
MANIFEST="${MANIFEST:-${DPS_ROOT}/preference_head/cluster_head_manifest.tsv}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
TOP_PERCENT="${TOP_PERCENT:-0.04}"
PCS_NORM="${PCS_NORM:-max}"
PCS_POWER="${PCS_POWER:-1.0}"

if [ ! -f "${MANIFEST}" ]; then
  echo "Missing manifest: ${MANIFEST}"
  echo "Run: python preference_head/build_cluster_heads_manifest.py --split dev --target_group 100 --chunk_size 10 --output ${MANIFEST}"
  exit 1
fi

line=$(sed -n "$((INDEX + 1))p" "${MANIFEST}")
if [ -z "${line}" ]; then
  echo "Invalid index: ${INDEX} (manifest has fewer lines)"
  exit 1
fi

IFS=$'\t' read -r task K cluster_start cluster_end <<< "${line}"
task_slug=$(echo "${task}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
cluster_dir="${BASE_CLUSTER}/${task_slug}_k${K}"
out_dir="${BASE_OUT}/${task_slug}_k${K}"

if [ ! -f "${cluster_dir}/clusters.json" ]; then
  echo "Skipping ${task} (missing ${cluster_dir}/clusters.json)"
  exit 0
fi

echo "========================================="
echo "Task: ${task} | split=${SPLIT} | k=${K}"
echo "Clusters: ${cluster_dir}"
echo "Output: ${out_dir}"
echo "Cluster range: ${cluster_start}-${cluster_end}"
echo "========================================="
python "${DPS_ROOT}/preference_head/detect_cluster_heads.py" \
  --cluster_file "${cluster_dir}/clusters.json" \
  --model_path "${MODEL_PATH}" \
  --task "${task}" \
  --split "${SPLIT}" \
  --num_samples "${NUM_SAMPLES}" \
  --save_dir "${out_dir}" \
  --top_percent "${TOP_PERCENT}" \
  --pcs_norm "${PCS_NORM}" \
  --pcs_power "${PCS_POWER}" \
  --cluster_start "${cluster_start}" \
  --cluster_end "${cluster_end}"
