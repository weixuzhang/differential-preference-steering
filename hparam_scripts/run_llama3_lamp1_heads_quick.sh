#!/bin/bash
set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

# Cache directories
hf_cache="${ROOT}/decore/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true
export HF_DATASETS_OFFLINE=true
export TRANSFORMERS_OFFLINE=true
export HF_DATASETS_SINGLE_THREAD=true
export TOKENIZERS_PARALLELISM=false

# ---- Config ----
TASK="LaMP-1"
TASK_DECODER="LAMP_1"
TARGET_GROUP=100
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"
HEAD_COUNTS=(10 20 40 80 160)
DETECT_SAMPLES=50
EVAL_SAMPLES=50

K=$(python "${ROOT}/preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
if [ -z "${K}" ]; then
  echo "Failed to compute K for ${TASK}"
  exit 1
fi

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")
cluster_dir="${ROOT}/preference_head/cluster_runs/${task_slug}_k${K}"
emb_file="${cluster_dir}/embeddings_dev.npy"

if [ ! -f "${cluster_dir}/clusters.json" ]; then
  echo "Missing ${cluster_dir}/clusters.json (skip clustering per request)."
  exit 1
fi
if [ ! -f "${emb_file}" ]; then
  echo "Missing ${emb_file} (skip embedding per request)."
  exit 1
fi

MAX_HEADS=160
TOP_PERCENT_MAX=$(python -c "print(${MAX_HEADS}/1024)")
BASE_HEAD_DIR="${ROOT}/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}_quick50_base"

echo "========================================="
echo "Heads sweep (quick) | ${TASK} | k=${K}"
echo "Model: ${MODEL_NAME}"
echo "GPU: ${GPU_ID}"
echo "Detect once: ${BASE_HEAD_DIR}"
echo "Eval samples: ${EVAL_SAMPLES}"
echo "========================================="

if ! ls "${BASE_HEAD_DIR}/cluster_00/"*_ranked.json >/dev/null 2>&1; then
  echo "Detecting cluster heads once (for ranked PCS)..."
  python "${ROOT}/preference_head/detect_cluster_heads.py" \
    --cluster_file "${cluster_dir}/clusters.json" \
    --model_path "${MODEL_PATH}" \
    --task "${TASK}" \
    --split dev \
    --num_samples "${DETECT_SAMPLES}" \
    --max_samples_per_cluster "${DETECT_SAMPLES}" \
    --save_dir "${BASE_HEAD_DIR}" \
    --top_percent "${TOP_PERCENT_MAX}" \
    --pcs_norm max \
    --pcs_power 1.0 \
    --cluster_start 0 \
    --cluster_end "$((K - 1))"
else
  echo "Ranked heads already exist in ${BASE_HEAD_DIR}, skipping detection."
fi

for num_heads in "${HEAD_COUNTS[@]}"; do
  head_dir="${ROOT}/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}_quick50_h${num_heads}"
  run_dir="${ROOT}/decore/outputs/hparam/heads_quick/h${num_heads}"

  if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
    python "${ROOT}/preference_head/build_head_weights_from_ranked.py" \
      --src_dir "${BASE_HEAD_DIR}" \
      --out_dir "${head_dir}" \
      --num_heads "${num_heads}" \
      --pcs_norm max \
      --pcs_power 1.0
  fi

  mkdir -p "${run_dir}"
  python "${ROOT}/decore/scripts/run_weighted_dps.py" \
    --task "${TASK_DECODER}" \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_type instruct \
    --cluster_file "${cluster_dir}/clusters.json" \
    --cluster_heads_dir "${head_dir}" \
    --embeddings_file "${emb_file}" \
    --routing soft \
    --temperature 1.0 \
    --num_samples "${EVAL_SAMPLES}" \
    --run_dir "${run_dir}"
done

echo "Heads sweep complete."
