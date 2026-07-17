#!/bin/bash
#SBATCH --job-name=llama3_l1_heads
#SBATCH --output=llama3_l1_heads_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL

set -euo pipefail

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

# Cache directories
hf_cache="$(pwd)/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true
export HF_DATASETS_OFFLINE=true
export TRANSFORMERS_OFFLINE=true
export HF_DATASETS_SINGLE_THREAD=true

# ---- Config ----
TASK="LaMP-1"
TASK_DECODER="LAMP_1"
TARGET_GROUP=100
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"
HEAD_COUNTS=(10 20 40 80 160)
NUM_SAMPLES=100

K=$(python "preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
if [ -z "${K}" ]; then
  echo "Failed to compute K for ${TASK}"
  exit 1
fi

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
cluster_dir="results/preference_head/cluster_runs/${task_slug}_k${K}"
model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")
emb_file="${cluster_dir}/embeddings_dev.npy"

echo "========================================="
echo "Head-count sweep | ${TASK} | k=${K}"
echo "Model: ${MODEL_NAME}"
echo "Clusters: ${cluster_dir}"
echo "Embeddings: ${emb_file}"
echo "========================================="

if [ ! -f "${cluster_dir}/clusters.json" ]; then
  echo "[1/3] Clustering dev profiles..."
  python "preference_head/cluster_profiles.py" \
    --task "${TASK}" \
    --split dev \
    --k "${K}" \
    --output_dir "${cluster_dir}" \
    --save_embeddings \
    --embedding_model "${EMB_MODEL}"
else
  echo "[1/3] Clusters already exist, skipping."
fi

if [ ! -f "${emb_file}" ]; then
  echo "[2/3] Building dev embeddings for routing..."
  python "preference_head/embed_profiles.py" \
    --task "${TASK}" \
    --split dev \
    --output_file "${emb_file}" \
    --meta_file "${cluster_dir}/embeddings_dev.json" \
    --embedding_model "${EMB_MODEL}"
else
  echo "[2/3] Embeddings already exist, skipping."
fi

echo "[3/3] Detect heads + run weighted DPS for each head count..."
for num_heads in "${HEAD_COUNTS[@]}"; do
  top_percent=$(python -c "print(${num_heads}/1024)")
  head_dir="results/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}_h${num_heads}"
  run_dir="$(pwd)/outputs/hparam/heads/h${num_heads}"

  echo "-----------------------------------------"
  echo "Heads: ${num_heads} (top_percent=${top_percent})"
  echo "Head dir: ${head_dir}"
  echo "Run dir: ${run_dir}"
  echo "-----------------------------------------"

  if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
    echo "Detecting cluster heads..."
    python "preference_head/detect_cluster_heads.py" \
      --cluster_file "${cluster_dir}/clusters.json" \
      --model_path "${MODEL_PATH}" \
      --task "${TASK}" \
      --split dev \
      --num_samples "${NUM_SAMPLES}" \
      --save_dir "${head_dir}" \
      --top_percent "${top_percent}" \
      --pcs_norm max \
      --pcs_power 1.0 \
      --cluster_start 0 \
      --cluster_end "$((K - 1))"
  else
    echo "Head weights already exist, skipping detection."
  fi

  mkdir -p "${run_dir}"
  python "$(pwd)/scripts/run_weighted_dps.py" \
    --task "${TASK_DECODER}" \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_type instruct \
    --cluster_file "${cluster_dir}/clusters.json" \
    --cluster_heads_dir "${head_dir}" \
    --embeddings_file "${emb_file}" \
    --routing soft \
    --temperature 1.0 \
    --run_dir "${run_dir}"
done
