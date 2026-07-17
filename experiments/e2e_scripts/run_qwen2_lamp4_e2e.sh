#!/bin/bash
#SBATCH --job-name=qwen2_lamp4_e2e
#SBATCH --output=qwen2_lamp4_e2e_%j.out
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

# ---- Config ----
TASK="LaMP-4"
TASK_DECODER="LAMP_4"
TARGET_GROUP=100
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH="Qwen/Qwen2-7B-Instruct"
MODEL_NAME="Qwen2-7B-Instruct"

K=$(python "preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
if [ -z "${K}" ]; then
  echo "Failed to compute K for ${TASK}"
  exit 1
fi

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
cluster_dir="results/preference_head/cluster_runs/${task_slug}_k${K}"
model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")
head_dir="results/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}"

echo "========================================="
echo "Task: ${TASK} | k=${K} (target group ${TARGET_GROUP})"
echo "Model: ${MODEL_NAME}"
echo "Clusters: ${cluster_dir}"
echo "Heads: ${head_dir}"
echo "========================================="

echo "[1/4] Clustering dev profiles..."
python "preference_head/cluster_profiles.py"   --task "${TASK}"   --split dev   --k "${K}"   --output_dir "${cluster_dir}"   --save_embeddings   --embedding_model "${EMB_MODEL}"

echo "[2/4] Building dev embeddings for routing..."
python "preference_head/embed_profiles.py"   --task "${TASK}"   --split dev   --output_file "${cluster_dir}/embeddings_dev.npy"   --meta_file "${cluster_dir}/embeddings_dev.json"   --embedding_model "${EMB_MODEL}"

echo "[3/4] Detecting cluster heads (${MODEL_NAME})..."
python "preference_head/detect_cluster_heads.py"   --cluster_file "${cluster_dir}/clusters.json"   --model_path "${MODEL_PATH}"   --task "${TASK}"   --split dev   --num_samples 100   --save_dir "${head_dir}"   --top_percent 0.04   --pcs_norm max   --pcs_power 1.0   --cluster_start 0   --cluster_end "$((K - 1))"

echo "[4/4] Running weighted DPS..."
python "$(pwd)/scripts/run_weighted_dps.py"   --task "${TASK_DECODER}"   --model_path "${MODEL_PATH}"   --model_name "${MODEL_NAME}"   --model_type instruct   --cluster_file "${cluster_dir}/clusters.json"   --cluster_heads_dir "${head_dir}"   --embeddings_file "${cluster_dir}/embeddings_dev.npy"   --routing soft   --temperature 1.0
