#!/bin/bash
#SBATCH --job-name=llama3_l1_ablate
#SBATCH --output=llama3_l1_ablate_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL

set -euo pipefail

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

# Cache directories
hf_cache="${ROOT}/dps/.cache/huggingface"
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
NUM_SAMPLES=100
TOP_PERCENT=$(python -c "print(40/1024)")
RANDOM_SEED=1234

K=$(python "${ROOT}/preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
if [ -z "${K}" ]; then
  echo "Failed to compute K for ${TASK}"
  exit 1
fi

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
cluster_dir="${ROOT}/preference_head/cluster_runs/${task_slug}_k${K}"
model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")
head_dir="${ROOT}/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}"
emb_file="${cluster_dir}/embeddings_dev.npy"
random_head_dir="${head_dir}_random"
random_mask_dir="${head_dir}_random_mask"
agg_heads="${head_dir}/aggregate_top_heads.json"

echo "========================================="
echo "Ablation | ${TASK} | k=${K}"
echo "Model: ${MODEL_NAME}"
echo "Clusters: ${cluster_dir}"
echo "Heads: ${head_dir}"
echo "Random heads: ${random_head_dir}"
echo "========================================="

if [ ! -f "${cluster_dir}/clusters.json" ]; then
  echo "[1/3] Clustering dev profiles..."
  python "${ROOT}/preference_head/cluster_profiles.py" \
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
  python "${ROOT}/preference_head/embed_profiles.py" \
    --task "${TASK}" \
    --split dev \
    --output_file "${emb_file}" \
    --meta_file "${cluster_dir}/embeddings_dev.json" \
    --embedding_model "${EMB_MODEL}"
else
  echo "[2/3] Embeddings already exist, skipping."
fi

if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
  echo "[3/3] Detecting cluster heads..."
  python "${ROOT}/preference_head/detect_cluster_heads.py" \
    --cluster_file "${cluster_dir}/clusters.json" \
    --model_path "${MODEL_PATH}" \
    --task "${TASK}" \
    --split dev \
    --num_samples "${NUM_SAMPLES}" \
    --save_dir "${head_dir}" \
    --top_percent "${TOP_PERCENT}" \
    --pcs_norm max \
    --pcs_power 1.0 \
    --cluster_start 0 \
    --cluster_end "$((K - 1))"
else
  echo "[3/3] Head weights already exist, skipping detection."
fi

echo "Aggregating cluster heads for validation..."
python "${ROOT}/preference_head/aggregate_cluster_heads.py" \
  --cluster_heads_dir "${head_dir}" \
  --top_percent "${TOP_PERCENT}" \
  --output_file "${agg_heads}"

echo "Validating head quality (global aggregation)..."
python "${ROOT}/preference_head/validate_preference_heads.py" \
  --model_path "${MODEL_PATH}" \
  --preference_heads_file "${agg_heads}" \
  --task "${TASK}" \
  --num_samples "${NUM_SAMPLES}"

echo "Creating randomized head weights..."
python "${ROOT}/preference_head/randomize_cluster_head_weights.py" \
  --src_dir "${head_dir}" \
  --out_dir "${random_head_dir}" \
  --seed "${RANDOM_SEED}"

echo "Running weighted DPS with random heads..."
mkdir -p "${ROOT}/dps/outputs/hparam/ablation/random_heads"
python "${ROOT}/dps/scripts/run_weighted_dps.py" \
  --task "${TASK_DECODER}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type instruct \
  --cluster_file "${cluster_dir}/clusters.json" \
  --cluster_heads_dir "${random_head_dir}" \
  --embeddings_file "${emb_file}" \
  --routing soft \
  --temperature 1.0 \
  --run_dir "${ROOT}/dps/outputs/hparam/ablation/random_heads"

echo "Creating random head masks (true masking)..."
python "${ROOT}/preference_head/random_mask_cluster_head_weights.py" \
  --src_dir "${head_dir}" \
  --out_dir "${random_mask_dir}" \
  --seed "${RANDOM_SEED}" \
  --top_percent "${TOP_PERCENT}"

echo "Running weighted DPS with random head masks..."
mkdir -p "${ROOT}/dps/outputs/hparam/ablation/random_mask"
python "${ROOT}/dps/scripts/run_weighted_dps.py" \
  --task "${TASK_DECODER}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type instruct \
  --cluster_file "${cluster_dir}/clusters.json" \
  --cluster_heads_dir "${random_mask_dir}" \
  --embeddings_file "${emb_file}" \
  --routing soft \
  --temperature 1.0 \
  --run_dir "${ROOT}/dps/outputs/hparam/ablation/random_mask"
