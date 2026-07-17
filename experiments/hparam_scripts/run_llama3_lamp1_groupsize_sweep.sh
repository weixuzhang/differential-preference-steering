#!/bin/bash
#SBATCH --job-name=llama3_l1_groups
#SBATCH --output=llama3_l1_groups_%j.out
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
GROUP_SIZES=(10 50 100 200 400)
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"
NUM_SAMPLES=100
NUM_SAMPLES_SMALL=25
SMALL_GROUP_SIZE=10
CHUNK_SIZE_SMALL=25
TOP_PERCENT=$(python -c "print(40/1024)")

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")

echo "========================================="
echo "Group-size sweep | ${TASK}"
echo "Model: ${MODEL_NAME}"
echo "Group sizes: ${GROUP_SIZES[*]}"
echo "========================================="

for target_group in "${GROUP_SIZES[@]}"; do
  K=$(python "${ROOT}/preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${target_group}")
  if [ -z "${K}" ]; then
    echo "Failed to compute K for ${TASK} (target_group=${target_group})"
    continue
  fi

  cluster_dir="${ROOT}/preference_head/cluster_runs/${task_slug}_k${K}"
  head_dir="${ROOT}/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}"
  emb_file="${cluster_dir}/embeddings_dev.npy"
  run_dir="${ROOT}/dps/outputs/hparam/groupsize/g${target_group}"

  echo "-----------------------------------------"
  echo "Target group: ${target_group} | k=${K}"
  echo "Clusters: ${cluster_dir}"
  echo "Heads: ${head_dir}"
  echo "Run dir: ${run_dir}"
  echo "-----------------------------------------"

  if [ ! -f "${cluster_dir}/clusters.json" ]; then
    echo "Clustering dev profiles..."
    python "${ROOT}/preference_head/cluster_profiles.py" \
      --task "${TASK}" \
      --split dev \
      --k "${K}" \
      --output_dir "${cluster_dir}" \
      --save_embeddings \
      --embedding_model "${EMB_MODEL}"
  else
    echo "Clusters already exist, skipping."
  fi

  if [ ! -f "${emb_file}" ]; then
    echo "Building dev embeddings for routing..."
    python "${ROOT}/preference_head/embed_profiles.py" \
      --task "${TASK}" \
      --split dev \
      --output_file "${emb_file}" \
      --meta_file "${cluster_dir}/embeddings_dev.json" \
      --embedding_model "${EMB_MODEL}"
  else
    echo "Embeddings already exist, skipping."
  fi

  if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
    echo "Detecting cluster heads..."
    if [ "${target_group}" -eq "${SMALL_GROUP_SIZE}" ]; then
      echo "Small group size detected; reducing num_samples and chunking detection."
      cluster_start=0
      while [ "${cluster_start}" -lt "${K}" ]; do
        cluster_end=$((cluster_start + CHUNK_SIZE_SMALL - 1))
        if [ "${cluster_end}" -ge "${K}" ]; then
          cluster_end=$((K - 1))
        fi
        python "${ROOT}/preference_head/detect_cluster_heads.py" \
          --cluster_file "${cluster_dir}/clusters.json" \
          --model_path "${MODEL_PATH}" \
          --task "${TASK}" \
          --split dev \
          --num_samples "${NUM_SAMPLES_SMALL}" \
          --save_dir "${head_dir}" \
          --top_percent "${TOP_PERCENT}" \
          --pcs_norm max \
          --pcs_power 1.0 \
          --cluster_start "${cluster_start}" \
          --cluster_end "${cluster_end}"
        cluster_start=$((cluster_end + 1))
      done
    else
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
    fi
  else
    echo "Head weights already exist, skipping detection."
  fi

  mkdir -p "${run_dir}"
  python "${ROOT}/dps/scripts/run_weighted_dps.py" \
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
