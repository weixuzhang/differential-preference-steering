#!/bin/bash
#SBATCH --job-name=embed_dev
#SBATCH --output=embed_dev_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories
hf_cache="/scratch/weixuz/dps/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true

# ---- Config ----
TARGET_GROUP=100
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
BASE_OUT="/scratch/weixuz/preference_head/cluster_runs"

TASKS=(
  "LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7"
  "LongLaMP-2" "LongLaMP-3" "LongLaMP-4"
)

for task in "${TASKS[@]}"; do
K=$(python preference_head/compute_k.py --task "${task}" --split dev --target_group "${TARGET_GROUP}")
  if [ -z "${K}" ]; then
    echo "Failed to compute K for ${task}"
    continue
  fi

  task_slug=$(echo "${task}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  out_dir="${BASE_OUT}/${task_slug}_k${K}"
  mkdir -p "${out_dir}"

  emb_path="${out_dir}/embeddings_dev.npy"
  meta_path="${out_dir}/embeddings_dev.json"

  echo "========================================="
  echo "Task: ${task} | split=dev"
  echo "Output: ${emb_path}"
  echo "========================================="
  python preference_head/embed_profiles.py \
    --task "${task}" \
    --split dev \
    --output_file "${emb_path}" \
    --meta_file "${meta_path}" \
    --embedding_model "${EMB_MODEL}"
done
