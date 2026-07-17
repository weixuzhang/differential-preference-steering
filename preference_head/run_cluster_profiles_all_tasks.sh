#!/bin/bash
#SBATCH --job-name=cluster_all
#SBATCH --output=cluster_all_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories
hf_cache=".cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true

# ---- Config ----
TARGET_GROUP=100
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
BASE_OUT="results/preference_head/cluster_runs"

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
  echo "========================================="
  echo "Task: ${task} | k=${K} (target group ${TARGET_GROUP})"
  echo "Output: ${out_dir}"
  echo "========================================="
  python preference_head/cluster_profiles.py \
    --task "${task}" \
    --split dev \
    --k "${K}" \
    --output_dir "${out_dir}" \
    --save_embeddings \
    --embedding_model "${EMB_MODEL}"
done
