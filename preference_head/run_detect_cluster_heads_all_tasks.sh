#!/bin/bash
#SBATCH --job-name=cluster_heads_all
#SBATCH --output=cluster_heads_all_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true

# ---- Config ----
TARGET_GROUP=100
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
BASE_CLUSTER="/scratch/weixuz/preference_head/cluster_runs"
BASE_OUT="/scratch/weixuz/preference_head/cluster_heads"

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
  cluster_dir="${BASE_CLUSTER}/${task_slug}_k${K}"
  out_dir="${BASE_OUT}/${task_slug}_k${K}"

  if [ ! -f "${cluster_dir}/clusters.json" ]; then
    echo "Skipping ${task} (missing ${cluster_dir}/clusters.json)"
    continue
  fi

  echo "========================================="
  echo "Task: ${task} | k=${K} (target group ${TARGET_GROUP})"
  echo "Clusters: ${cluster_dir}"
  echo "Output: ${out_dir}"
  echo "========================================="
  python preference_head/detect_cluster_heads.py \
    --cluster_file "${cluster_dir}/clusters.json" \
    --model_path "${MODEL_PATH}" \
    --task "${task}" \
    --split dev \
    --num_samples 100 \
    --save_dir "${out_dir}" \
    --top_percent 0.04 \
    --pcs_norm max \
    --pcs_power 1.0
done
