#!/bin/bash
#SBATCH --job-name=dps_weighted
#SBATCH --output=dps_weighted_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=0-8

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
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"
BASE_CLUSTER="/scratch/weixuz/dps/preference_head/cluster_runs"
BASE_HEADS="/scratch/weixuz/dps/preference_head/cluster_heads"

TASKS=(
  "LAMP_1" "LAMP_2" "LAMP_3" "LAMP_4" "LAMP_5" "LAMP_7"
  "LongLaMP_2" "LongLaMP_3" "LongLaMP_4"
)

task="${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "${task}" ]; then
  echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

task_ds=$(echo "${task}" | sed 's/^LAMP_/LaMP-/; s/^LongLaMP_/LongLaMP-/')
K=$(python preference_head/compute_k.py --task "${task_ds}" --split dev --target_group "${TARGET_GROUP}")
if [ -z "${K}" ]; then
  echo "Failed to compute K for ${task}"
  exit 1
fi

task_slug=$(echo "${task}" | tr '[:upper:]' '[:lower:]' | tr -d '_')
cluster_dir="${BASE_CLUSTER}/${task_slug}_k${K}"
head_dir="${BASE_HEADS}/${task_slug}_k${K}"
dev_emb="${cluster_dir}/embeddings_dev.npy"

if [ ! -f "${cluster_dir}/clusters.json" ]; then
  echo "Skipping ${task} (missing ${cluster_dir}/clusters.json)"
  exit 0
fi
if [ ! -d "${head_dir}" ]; then
  echo "Skipping ${task} (missing ${head_dir})"
  exit 0
fi
if [ ! -f "${dev_emb}" ]; then
  echo "Skipping ${task} (missing ${dev_emb})"
  exit 0
fi

echo "========================================="
echo "Task: ${task} | k=${K} (target group ${TARGET_GROUP})"
echo "Clusters: ${cluster_dir}"
echo "Heads: ${head_dir}"
echo "Dev embeddings: ${dev_emb}"
echo "========================================="
python scripts/run_weighted_dps.py \
  --task "${task}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --cluster_file "${cluster_dir}/clusters.json" \
  --cluster_heads_dir "${head_dir}" \
  --embeddings_file "${dev_emb}" \
  --routing soft \
  --temperature 1.0
