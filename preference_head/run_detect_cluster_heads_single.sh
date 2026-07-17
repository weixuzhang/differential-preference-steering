#!/bin/bash
#SBATCH --job-name=cluster_heads_l1
#SBATCH --output=cluster_heads_l1_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
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
TASK="LaMP-1"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
TARGET_GROUP=100
K=$(python preference_head/compute_k.py --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
CLUSTER_DIR="/scratch/weixuz/preference_head/cluster_runs/lamp1_k${K}"
OUT_DIR="/scratch/weixuz/preference_head/cluster_heads/lamp1_k${K}"

python preference_head/detect_cluster_heads.py \
  --cluster_file "${CLUSTER_DIR}/clusters.json" \
  --model_path "${MODEL_PATH}" \
  --task "${TASK}" \
  --split dev \
  --num_samples 100 \
  --save_dir "${OUT_DIR}" \
  --pcs_norm max \
  --pcs_power 1.0
