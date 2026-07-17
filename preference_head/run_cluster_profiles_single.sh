#!/bin/bash
#SBATCH --job-name=cluster_lamp1
#SBATCH --output=cluster_lamp1_%j.out
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
TASK="LaMP-1"
TARGET_GROUP=100
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
K=$(python preference_head/compute_k.py --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
OUT_DIR="/scratch/weixuz/preference_head/cluster_runs/lamp1_k${K}"

python preference_head/cluster_profiles.py \
  --task "${TASK}" \
  --split dev \
  --k "${K}" \
  --output_dir "${OUT_DIR}" \
  --save_embeddings \
  --embedding_model "${EMB_MODEL}"
