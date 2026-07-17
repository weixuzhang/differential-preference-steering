#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --output=sanity_check_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories
hf_cache="/scratch/weixuz/dps/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_HUB_CACHE="${hf_cache}/hub"
export HF_DATASETS_CACHE="${hf_cache}/datasets"
export HF_EVALUATE_CACHE="${hf_cache}/evaluate"
export WANDB_DISABLED=true
export HF_OFFLINE=true

python scripts/sanity_check_offline.py
