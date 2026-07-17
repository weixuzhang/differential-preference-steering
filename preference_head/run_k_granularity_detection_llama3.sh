#!/bin/bash
#SBATCH --job-name=k_detect      # Job name
#SBATCH --output=k_detect_%j.out   # Output file (%j = job ID)
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --gpus-per-node=1          # Request 1 GPU
#SBATCH --time=24:00:00            # Max runtime (24 hours)
#SBATCH --mail-type=ALL

# k-granularity preference head detection (LLaMA3-8B-Instruct)
# Uses num_samples as a proxy for user granularity.

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories (same as decore)
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_OFFLINE=true

MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
TOP_PERCENT=0.04
BASE_DIR="/scratch/weixuz/preference_head/preference_scores_k"

ks=(1 2 5 10 25 50 100 200 2500)
tasks=(LaMP-1 LaMP-3)

for task in "${tasks[@]}"; do
  for k in "${ks[@]}"; do
    outdir="${BASE_DIR}/${task}/k${k}"
    mkdir -p "${outdir}"
    echo "-----------------------------------------"
    echo "Task: ${task} | k=${k}"
    echo "-----------------------------------------"
    python preference_head_detection.py \
      --model_path ${MODEL_PATH} \
      --task ${task} \
      --split dev \
      --num_samples ${k} \
      --top_percent ${TOP_PERCENT} \
      --save_dir "${outdir}"
    echo ""
  done

done
