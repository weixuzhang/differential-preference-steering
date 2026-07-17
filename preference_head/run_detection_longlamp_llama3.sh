#!/bin/bash
#SBATCH --job-name=pref_longlamp      # Job name
#SBATCH --output=pref_longlamp_%j.out   # Output file (%j = job ID)
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --gpus-per-node=1          # Request 1 GPU
#SBATCH --time=06:00:00            # Max runtime (6 hours)
#SBATCH --mail-type=ALL

# Preference head detection for LongLaMP tasks (LLaMA3-8B-Instruct)

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories (same as decore)
hf_cache="/scratch/weixuz/dps/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_OFFLINE=true

MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SAMPLES=400
TOP_PERCENT=0.04
SAVE_DIR="/scratch/weixuz/preference_head/preference_scores"

mkdir -p ${SAVE_DIR}

echo "========================================="
echo "Preference Head Detection - LongLaMP"
echo "Model: ${MODEL_PATH}"
echo "Samples: ${NUM_SAMPLES}"
echo "Top %: ${TOP_PERCENT}"
echo "Save dir: ${SAVE_DIR}"
echo "========================================="

tasks=(LongLaMP-2 LongLaMP-3 LongLaMP-4)
for task in "${tasks[@]}"; do
  echo "-----------------------------------------"
  echo "Detecting preference heads for ${task}"
  echo "-----------------------------------------"
  python preference_head_detection.py \
    --model_path ${MODEL_PATH} \
    --task ${task} \
    --split dev \
    --num_samples ${NUM_SAMPLES} \
    --top_percent ${TOP_PERCENT} \
    --save_dir ${SAVE_DIR}
  echo ""
done

echo "========================================="
echo "Detection complete!"
echo "========================================="
