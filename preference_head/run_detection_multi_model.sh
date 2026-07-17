#!/bin/bash
#SBATCH --job-name=pref_head_multi
#SBATCH --output=preference_head_detection_multi_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL

# Preference Head Detection Script for Multiple Models
# Detects preference heads for all models on all LaMP & LongLaMP tasks

# Activate environment
source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_OFFLINE=true

# Detection parameters
NUM_SAMPLES=400  # Number of samples for detection
TOP_PERCENT=0.04  # Select top 4% of heads (40 heads)
SAVE_DIR="/scratch/weixuz/preference_head/preference_scores"

# Create save directory
mkdir -p ${SAVE_DIR}

# Models to detect preference heads for
MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "meta-llama/Llama-2-7b-chat-hf"
  "Qwen/Qwen2-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

MODEL_NAMES=(
  "Meta-Llama-3-8B-Instruct"
  "Llama-2-7b-chat-hf"
  "Qwen2-7B-Instruct"
  "Mistral-7B-Instruct-v0.3"
)

# Tasks to detect
TASKS=(
  "LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7"
  "LongLaMP-2" "LongLaMP-3" "LongLaMP-4"
)

echo "========================================="
echo "Multi-Model Preference Head Detection"
echo "========================================="
echo "Models: ${#MODELS[@]}"
echo "Tasks: ${#TASKS[@]}"
echo "Samples per task: ${NUM_SAMPLES}"
echo "Top %: ${TOP_PERCENT}"
echo "Save dir: ${SAVE_DIR}"
echo "========================================="
echo ""

total_combinations=$((${#MODELS[@]} * ${#TASKS[@]}))
current=0

for model_idx in "${!MODELS[@]}"; do
  MODEL_PATH="${MODELS[$model_idx]}"
  MODEL_NAME="${MODEL_NAMES[$model_idx]}"
  
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "MODEL: ${MODEL_NAME}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  
  for task in "${TASKS[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_combinations] Detecting ${task} for ${MODEL_NAME}..."
    
    python preference_head_detection.py \
      --model_path "${MODEL_PATH}" \
      --task "${task}" \
      --split dev \
      --num_samples ${NUM_SAMPLES} \
      --top_percent ${TOP_PERCENT} \
      --save_dir ${SAVE_DIR}
    
    echo ""
  done
done

echo "========================================="
echo "ALL DETECTIONS COMPLETE!"
echo "========================================="
echo ""
echo "Results saved:"
ls -lh ${SAVE_DIR}/*_top_heads.json | wc -l
echo " preference head files created"
echo ""
echo "Summary by model:"
for model_name in "${MODEL_NAMES[@]}"; do
  # Clean model name for file matching
  model_clean=$(echo $model_name | tr '/' '_' | tr '-' '_')
  echo ""
  echo "  ${model_name}:"
  for task in "${TASKS[@]}"; do
    task_clean=$(echo $task | tr '-' '_')
    file="${SAVE_DIR}/${model_clean}_${task_clean}_top_heads.json"
    if [ -f "$file" ]; then
      num_heads=$(python3 -c "import json; print(json.load(open('$file'))['num_heads_selected'])" 2>/dev/null || echo "?")
      echo "    ✅ $task: $num_heads heads"
    else
      echo "    ❌ $task: Not found"
    fi
  done
done
echo ""
echo "========================================="
