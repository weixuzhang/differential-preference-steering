#!/bin/bash
#SBATCH --job-name=dps_multi_model
#SBATCH --output=dps_multi_model_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL

# DPS experiments on all models and all tasks
# Runs DPS with detected preference heads for multiple models

source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories
hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "========================================="
echo "DPS Multi-Model Experiments"
echo "========================================="
echo ""

# Model configurations
declare -A MODEL_CONFIGS
MODEL_CONFIGS["llama3_8b_instruct"]="Meta-Llama-3-8B-Instruct"
MODEL_CONFIGS["llama2_7b_instruct"]="Llama-2-7b-chat-hf"
MODEL_CONFIGS["qwen2_7b_instruct"]="Qwen2-7B-Instruct"
MODEL_CONFIGS["mistral_7b_instruct"]="Mistral-7B-Instruct-v0.3"

# Tasks
TASKS=("lamp_1" "lamp_2" "lamp_3" "lamp_4" "lamp_5" "lamp_7" "longlamp_1" "longlamp_2" "longlamp_3" "longlamp_4")
TASK_DISPLAY=("LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7" "LongLaMP-1" "LongLaMP-2" "LongLaMP-3" "LongLaMP-4")

PREF_HEAD_DIR="results/preference_head/preference_scores"

# Check which model-task combinations have preference heads
echo "Checking for detected preference heads..."
echo ""

declare -A READY

for model_key in "${!MODEL_CONFIGS[@]}"; do
  model_name="${MODEL_CONFIGS[$model_key]}"
  model_clean=$(echo $model_name | tr '/' '_' | tr '-' '_')
  
  echo "Model: ${model_name}"
  
  for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    task_display="${TASK_DISPLAY[$idx]}"
    task_clean=$(echo $task_display | tr '-' '_')
    
    pref_file="${PREF_HEAD_DIR}/${model_clean}_${task_clean}_top_heads.json"
    
    if [ -f "$pref_file" ]; then
      READY["${model_key}_${task}"]=1
      echo "  ✅ ${task_display}: Ready"
    else
      READY["${model_key}_${task}"]=0
      echo "  ❌ ${task_display}: Missing preference heads"
    fi
  done
  echo ""
done

# Run experiments
total_experiments=0
completed_experiments=0

for model_key in "${!MODEL_CONFIGS[@]}"; do
  model_name="${MODEL_CONFIGS[$model_key]}"
  
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "MODEL: ${model_name}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  
  for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    task_display="${TASK_DISPLAY[$idx]}"
    
    if [ "${READY[${model_key}_${task}]}" == "1" ]; then
      total_experiments=$((total_experiments + 1))
      echo "Running ${task_display} with ${model_name}..."
      
      python scripts/main.py \
        experiment=${task}/dps/${model_key} \
        decoder.configs.num_preference_heads=40
      
      if [ $? -eq 0 ]; then
        completed_experiments=$((completed_experiments + 1))
        echo "✅ Completed ${task_display} with ${model_name}"
      else
        echo "❌ Failed ${task_display} with ${model_name}"
      fi
      echo ""
    else
      echo "⏭️  Skipping ${task_display} with ${model_name} (no preference heads)"
      echo ""
    fi
  done
done

echo "========================================="
echo "DPS Multi-Model Experiments Complete!"
echo "========================================="
echo ""
echo "Completed: ${completed_experiments}/${total_experiments} experiments"
echo ""

# Evaluate predictions
echo "Evaluating all predictions..."
python evaluate_predictions.py --latest

echo ""
echo "========================================="
echo "All tasks completed!"
echo "Check outputs/ directory for results"
echo "========================================="
