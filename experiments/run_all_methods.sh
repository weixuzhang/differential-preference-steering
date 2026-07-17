#!/bin/bash
#SBATCH --job-name=all_methods
#SBATCH --output=all_methods_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL

# Comprehensive experiment runner for ALL methods
# Runs: Baseline, DPS, DeCoRe, CAD, CD on all models and tasks

source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories
hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           COMPREHENSIVE MULTI-METHOD EXPERIMENTS                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
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

# Methods to run
METHODS=("baseline" "context_aware_decoding" "contrastive_decoding" "decore_entropy" "dps")
METHOD_DISPLAY=("Baseline" "CAD" "CD" "DeCoRe" "DPS")

PREF_HEAD_DIR="results/preference_head/preference_scores"

# Statistics
total_experiments=0
completed_experiments=0
skipped_experiments=0
failed_experiments=0

# Check which experiments can run
echo "Checking experiment prerequisites..."
echo ""

declare -A CAN_RUN

for model_key in "${!MODEL_CONFIGS[@]}"; do
  model_name="${MODEL_CONFIGS[$model_key]}"
  
  for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    task_display="${TASK_DISPLAY[$idx]}"
    
    for method_idx in "${!METHODS[@]}"; do
      method="${METHODS[$method_idx]}"
      method_name="${METHOD_DISPLAY[$method_idx]}"
      
      key="${model_key}_${task}_${method}"
      
      # Check if DPS has preference heads
      if [ "$method" == "dps" ]; then
        model_clean=$(echo $model_name | tr '/' '_' | tr '-' '_')
        task_clean=$(echo $task_display | tr '-' '_')
        pref_file="${PREF_HEAD_DIR}/${model_clean}_${task_clean}_top_heads.json"
        
        if [ -f "$pref_file" ]; then
          CAN_RUN[$key]=1
        else
          CAN_RUN[$key]=0
        fi
      else
        # Other methods can always run
        CAN_RUN[$key]=1
      fi
    done
  done
done

# Run experiments
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting experiments..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for model_key in "${!MODEL_CONFIGS[@]}"; do
  model_name="${MODEL_CONFIGS[$model_key]}"
  
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  MODEL: ${model_name}"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
  
  for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    task_display="${TASK_DISPLAY[$idx]}"
    
    echo "  ┌─────────────────────────────────────────────────────────────"
    echo "  │ Task: ${task_display}"
    echo "  └─────────────────────────────────────────────────────────────"
    
    for method_idx in "${!METHODS[@]}"; do
      method="${METHODS[$method_idx]}"
      method_name="${METHOD_DISPLAY[$method_idx]}"
      
      key="${model_key}_${task}_${method}"
      
      if [ "${CAN_RUN[$key]}" == "1" ]; then
        total_experiments=$((total_experiments + 1))
        echo "    ▶ Running ${method_name}..."
        
        # Run experiment
        python scripts/main.py \
          experiment=${task}/${method}/${model_key} \
          2>&1 | grep -E "(Accuracy|F1|MAE|RMSE|ROUGE|METEOR|Error)" || true
        
        if [ $? -eq 0 ]; then
          completed_experiments=$((completed_experiments + 1))
          echo "    ✅ ${method_name} completed"
        else
          failed_experiments=$((failed_experiments + 1))
          echo "    ❌ ${method_name} failed"
        fi
      else
        skipped_experiments=$((skipped_experiments + 1))
        echo "    ⏭️  ${method_name} skipped (no preference heads)"
      fi
    done
    echo ""
  done
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT SUMMARY                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Total experiments:     $((total_experiments + skipped_experiments))"
echo "  ✅ Completed:          ${completed_experiments}"
echo "  ❌ Failed:             ${failed_experiments}"
echo "  ⏭️  Skipped:            ${skipped_experiments}"
echo ""
echo "  Methods tested:        ${#METHODS[@]} (Baseline, CAD, CD, DeCoRe, DPS)"
echo "  Models tested:         ${#MODEL_CONFIGS[@]}"
echo "  Tasks tested:          ${#TASKS[@]}"
echo ""

# Evaluate predictions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluating all predictions..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python evaluate_predictions.py --latest

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ALL EXPERIMENTS COMPLETE!                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved in: outputs/"
echo "Evaluation summary: outputs/evaluation_summary.json"
echo ""
