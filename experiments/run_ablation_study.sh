#!/bin/bash
#SBATCH --job-name=dps_ablation
#SBATCH --output=dps_ablation_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL

# DPS Ablation Study: Number of Preference Heads
# Tests: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 heads
# Model: LLaMA-3-8B-Instruct
# Tasks: All 10 tasks (LaMP + LongLaMP)

source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories
hf_cache="/scratch/weixuz/dps/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         DPS ABLATION STUDY: Number of Preference Heads          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

MODEL="llama3_8b_instruct"
TASKS=("lamp_1" "lamp_2" "lamp_3" "lamp_4" "lamp_5" "lamp_7" "longlamp_1" "longlamp_2" "longlamp_3" "longlamp_4")
TASK_DISPLAY=("LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7" "LongLaMP-1" "LongLaMP-2" "LongLaMP-3" "LongLaMP-4")
NUM_HEADS=(10 20 30 40 50 60 70 80 90 100)

PREF_HEAD_DIR="/scratch/weixuz/dps/preference_head/preference_scores"

echo "Model: ${MODEL}"
echo "Tasks: ${#TASKS[@]}"
echo "Head counts to test: ${NUM_HEADS[@]}"
echo "Total experiments: $((${#TASKS[@]} * ${#NUM_HEADS[@]}))"
echo ""

# Check if preference heads are detected
echo "Checking for detected preference heads..."
model_clean="Meta_Llama_3_8B_Instruct"

all_heads_detected=true
for idx in "${!TASKS[@]}"; do
  task_display="${TASK_DISPLAY[$idx]}"
  task_clean=$(echo $task_display | tr '-' '_')
  pref_file="${PREF_HEAD_DIR}/${model_clean}_${task_clean}_top_heads.json"
  
  if [ ! -f "$pref_file" ]; then
    echo "  ❌ ${task_display}: Preference heads not detected"
    all_heads_detected=false
  else
    echo "  ✅ ${task_display}: Preference heads ready"
  fi
done

if [ "$all_heads_detected" = false ]; then
  echo ""
  echo "⚠️  Warning: Some preference heads are missing!"
  echo "Please run preference head detection first:"
  echo "  cd /scratch/weixuz/dps/preference_head"
  echo "  sbatch run_detection_all_tasks.sh"
  echo ""
  exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting ablation experiments..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

total_experiments=0
completed_experiments=0
failed_experiments=0

for idx in "${!TASKS[@]}"; do
  task="${TASKS[$idx]}"
  task_display="${TASK_DISPLAY[$idx]}"
  
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Task: ${task_display}"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
  
  for num_head in "${NUM_HEADS[@]}"; do
    total_experiments=$((total_experiments + 1))
    echo "  [$total_experiments/$((${#TASKS[@]} * ${#NUM_HEADS[@]}))] Testing ${num_head} heads..."
    
    # Run experiment
    python scripts/main.py \
      experiment=${task}/dps_ablation/${MODEL}_heads${num_head} \
      2>&1 | tail -n 5
    
    if [ $? -eq 0 ]; then
      completed_experiments=$((completed_experiments + 1))
      echo "  ✅ Completed ${num_head} heads"
    else
      failed_experiments=$((failed_experiments + 1))
      echo "  ❌ Failed ${num_head} heads"
    fi
  done
  
  echo ""
  echo "  Progress: ${completed_experiments}/${total_experiments} completed"
  echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                 ABLATION STUDY COMPLETE!                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Total experiments:     ${total_experiments}"
echo "  ✅ Completed:          ${completed_experiments}"
echo "  ❌ Failed:             ${failed_experiments}"
echo ""
echo "  Tasks tested:          ${#TASKS[@]}"
echo "  Head counts tested:    ${#NUM_HEADS[@]}"
echo "  Model:                 ${MODEL}"
echo ""

# Create ablation analysis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running ablation analysis..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/analyze_ablation.py

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ALL TASKS COMPLETED!                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved in: outputs/"
echo "Ablation analysis: outputs/ablation_analysis.json"
echo "Ablation plots: outputs/ablation_plots/"
echo ""

