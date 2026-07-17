#!/bin/bash
#SBATCH --job-name=dps_all_tasks
#SBATCH --output=dps_all_tasks_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL

# DPS (Differential Preference Steering) experiments on ALL LaMP datasets
# Runs DPS with detected preference heads for LaMP-1, 2, 3, 4

source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories (same as decore)
hf_cache="/scratch/weixuz/dps/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "========================================="
echo "DPS Experiments on ALL LaMP & LongLaMP Tasks"
echo "========================================="
echo ""

# Check which tasks have preference heads detected
echo "Checking for detected preference heads..."
PREF_HEAD_DIR="/scratch/weixuz/dps/preference_head/preference_scores"

declare -A TASKS_READY
declare -A LONGLAMP_READY

# Check LaMP tasks
for task_num in 1 2 3 4 5 7; do
  task_name="LaMP-${task_num}"
  task_file="${PREF_HEAD_DIR}/Meta-Llama-3-8B-Instruct_LaMP_${task_num}_top_heads.json"
  if [ -f "$task_file" ]; then
    TASKS_READY[$task_num]=1
    echo "  ✅ ${task_name}: Preference heads found"
  else
    TASKS_READY[$task_num]=0
    echo "  ❌ ${task_name}: Preference heads NOT found (skipping)"
  fi
done

# Check LongLaMP tasks
for task_num in 1 2 3 4; do
  task_name="LongLaMP-${task_num}"
  task_file="${PREF_HEAD_DIR}/Meta-Llama-3-8B-Instruct_LongLaMP_${task_num}_top_heads.json"
  if [ -f "$task_file" ]; then
    LONGLAMP_READY[$task_num]=1
    echo "  ✅ ${task_name}: Preference heads found"
  else
    LONGLAMP_READY[$task_num]=0
    echo "  ❌ ${task_name}: Preference heads NOT found (skipping)"
  fi
done
echo ""

# LaMP-1: Citation Identification
if [ ${TASKS_READY[1]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 1/4: LaMP-1 (Citation Identification)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=lamp_1/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LaMP-2: Movie Tagging
if [ ${TASKS_READY[2]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 2/4: LaMP-2 (Movie Tagging)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=lamp_2/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LaMP-3: Score Prediction
if [ ${TASKS_READY[3]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 3/4: LaMP-3 (Score Prediction)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=lamp_3/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LaMP-4: News Headline Generation
if [ ${TASKS_READY[4]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 4/6: LaMP-4 (News Headline Generation)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=lamp_4/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LaMP-5: Scholarly Title Generation
if [ ${TASKS_READY[5]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 5/6: LaMP-5 (Scholarly Title Generation)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=lamp_5/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LaMP-7: Tweet Paraphrasing
if [ ${TASKS_READY[7]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 6/10: LaMP-7 (Tweet Paraphrasing)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=lamp_7/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LongLaMP-1: Email Generation
if [ ${LONGLAMP_READY[1]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 7/10: LongLaMP-1 (Email Generation)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=longlamp_1/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LongLaMP-2: Abstract Generation
if [ ${LONGLAMP_READY[2]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 8/10: LongLaMP-2 (Abstract Generation)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=longlamp_2/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LongLaMP-3: Topic Writing
if [ ${LONGLAMP_READY[3]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 9/10: LongLaMP-3 (Topic Writing)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=longlamp_3/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

# LongLaMP-4: Product Review Generation
if [ ${LONGLAMP_READY[4]} -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Task 10/10: LongLaMP-4 (Product Review Generation)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python scripts/main.py \
    experiment=longlamp_4/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
fi

echo "========================================="
echo "DPS experiments complete!"
echo "========================================="
echo ""

# Evaluate predictions
echo "Evaluating all predictions..."
python evaluate_predictions.py --latest

echo ""
echo "========================================="
echo "All tasks completed!"
echo "Check outputs/ directory for results"
echo "========================================="

