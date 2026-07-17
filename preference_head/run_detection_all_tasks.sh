#!/bin/bash
#SBATCH --job-name=pref_head_all
#SBATCH --output=preference_head_detection_all_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL

# Preference Head Detection Script for ALL LaMP & LongLaMP Tasks
# Detects preference heads for LLaMA3-8B-Instruct on LaMP-1,2,3,4,5,7 and LongLaMP-2,3,4

# Activate environment
source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories (same as decore)
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_OFFLINE=true

# Model path (will use cached version)
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

# Detection parameters
NUM_SAMPLES=400  # Number of samples for detection
TOP_PERCENT=0.04  # Select top 4% of heads (40 heads)
SAVE_DIR="/scratch/weixuz/preference_head/preference_scores"

# Create save directory
mkdir -p ${SAVE_DIR}

echo "========================================="
echo "Preference Head Detection - ALL TASKS"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Samples: ${NUM_SAMPLES}"
echo "Top %: ${TOP_PERCENT}"
echo "Save dir: ${SAVE_DIR}"
echo "========================================="
echo ""

# LaMP-1: Citation Identification
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 1/9: LaMP-1 (Citation Identification)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LaMP-1 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LaMP-2: Movie Tagging
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 2/9: LaMP-2 (Movie Tagging)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LaMP-2 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LaMP-3: Score Prediction
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 3/9: LaMP-3 (Score Prediction)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LaMP-3 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LaMP-4: News Headline Generation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 4/9: LaMP-4 (News Headline Generation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LaMP-4 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LaMP-5: Scholarly Title Generation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 5/9: LaMP-5 (Scholarly Title Generation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LaMP-5 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LaMP-7: Tweet Paraphrasing
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 6/9: LaMP-7 (Tweet Paraphrasing)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LaMP-7 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LongLaMP-2: Abstract Generation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 7/9: LongLaMP-2 (Abstract Generation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LongLaMP-2 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LongLaMP-3: Topic Writing
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 8/9: LongLaMP-3 (Topic Writing)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LongLaMP-3 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""

# LongLaMP-4: Product Review Generation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task 9/9: LongLaMP-4 (Product Review Generation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python preference_head_detection.py \
  --model_path ${MODEL_PATH} \
  --task LongLaMP-4 \
  --split dev \
  --num_samples ${NUM_SAMPLES} \
  --top_percent ${TOP_PERCENT} \
  --save_dir ${SAVE_DIR}

echo ""
echo "========================================="
echo "ALL TASKS COMPLETE!"
echo "========================================="
echo ""
echo "Results saved:"
ls -lh ${SAVE_DIR}/*_top_heads.json
echo ""
echo "Summary:"
for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4 LaMP-5 LaMP-7 LongLaMP-2 LongLaMP-3 LongLaMP-4; do
  task_clean=$(echo $task | tr '-' '_')
  file="${SAVE_DIR}/Meta-Llama-3-8B-Instruct_${task_clean}_top_heads.json"
  if [ -f "$file" ]; then
    num_heads=$(python3 -c "import json; print(json.load(open('$file'))['num_heads_selected'])" 2>/dev/null || echo "?")
    echo "  ✅ $task: $num_heads preference heads detected"
  else
    echo "  ❌ $task: Not found"
  fi
done
echo ""
echo "========================================="
