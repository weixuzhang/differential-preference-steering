#!/bin/bash

# Create ablation study configs for different numbers of preference heads
# Testing: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 heads
# On all tasks, but only LLaMA-3-8B model

MODEL="llama3_8b_instruct"
TASKS=("lamp_1" "lamp_2" "lamp_3" "lamp_4" "lamp_5" "lamp_7" "longlamp_1" "longlamp_2" "longlamp_3" "longlamp_4")

declare -A TASK_NAMES
TASK_NAMES["lamp_1"]="LaMP-1"
TASK_NAMES["lamp_2"]="LaMP-2"
TASK_NAMES["lamp_3"]="LaMP-3"
TASK_NAMES["lamp_4"]="LaMP-4"
TASK_NAMES["lamp_5"]="LaMP-5"
TASK_NAMES["lamp_7"]="LaMP-7"
TASK_NAMES["longlamp_1"]="LongLaMP-1"
TASK_NAMES["longlamp_2"]="LongLaMP-2"
TASK_NAMES["longlamp_3"]="LongLaMP-3"
TASK_NAMES["longlamp_4"]="LongLaMP-4"

declare -A MAX_TOKENS
MAX_TOKENS["lamp_1"]=32
MAX_TOKENS["lamp_2"]=32
MAX_TOKENS["lamp_3"]=128
MAX_TOKENS["lamp_4"]=64
MAX_TOKENS["lamp_5"]=32
MAX_TOKENS["lamp_7"]=64
MAX_TOKENS["longlamp_1"]=256
MAX_TOKENS["longlamp_2"]=256
MAX_TOKENS["longlamp_3"]=256
MAX_TOKENS["longlamp_4"]=256

# Number of heads to test
NUM_HEADS=(10 20 30 40 50 60 70 80 90 100)

created=0
skipped=0

echo "Creating ablation study configs..."
echo "Model: ${MODEL}"
echo "Tasks: ${#TASKS[@]}"
echo "Head counts: ${NUM_HEADS[@]}"
echo ""

for task in "${TASKS[@]}"; do
  for num_head in "${NUM_HEADS[@]}"; do
    # Create directory for ablation configs
    mkdir -p "configs/experiment/${task}/dps_ablation"
    
    config_file="configs/experiment/${task}/dps_ablation/${MODEL}_heads${num_head}.yaml"
    
    if [ -f "$config_file" ]; then
      echo "Skipping $config_file (exists)"
      skipped=$((skipped + 1))
      continue
    fi
    
    echo "Creating $config_file"
    created=$((created + 1))
    
    cat > "$config_file" << INNEREOF
# @package _global_
defaults:
  - override /model: ${MODEL}
  - override /data: ${task}
  - override /decoder: dps

# DPS Ablation Study: ${num_head} preference heads for ${TASK_NAMES[$task]}
decoder:
  configs:
    task: ${TASK_NAMES[$task]}
    num_preference_heads: ${num_head}
    post_softmax: True

model:
  configs:
    max_new_tokens: ${MAX_TOKENS[$task]}

# Ablation study identifier
ablation:
  study: num_preference_heads
  value: ${num_head}
INNEREOF
  done
done

echo ""
echo "========================================="
echo "Summary:"
echo "  Created: $created configs"
echo "  Skipped: $skipped configs"
echo "  Tasks:   ${#TASKS[@]}"
echo "  Head counts: ${#NUM_HEADS[@]}"
echo "  Total expected: $((${#TASKS[@]} * ${#NUM_HEADS[@]}))"
echo "========================================="
