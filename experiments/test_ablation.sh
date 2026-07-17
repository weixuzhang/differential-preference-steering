#!/bin/bash

# Quick test for ablation study
# Tests a few head counts on one task with small sample

source /scratch/weixuz/envs/decore/bin/activate

hf_cache="/scratch/weixuz/dps/.cache/huggingface"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "Testing ablation study configs..."
echo ""

# Test with LaMP-1 and a few head counts
TASK="lamp_1"
MODEL="llama3_8b_instruct"

for num_heads in 10 40 100; do
  echo "Testing ${num_heads} heads on LaMP-1..."
  python scripts/main.py \
    experiment=${TASK}/dps_ablation/${MODEL}_heads${num_heads} \
    data.num_samples=5 \
    debug=true
  echo ""
done

echo "✅ Ablation test complete!"

