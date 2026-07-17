#!/bin/bash

# Test DPS (Differential Preference Steering) integration
# Quick test with small number of samples

echo "========================================="
echo "Testing DPS Integration"
echo "========================================="
echo ""

# Activate environment
source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories (same as decore)
hf_cache="/scratch/weixuz/dps/.cache/huggingface"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_OFFLINE=true
export WANDB_DISABLED=true

# Test on LaMP-1 with DPS (only 5 samples for quick test)
echo "Testing DPS on LaMP-1 (5 samples)..."
python scripts/main.py \
  experiment=lamp_1/dps/llama3_8b_instruct \
  data.num_samples=5 \
  debug=true

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DPS test successful!"
    echo ""
else
    echo ""
    echo "❌ DPS test failed!"
    echo ""
    exit 1
fi

