#!/bin/bash

# Quick test for CAD and CD methods

source /scratch/weixuz/envs/decore/bin/activate

hf_cache=".cache/huggingface"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "Testing Context-Aware Decoding (CAD)..."
python scripts/main.py \
  experiment=lamp_1/context_aware_decoding/llama3_8b_instruct \
  data.num_samples=5 \
  debug=true

echo ""
echo "Testing Contrastive Decoding (CD)..."
python scripts/main.py \
  experiment=lamp_1/contrastive_decoding/llama3_8b_instruct \
  data.num_samples=5 \
  debug=true

echo ""
echo "✅ Tests complete!"
