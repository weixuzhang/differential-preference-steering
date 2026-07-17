#!/bin/bash
# download lamp dataset
# pip install gdown
# gdown --id 1SgomdWGZo-c74IMoR23vcKXKXEqio350

hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
# Set HF_TOKEN in your environment (do not hardcode tokens here)

export WANDB_DISABLED=true
export HF_OFFLINE=false
export HF_HUB_OFFLINE=0

# python scripts/main.py experiment=lamp_1/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
# python scripts/main.py experiment=lamp_2/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
# python scripts/main.py experiment=lamp_3/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
# python scripts/main.py experiment=lamp_4/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50

# Baseline experiments (no DeCoRe, just standard model)
# python scripts/main.py experiment=lamp_1/baseline/llama3_8b_instruct
# python scripts/main.py experiment=lamp_2/baseline/llama3_8b_instruct
# python scripts/main.py experiment=lamp_3/baseline/llama3_8b_instruct
# python scripts/main.py experiment=lamp_4/baseline/llama3_8b_instruct

# BM25 (fast, token-based)
python scripts/main.py  experiment=lamp_1/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50

# # Contriever (better quality, semantic)
# python scripts/main.py  experiment=lamp_3/decore_entropy/llama3_8b_instruct \
#   data.retriever=contriever

# # Run all 4 LAMP tasks with DeCoRe + RAG
# for task in 1 2 3 4; do
#   python scripts/main.py  experiment=lamp_${task}/decore_entropy/llama3_8b_instruct \
#     decoder.configs.num_retrieval_heads=50
# done

# # 3 profiles (conservative)
# python scripts/main.py  experiment=lamp_3/decore_entropy/llama3_8b_instruct \
#   data.num_retrieve=3

# # 10 profiles (more context)
# python scripts/main.py  experiment=lamp_3/decore_entropy/llama3_8b_instruct \
#   data.num_retrieve=10