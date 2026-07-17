hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
# Set HF_TOKEN in your environment (do not hardcode tokens here)

export WANDB_DISABLED=true
# Set to false to download models, true to use cached only
export HF_OFFLINE=false
export HF_HUB_OFFLINE=0

python scripts/main.py experiment=lamp_5/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50

