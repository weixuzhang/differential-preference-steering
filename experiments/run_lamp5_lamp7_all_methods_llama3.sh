#!/bin/bash
#SBATCH --job-name=lamp5_7_all      # Job name
#SBATCH --output=lamp5_7_all_%j.out   # Output file (%j = job ID)
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --gpus-per-node=1          # Request 1 GPU
#SBATCH --time=06:00:00            # Max runtime (6 hours)
#SBATCH --mail-type=ALL

# Baseline + DeCoRe + DPS on LaMP-5 and LaMP-7 (LLaMA3-8B-Instruct)

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories (same as decore)
hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "========================================="
echo "LaMP-5/7 Experiments - All Methods"
echo "========================================="

tasks=(lamp_5 lamp_7)
for task in "${tasks[@]}"; do
  echo "-----------------------------------------"
  echo "Task: ${task}"
  echo "-----------------------------------------"
  python scripts/main.py experiment=${task}/baseline/llama3_8b_instruct
  python scripts/main.py experiment=${task}/decore_entropy/llama3_8b_instruct \
    decoder.configs.num_retrieval_heads=50
  python scripts/main.py experiment=${task}/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=40
  echo ""
done

echo "========================================="
echo "Evaluation"
echo "========================================="
python evaluate_predictions.py --latest
