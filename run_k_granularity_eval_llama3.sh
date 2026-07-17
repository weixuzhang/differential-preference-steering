#!/bin/bash
#SBATCH --job-name=k_eval      # Job name
#SBATCH --output=k_eval_%j.out   # Output file (%j = job ID)
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --gpus-per-node=1          # Request 1 GPU
#SBATCH --time=24:00:00            # Max runtime (24 hours)
#SBATCH --mail-type=ALL

# k-granularity DPS evaluation (LLaMA3-8B-Instruct)
# Uses preference head files from preference_scores_k.

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories (same as decore)
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

ks=(1 2 5 10 25 50 100 200 2500)
tasks=(lamp_1 lamp_3)

for task in "${tasks[@]}"; do
  for k in "${ks[@]}"; do
    heads_dir="/scratch/weixuz/preference_head/preference_scores_k/LaMP-${task#lamp_}/k${k}"
    echo "-----------------------------------------"
    echo "Task: ${task} | k=${k}"
    echo "Preference heads dir: ${heads_dir}"
    echo "-----------------------------------------"
    python scripts/main.py \
      experiment=${task}/dps/llama3_8b_instruct \
      decoder.configs.preference_heads_dir=${heads_dir} \
      decoder.configs.num_preference_heads=40
    echo ""
  done

done

echo "========================================="
echo "Evaluation"
echo "========================================="
python evaluate_predictions.py --latest
