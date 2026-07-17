#!/bin/bash
#SBATCH --job-name=dps_tune_l2l4      # Job name
#SBATCH --output=dps_tune_l2l4_%j.out   # Output file (%j = job ID)
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --gpus-per-node=1          # Request 1 GPU
#SBATCH --time=24:00:00            # Max runtime (24 hours)
#SBATCH --mail-type=ALL

# DPS tuning sweep for LaMP-2 and LaMP-4 (LLaMA3-8B-Instruct)

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories (same as decore)
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "========================================="
echo "DPS Tuning - LaMP-2"
echo "========================================="
python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=10,20,40,60 \
  decoder.configs.alpha_cap=null,3.0,5.0 \
  decoder.configs.scale_alpha=false,true \
  --multirun

echo "========================================="
echo "DPS Tuning - LaMP-4"
echo "========================================="
python scripts/main.py experiment=lamp_4/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=10,20,40,60 \
  decoder.configs.alpha_cap=null,3.0,5.0 \
  decoder.configs.scale_alpha=false,true \
  --multirun

echo "========================================="
echo "Evaluation"
echo "========================================="
python evaluate_predictions.py --latest --outputs-dir multirun
