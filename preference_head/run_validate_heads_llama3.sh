#!/bin/bash
#SBATCH --job-name=validate_heads      # Job name
#SBATCH --output=validate_heads_%j.out   # Output file (%j = job ID)
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --gpus-per-node=1          # Request 1 GPU
#SBATCH --time=06:00:00            # Max runtime (6 hours)
#SBATCH --mail-type=ALL

# Validate preference heads and generate visualizations (LLaMA3-8B-Instruct)

source /scratch/weixuz/envs/decore/bin/activate

# Cache directories (same as decore)
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_OFFLINE=true

MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
PREF_DIR="/scratch/weixuz/preference_head/preference_scores"
OUT_DIR="/scratch/weixuz/preference_head/visualizations"

mkdir -p ${OUT_DIR}

echo "========================================="
echo "Validate Preference Heads"
echo "========================================="

python validate_preference_heads.py \
  --model_path ${MODEL_PATH} \
  --preference_heads_file ${PREF_DIR}/Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json \
  --task LaMP-1 \
  --num_samples 100

python validate_preference_heads.py \
  --model_path ${MODEL_PATH} \
  --preference_heads_file ${PREF_DIR}/Meta-Llama-3-8B-Instruct_LaMP_2_top_heads.json \
  --task LaMP-2 \
  --num_samples 100

python validate_preference_heads.py \
  --model_path ${MODEL_PATH} \
  --preference_heads_file ${PREF_DIR}/Meta-Llama-3-8B-Instruct_LaMP_4_top_heads.json \
  --task LaMP-4 \
  --num_samples 100

echo "========================================="
echo "Visualizations"
echo "========================================="

python visualize_heads.py \
  --pcs_file ${PREF_DIR}/Meta-Llama-3-8B-Instruct_LaMP_2_pcs.json \
  --ranked_file ${PREF_DIR}/Meta-Llama-3-8B-Instruct_LaMP_2_ranked.json \
  --output_dir ${OUT_DIR}/LaMP-2

python visualize_heads.py \
  --pcs_file ${PREF_DIR}/Meta-Llama-3-8B-Instruct_LaMP_4_pcs.json \
  --ranked_file ${PREF_DIR}/Meta-Llama-3-8B-Instruct_LaMP_4_ranked.json \
  --output_dir ${OUT_DIR}/LaMP-4
