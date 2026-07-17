#!/bin/bash
#SBATCH --job-name=dps_lamp4      # Job name
#SBATCH --output=dps_lamp4.out   # Output file (%j = job ID)
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --gpus-per-node=1          # Request 1 GPU
#SBATCH --time=06:00:00            # Max runtime (6 hours)
#SBATCH --mail-type=ALL

# DPS (Differential Preference Steering) experiments on LaMP datasets
# Uses detected preference heads to personalize model generation

source /scratch/weixuz/envs/decore/bin/activate

# Set cache directories (same as decore)
hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

export WANDB_DISABLED=true
export HF_OFFLINE=true

echo "========================================="
echo "DPS Experiments on LaMP"
echo "========================================="
echo ""

# LaMP-1: Citation Identification (40 preference heads)
# echo "Running DPS on LaMP-1..."
# python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct decoder.configs.num_preference_heads=40
# echo ""

# LaMP-2: Movie Tagging (uncomment when you have preference heads for LaMP-2)
# echo "Running DPS on LaMP-2..."
# python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct decoder.configs.num_preference_heads=40
# echo ""

# LaMP-3: Score Prediction (uncomment when you have preference heads for LaMP-3)
# echo "Running DPS on LaMP-3..."
# python scripts/main.py experiment=lamp_3/dps/llama3_8b_instruct decoder.configs.num_preference_heads=40
# echo ""

# LaMP-4: News Headline Generation (uncomment when you have preference heads for LaMP-4)
# echo "Running DPS on LaMP-4..."
python scripts/main.py experiment=lamp_4/dps/llama3_8b_instruct decoder.configs.num_preference_heads=40
# echo ""

# LaMP-5: Scholarly Title Generation    (uncomment when you have preference heads for LaMP-5)
echo "========================================="
python scripts/main.py experiment=lamp_5/dps/llama3_8b_instruct decoder.configs.num_preference_heads=40
echo ""

# LaMP-7: Tweet Paraphrasing    (uncomment when you have preference heads for LaMP-7)
echo "========================================="
python scripts/main.py experiment=lamp_7/dps/llama3_8b_instruct decoder.configs.num_preference_heads=40
echo ""

# Evaluate predictions
echo "========================================="
echo "DPS experiments complete!"
echo "========================================="
python evaluate_predictions.py --latest

