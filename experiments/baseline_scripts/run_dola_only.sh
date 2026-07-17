#!/bin/bash

set -euo pipefail

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

hf_cache="${ROOT}/dps/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true
export HF_DATASETS_OFFLINE=true
export TRANSFORMERS_OFFLINE=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

TASK="${1:-lamp_1}"
MODEL="${2:-llama3_8b_instruct}"

cd "${ROOT}/decore"

echo "Running DoLa baseline: ${TASK} | ${MODEL}"
python scripts/main.py experiment="${TASK}/baseline/${MODEL}" decoder=dola
