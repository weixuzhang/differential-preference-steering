#!/bin/bash
set -euo pipefail

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

# Cache directories
hf_cache="$(pwd)/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true
export HF_DATASETS_OFFLINE=true
export TRANSFORMERS_OFFLINE=true
export HF_DATASETS_SINGLE_THREAD=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "${ROOT}/decore"

# ---- Quick config ----
TASK="LaMP-1"
TASK_DECODER="LAMP_1"
TARGET_GROUP=100
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"
NUM_SAMPLES=50
TOP_PERCENT_40=$(python -c "print(40/1024)")

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")

K=$(python "preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
if [ -z "${K}" ]; then
  echo "Failed to compute K for ${TASK}"
  exit 1
fi

cluster_dir="results/preference_head/cluster_runs/${task_slug}_k${K}"
emb_file="${cluster_dir}/embeddings_dev.npy"

ensure_clusters() {
  if [ ! -f "${cluster_dir}/clusters.json" ]; then
    python "preference_head/cluster_profiles.py" \
      --task "${TASK}" \
      --split dev \
      --k "${K}" \
      --output_dir "${cluster_dir}" \
      --save_embeddings \
      --embedding_model "${EMB_MODEL}"
  fi
  if [ ! -f "${emb_file}" ]; then
    python "preference_head/embed_profiles.py" \
      --task "${TASK}" \
      --split dev \
      --output_file "${emb_file}" \
      --meta_file "${cluster_dir}/embeddings_dev.json" \
      --embedding_model "${EMB_MODEL}"
  fi
}

detect_heads() {
  local head_dir="$1"
  local top_percent="$2"
  if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
    python "preference_head/detect_cluster_heads.py" \
      --cluster_file "${cluster_dir}/clusters.json" \
      --model_path "${MODEL_PATH}" \
      --task "${TASK}" \
      --split dev \
      --num_samples "${NUM_SAMPLES}" \
      --save_dir "${head_dir}" \
      --top_percent "${top_percent}" \
      --pcs_norm max \
      --pcs_power 1.0 \
      --cluster_start 0 \
      --cluster_end "$((K - 1))"
  fi
}

ensure_clusters

echo "========================================="
echo "Quick hparam runs | ${TASK} | k=${K} | samples=${NUM_SAMPLES}"
echo "Model: ${MODEL_NAME}"
echo "========================================="

# 1) Heads sweep (quick subset)
HEAD_COUNTS=(20 40)
for num_heads in "${HEAD_COUNTS[@]}"; do
  top_percent=$(python -c "print(${num_heads}/1024)")
  head_dir="results/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}_quick50_h${num_heads}"
  run_dir="$(pwd)/outputs/hparam/quick/heads/h${num_heads}"
  detect_heads "${head_dir}" "${top_percent}"
  mkdir -p "${run_dir}"
  python "$(pwd)/scripts/run_weighted_dps.py" \
    --task "${TASK_DECODER}" \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_type instruct \
    --cluster_file "${cluster_dir}/clusters.json" \
    --cluster_heads_dir "${head_dir}" \
    --embeddings_file "${emb_file}" \
    --routing soft \
    --temperature 1.0 \
    --num_samples "${NUM_SAMPLES}" \
    --run_dir "${run_dir}"
done

# 2) Gamma sweep (adaptive vs fixed)
head_dir_40="results/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}_quick50_h40"
detect_heads "${head_dir_40}" "${TOP_PERCENT_40}"
mkdir -p "$(pwd)/outputs/hparam/quick/gamma/adaptive"
python "$(pwd)/scripts/run_weighted_dps.py" \
  --task "${TASK_DECODER}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type instruct \
  --cluster_file "${cluster_dir}/clusters.json" \
  --cluster_heads_dir "${head_dir_40}" \
  --embeddings_file "${emb_file}" \
  --routing soft \
  --temperature 1.0 \
  --scale_alpha \
  --num_samples "${NUM_SAMPLES}" \
  --run_dir "$(pwd)/outputs/hparam/quick/gamma/adaptive"

FIXED_ALPHA=0.5
mkdir -p "$(pwd)/outputs/hparam/quick/gamma/fixed_alpha_${FIXED_ALPHA}"
python "$(pwd)/scripts/run_weighted_dps.py" \
  --task "${TASK_DECODER}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type instruct \
  --cluster_file "${cluster_dir}/clusters.json" \
  --cluster_heads_dir "${head_dir_40}" \
  --embeddings_file "${emb_file}" \
  --routing soft \
  --temperature 1.0 \
  --alpha "${FIXED_ALPHA}" \
  --num_samples "${NUM_SAMPLES}" \
  --run_dir "$(pwd)/outputs/hparam/quick/gamma/fixed_alpha_${FIXED_ALPHA}"

# 3) Group-size sweep (quick subset)
GROUP_SIZES=(50 100)
for target_group in "${GROUP_SIZES[@]}"; do
  K_g=$(python "preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${target_group}")
  if [ -z "${K_g}" ]; then
    echo "Failed to compute K for ${TASK} (target_group=${target_group})"
    continue
  fi
  cluster_dir_g="results/preference_head/cluster_runs/${task_slug}_k${K_g}"
  emb_file_g="${cluster_dir_g}/embeddings_dev.npy"
  head_dir_g="results/preference_head/cluster_heads/${task_slug}_k${K_g}_${model_slug}_quick50"
  run_dir_g="$(pwd)/outputs/hparam/quick/groupsize/g${target_group}"

  if [ ! -f "${cluster_dir_g}/clusters.json" ]; then
    python "preference_head/cluster_profiles.py" \
      --task "${TASK}" \
      --split dev \
      --k "${K_g}" \
      --output_dir "${cluster_dir_g}" \
      --save_embeddings \
      --embedding_model "${EMB_MODEL}"
  fi
  if [ ! -f "${emb_file_g}" ]; then
    python "preference_head/embed_profiles.py" \
      --task "${TASK}" \
      --split dev \
      --output_file "${emb_file_g}" \
      --meta_file "${cluster_dir_g}/embeddings_dev.json" \
      --embedding_model "${EMB_MODEL}"
  fi
  if [ ! -f "${head_dir_g}/cluster_00/head_weights.json" ]; then
    python "preference_head/detect_cluster_heads.py" \
      --cluster_file "${cluster_dir_g}/clusters.json" \
      --model_path "${MODEL_PATH}" \
      --task "${TASK}" \
      --split dev \
      --num_samples "${NUM_SAMPLES}" \
      --save_dir "${head_dir_g}" \
      --top_percent "${TOP_PERCENT_40}" \
      --pcs_norm max \
      --pcs_power 1.0 \
      --cluster_start 0 \
      --cluster_end "$((K_g - 1))"
  fi
  mkdir -p "${run_dir_g}"
  python "$(pwd)/scripts/run_weighted_dps.py" \
    --task "${TASK_DECODER}" \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_type instruct \
    --cluster_file "${cluster_dir_g}/clusters.json" \
    --cluster_heads_dir "${head_dir_g}" \
    --embeddings_file "${emb_file_g}" \
    --routing soft \
    --temperature 1.0 \
    --num_samples "${NUM_SAMPLES}" \
    --run_dir "${run_dir_g}"
done

# 4) Ablation (random heads + random masks)
head_dir_ablate="results/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}_quick50_h40"
detect_heads "${head_dir_ablate}" "${TOP_PERCENT_40}"
agg_heads="${head_dir_ablate}/aggregate_top_heads.json"
random_head_dir="${head_dir_ablate}_random"
random_mask_dir="${head_dir_ablate}_random_mask"

python "preference_head/aggregate_cluster_heads.py" \
  --cluster_heads_dir "${head_dir_ablate}" \
  --top_percent "${TOP_PERCENT_40}" \
  --output_file "${agg_heads}"

python "preference_head/validate_preference_heads.py" \
  --model_path "${MODEL_PATH}" \
  --preference_heads_file "${agg_heads}" \
  --task "${TASK}" \
  --num_samples "${NUM_SAMPLES}"

python "preference_head/randomize_cluster_head_weights.py" \
  --src_dir "${head_dir_ablate}" \
  --out_dir "${random_head_dir}" \
  --seed 1234

mkdir -p "$(pwd)/outputs/hparam/quick/ablation/random_heads"
python "$(pwd)/scripts/run_weighted_dps.py" \
  --task "${TASK_DECODER}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type instruct \
  --cluster_file "${cluster_dir}/clusters.json" \
  --cluster_heads_dir "${random_head_dir}" \
  --embeddings_file "${emb_file}" \
  --routing soft \
  --temperature 1.0 \
  --num_samples "${NUM_SAMPLES}" \
  --run_dir "$(pwd)/outputs/hparam/quick/ablation/random_heads"

python "preference_head/random_mask_cluster_head_weights.py" \
  --src_dir "${head_dir_ablate}" \
  --out_dir "${random_mask_dir}" \
  --seed 1234 \
  --top_percent "${TOP_PERCENT_40}"

mkdir -p "$(pwd)/outputs/hparam/quick/ablation/random_mask"
python "$(pwd)/scripts/run_weighted_dps.py" \
  --task "${TASK_DECODER}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type instruct \
  --cluster_file "${cluster_dir}/clusters.json" \
  --cluster_heads_dir "${random_mask_dir}" \
  --embeddings_file "${emb_file}" \
  --routing soft \
  --temperature 1.0 \
  --num_samples "${NUM_SAMPLES}" \
  --run_dir "$(pwd)/outputs/hparam/quick/ablation/random_mask"

echo "Quick hparam runs complete."
