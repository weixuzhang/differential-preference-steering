#!/bin/bash
set -euo pipefail

GPU_ID="${1:-0}"
shift || true
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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

# ---- Config ----
TASK="LaMP-1"
TASK_DECODER="LAMP_1"
if [ "$#" -gt 0 ]; then
  GROUP_SIZES=("$@")
else
  GROUP_SIZES=(10 50 100 200 400)
fi
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"
EVAL_SAMPLES=50
NUM_SAMPLES=25
NUM_SAMPLES_SMALL=5
SMALL_GROUP_SIZE=10
CHUNK_SIZE_SMALL=50
TOP_PERCENT_40=$(python -c "print(40/1024)")

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")

# Reuse cached embeddings (skip embedding model).
K_BASE=$(python "preference_head/compute_k.py" --task "${TASK}" --split dev --target_group 100)
BASE_CLUSTER="results/preference_head/cluster_runs/${task_slug}_k${K_BASE}"
BASE_EMB="${BASE_CLUSTER}/embeddings_dev.npy"
if [ ! -f "${BASE_EMB}" ]; then
  echo "Missing base embeddings: ${BASE_EMB}"
  exit 1
fi

echo "========================================="
echo "Group-size sweep (quick) | ${TASK}"
echo "Model: ${MODEL_NAME}"
echo "GPU: ${GPU_ID}"
echo "Eval samples: ${EVAL_SAMPLES}"
echo "========================================="

for target_group in "${GROUP_SIZES[@]}"; do
  K=$(python "preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${target_group}")
  if [ -z "${K}" ]; then
    echo "Failed to compute K for ${TASK} (target_group=${target_group})"
    continue
  fi

  cluster_dir="results/preference_head/cluster_runs/${task_slug}_k${K}"
  emb_file="${cluster_dir}/embeddings_dev.npy"
  head_dir="results/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}_quick50"
  run_dir="$(pwd)/outputs/hparam/groupsize_quick/g${target_group}"

  if [ ! -f "${cluster_dir}/clusters.json" ]; then
    echo "Building clusters from cached embeddings (k=${K})..."
    python - <<PY
import json
from pathlib import Path
import numpy as np

def kmeans(embeddings, k, max_iter=25, tol=1e-4, seed=1234):
    rng = np.random.default_rng(seed)
    n, dim = embeddings.shape
    centroids = embeddings[rng.integers(0, n, size=1)].astype(np.float32)
    while centroids.shape[0] < k:
        distances = np.sum((embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        closest = np.min(distances, axis=1)
        probs = closest / max(np.sum(closest), 1e-12)
        idx = rng.choice(n, p=probs)
        centroids = np.vstack([centroids, embeddings[idx]])
    for _ in range(max_iter):
        distances = np.sum((embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            pts = embeddings[labels == c]
            if len(pts) == 0:
                new_centroids[c] = embeddings[rng.integers(0, embeddings.shape[0])]
            else:
                new_centroids[c] = pts.mean(axis=0)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break
    return labels, centroids

task = "${TASK}"
split = "dev"
k = int("${K}")
base_emb = Path("${BASE_EMB}")
out_dir = Path("${cluster_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

embeddings = np.load(base_emb)
labels, centroids = kmeans(embeddings.astype(np.float32), k)
cluster_sizes = [int((labels == i).sum()) for i in range(k)]

emb_path = out_dir / "embeddings_dev.npy"
np.save(emb_path, embeddings)

cluster_data = {
    "task": task,
    "split": split,
    "k": k,
    "embedding_model": "cached",
    "max_profiles": 5,
    "max_length": 256,
    "normalize": False,
    "num_samples": int(embeddings.shape[0]),
    "cluster_sizes": cluster_sizes,
    "cluster_assignments": labels.tolist(),
    "centroids": centroids.tolist(),
    "embeddings_path": str(emb_path),
}

with (out_dir / "clusters.json").open("w") as f:
    json.dump(cluster_data, f, indent=2)
print(f"Saved cluster assignments to: {out_dir / 'clusters.json'}")
PY
  fi

  if [ ! -f "${emb_file}" ]; then
    cp "${BASE_EMB}" "${emb_file}"
  fi

  if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
    echo "Detecting heads (k=${K}, target_group=${target_group})..."
    if [ "${target_group}" -eq "${SMALL_GROUP_SIZE}" ]; then
      cluster_start=0
      while [ "${cluster_start}" -lt "${K}" ]; do
        cluster_end=$((cluster_start + CHUNK_SIZE_SMALL - 1))
        if [ "${cluster_end}" -ge "${K}" ]; then
          cluster_end=$((K - 1))
        fi
        python "preference_head/detect_cluster_heads.py" \
          --cluster_file "${cluster_dir}/clusters.json" \
          --model_path "${MODEL_PATH}" \
          --task "${TASK}" \
          --split dev \
          --num_samples "${NUM_SAMPLES_SMALL}" \
          --max_samples_per_cluster "${NUM_SAMPLES_SMALL}" \
          --save_dir "${head_dir}" \
          --top_percent "${TOP_PERCENT_40}" \
          --pcs_norm max \
          --pcs_power 1.0 \
          --cluster_start "${cluster_start}" \
          --cluster_end "${cluster_end}"
        cluster_start=$((cluster_end + 1))
      done
    else
      python "preference_head/detect_cluster_heads.py" \
        --cluster_file "${cluster_dir}/clusters.json" \
        --model_path "${MODEL_PATH}" \
        --task "${TASK}" \
        --split dev \
        --num_samples "${NUM_SAMPLES}" \
        --max_samples_per_cluster "${NUM_SAMPLES}" \
        --save_dir "${head_dir}" \
        --top_percent "${TOP_PERCENT_40}" \
        --pcs_norm max \
        --pcs_power 1.0 \
        --cluster_start 0 \
        --cluster_end "$((K - 1))"
    fi
  fi

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
    --num_samples "${EVAL_SAMPLES}" \
    --run_dir "${run_dir}"
done

echo "Group-size sweep complete."
