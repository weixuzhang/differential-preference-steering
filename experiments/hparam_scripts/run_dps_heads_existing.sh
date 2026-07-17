#!/bin/bash
set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

# Cache directories (offline)
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

# ---- Config ----
TASK="LaMP-1"
TASK_DECODER="LAMP_1"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"
HEAD_COUNTS=(10 20 40 80 160)
NUM_SAMPLES="${2:-50}"
if [ "${NUM_SAMPLES}" -ne -1 ] && [ "${NUM_SAMPLES}" -le 0 ]; then
  echo "NUM_SAMPLES must be -1 (full dev) or >0"
  exit 1
fi

task_slug="lamp1"
model_slug="llama3-8b-instruct"
cluster_dir="results/preference_head/cluster_runs/${task_slug}_k25"
emb_file="${cluster_dir}/embeddings_dev.npy"

if [ ! -f "${cluster_dir}/clusters.json" ]; then
  echo "Missing clusters: ${cluster_dir}/clusters.json"
  exit 1
fi
if [ ! -f "${emb_file}" ]; then
  echo "Missing embeddings: ${emb_file}"
  exit 1
fi

echo "========================================="
echo "Heads sweep (DPS only) | ${TASK}"
echo "Model: ${MODEL_NAME}"
echo "GPU: ${GPU_ID}"
echo "Samples: ${NUM_SAMPLES}"
echo "Cluster dir: ${cluster_dir}"
echo "========================================="

for num_heads in "${HEAD_COUNTS[@]}"; do
  head_dir="results/preference_head/cluster_heads/${task_slug}_k25_${model_slug}_quick50_h${num_heads}"
  head_dir_filled="${head_dir}_filled"
  run_dir="$(pwd)/outputs/hparam/heads_quick/h${num_heads}"
  emb_subset="${cluster_dir}/embeddings_dev_n${NUM_SAMPLES}.npy"

  if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
    echo "Skipping h=${num_heads} (missing ${head_dir}/cluster_00/head_weights.json)"
    continue
  fi

  # Fill missing cluster head_weights with zeros so DPS can run.
  python - <<PY
import json
import shutil
from pathlib import Path
import numpy as np

cluster_dir = Path("${cluster_dir}")
head_dir = Path("${head_dir}")
filled_dir = Path("${head_dir_filled}")

with (cluster_dir / "clusters.json").open("r") as f:
    cluster_data = json.load(f)
k = int(cluster_data["k"])

example = head_dir / "cluster_00" / "head_weights.json"
if not example.exists():
    raise FileNotFoundError(f"Missing {example}")
with example.open("r") as f:
    template = json.load(f)
num_layers = int(template["num_layers"])
num_heads = int(template["num_heads"])

missing = []
for cid in range(k):
    if not (head_dir / f"cluster_{cid:02d}" / "head_weights.json").exists():
        missing.append(cid)

if not missing:
    print("No missing head weights; no fill needed.")
    raise SystemExit(0)

filled_dir.mkdir(parents=True, exist_ok=True)

for cid in range(k):
    src_cluster = head_dir / f"cluster_{cid:02d}"
    out_cluster = filled_dir / f"cluster_{cid:02d}"
    out_cluster.mkdir(parents=True, exist_ok=True)
    src_weights = src_cluster / "head_weights.json"
    if src_weights.exists():
        shutil.copy2(src_weights, out_cluster / "head_weights.json")
        src_npy = src_cluster / "head_weights.npy"
        if src_npy.exists():
            shutil.copy2(src_npy, out_cluster / "head_weights.npy")
        continue
    weights = np.zeros((num_layers, num_heads), dtype=np.float32)
    payload = dict(template)
    payload["cluster_id"] = cid
    payload["head_weights"] = weights.tolist()
    with (out_cluster / "head_weights.json").open("w") as f:
        json.dump(payload, f, indent=2)
    np.save(out_cluster / "head_weights.npy", weights)

print(f"Filled {len(missing)} missing clusters into {filled_dir}")
PY

  if [ -d "${head_dir_filled}" ]; then
    head_dir="${head_dir_filled}"
  fi

  if [ "${NUM_SAMPLES}" -gt 0 ]; then
    if [ ! -f "${emb_subset}" ]; then
      python - <<PY
import numpy as np
emb = np.load("${emb_file}")
np.save("${emb_subset}", emb[: int("${NUM_SAMPLES}")])
PY
    fi
  else
    emb_subset="${emb_file}"
  fi

  mkdir -p "${run_dir}"
  python "$(pwd)/scripts/run_weighted_dps.py" \
    --task "${TASK_DECODER}" \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_type instruct \
    --cluster_file "${cluster_dir}/clusters.json" \
    --cluster_heads_dir "${head_dir}" \
    --embeddings_file "${emb_subset}" \
    --routing soft \
    --temperature 1.0 \
    --num_samples "${NUM_SAMPLES}" \
    --run_dir "${run_dir}"
done

echo "Heads sweep DPS runs complete."
