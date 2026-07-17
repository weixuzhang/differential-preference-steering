#!/bin/bash
set -euo pipefail

GPU_ID="${1:-0}"
NUM_SAMPLES="${2:-200}"
if [ "${NUM_SAMPLES}" -le 0 ]; then
  echo "NUM_SAMPLES must be > 0"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

# Cache directories (offline)
hf_cache="${ROOT}/dps/.cache/huggingface"
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
TASKS=("LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7")
TARGET_GROUP=100
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="LLaMA3-8b-Instruct"

model_slug=$(echo "${MODEL_NAME}" | tr "[:upper:]" "[:lower:]" | tr -c "a-z0-9" "-" | sed "s/--*/-/g" | sed "s/^-//;s/-$//")

echo "========================================="
echo "Routing ablation | model=${MODEL_NAME} | samples=${NUM_SAMPLES}"
echo "GPU: ${GPU_ID}"
echo "========================================="

for TASK in "${TASKS[@]}"; do
  TASK_DECODER="LAMP_${TASK#LaMP-}"

  task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  cluster_dir=$(python - <<PY
import glob
import json
import os

task_slug = "${task_slug}"
target_group = int("${TARGET_GROUP}")
base = "/scratch/weixuz/dps/preference_head/cluster_runs"
cands = glob.glob(os.path.join(base, f"{task_slug}_k*"))
best = None
best_score = None
for c in cands:
    try:
        with open(os.path.join(c, "clusters.json"), "r") as f:
            data = json.load(f)
        k = int(data.get("k", 0))
        n = int(data.get("num_samples", 0))
        if k <= 0 or n <= 0:
            continue
        group_size = n / k
        score = abs(group_size - target_group)
        if best_score is None or score < best_score:
            best_score = score
            best = c
    except Exception:
        continue
if best:
    print(best)
PY
)
  if [ -z "${cluster_dir}" ]; then
    echo "Missing cluster runs for ${TASK}, skipping."
    continue
  fi
  K=$(python - <<PY
import json
with open("${cluster_dir}/clusters.json", "r") as f:
    data = json.load(f)
print(int(data.get("k", 0)))
PY
)
  if [ -z "${K}" ] || [ "${K}" -le 0 ]; then
    echo "Invalid k for ${TASK}, skipping."
    continue
  fi

  emb_file="${cluster_dir}/embeddings_dev.npy"
  head_dir="${ROOT}/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}"
  head_dir_filled="${head_dir}_filled"
  emb_subset="${cluster_dir}/embeddings_dev_n${NUM_SAMPLES}.npy"

  if [ ! -f "${cluster_dir}/clusters.json" ]; then
    echo "Missing ${cluster_dir}/clusters.json, skipping ${TASK}."
    continue
  fi
  if [ ! -f "${emb_file}" ]; then
    echo "Missing ${emb_file}, skipping ${TASK}."
    continue
  fi
  if [ ! -f "${head_dir}/cluster_00/head_weights.json" ]; then
    echo "Missing head weights in ${head_dir}, skipping ${TASK}."
    continue
  fi

  # Fill missing cluster head_weights with zeros if needed.
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

  if [ ! -f "${emb_subset}" ]; then
    python - <<PY
import numpy as np
emb = np.load("${emb_file}")
n = int("${NUM_SAMPLES}")
n = min(n, emb.shape[0])
np.save("${emb_subset}", emb[:n])
print(f"Saved embeddings subset: {n}")
PY
  fi

  echo "-----------------------------------------"
  echo "Task: ${TASK} | k=${K}"
  echo "Cluster dir: ${cluster_dir}"
  echo "Heads dir: ${head_dir}"
  echo "Embeddings: ${emb_subset}"
  echo "-----------------------------------------"

  run_dir_soft="${ROOT}/dps/outputs/hparam/routing_ablation/${task_slug}/soft"
  run_dir_hard="${ROOT}/dps/outputs/hparam/routing_ablation/${task_slug}/hard"
  mkdir -p "${run_dir_soft}" "${run_dir_hard}"

  python "${ROOT}/dps/scripts/run_weighted_dps.py" \
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
    --run_dir "${run_dir_soft}"

  python "${ROOT}/dps/scripts/run_weighted_dps.py" \
    --task "${TASK_DECODER}" \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_type instruct \
    --cluster_file "${cluster_dir}/clusters.json" \
    --cluster_heads_dir "${head_dir}" \
    --embeddings_file "${emb_subset}" \
    --routing hard \
    --temperature 1.0 \
    --num_samples "${NUM_SAMPLES}" \
    --run_dir "${run_dir_hard}"
done

echo "Routing ablation complete."
