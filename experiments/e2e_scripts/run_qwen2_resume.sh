#!/bin/bash
#SBATCH --job-name=qwen2_resume
#SBATCH --output=qwen2_resume_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL

set -euo pipefail

ROOT="/scratch/weixuz"
source "${ROOT}/envs/decore/bin/activate"

# Cache directories
hf_cache="${ROOT}/dps/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export WANDB_DISABLED=true
export HF_OFFLINE=true
export HF_DATASETS_OFFLINE=true
export TRANSFORMERS_OFFLINE=true

# Optional: keep thread usage low if running on login nodes
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export HF_DATASETS_SINGLE_THREAD=true

# ---- Config ----
TASK_INPUT="${1:-LaMP-1}"
TARGET_GROUP="${TARGET_GROUP:-100}"
EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH="Qwen/Qwen2-7B-Instruct"
MODEL_NAME="Qwen2-7B-Instruct"

if [[ "${TASK_INPUT}" == LAMP_* ]]; then
  TASK_DECODER="${TASK_INPUT}"
  TASK="LaMP-${TASK_INPUT#LAMP_}"
else
  TASK="${TASK_INPUT}"
  TASK_DECODER="LAMP_${TASK_INPUT#LaMP-}"
fi

K=$(python "${ROOT}/preference_head/compute_k.py" --task "${TASK}" --split dev --target_group "${TARGET_GROUP}")
if [ -z "${K}" ]; then
  echo "Failed to compute K for ${TASK}"
  exit 1
fi

task_slug=$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
cluster_dir="${ROOT}/preference_head/cluster_runs/${task_slug}_k${K}"
cluster_file="${cluster_dir}/clusters.json"
emb_file="${cluster_dir}/embeddings_dev.npy"

model_slug=$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
old_head_dir="${ROOT}/preference_head/cluster_heads/${task_slug}_k${K}"
head_dir="${ROOT}/preference_head/cluster_heads/${task_slug}_k${K}_${model_slug}"

export OLD_HEAD_DIR="${old_head_dir}"
export NEW_HEAD_DIR="${head_dir}"
export MODEL_NAME
export K

echo "========================================="
echo "Task: ${TASK} | k=${K} (target group ${TARGET_GROUP})"
echo "Model: ${MODEL_NAME}"
echo "Clusters: ${cluster_dir}"
echo "Old heads: ${old_head_dir}"
echo "New heads: ${head_dir}"
echo "========================================="

if [ ! -f "${cluster_file}" ]; then
  echo "[0/4] Clustering dev profiles (missing clusters.json)..."
  python "${ROOT}/preference_head/cluster_profiles.py" \
    --task "${TASK}" \
    --split dev \
    --k "${K}" \
    --output_dir "${cluster_dir}" \
    --save_embeddings \
    --embedding_model "${EMB_MODEL}"
fi

if [ ! -f "${emb_file}" ]; then
  echo "[0.5/4] Building dev embeddings for routing..."
  python "${ROOT}/preference_head/embed_profiles.py" \
    --task "${TASK}" \
    --split dev \
    --output_file "${emb_file}" \
    --meta_file "${cluster_dir}/embeddings_dev.json" \
    --embedding_model "${EMB_MODEL}"
fi

echo "[1/4] Copying existing Qwen2 heads..."
missing_ids=$(python - <<'PY'
import json
import os
import shutil
from pathlib import Path

src = Path(os.environ["OLD_HEAD_DIR"])
dst = Path(os.environ["NEW_HEAD_DIR"])
k = int(os.environ["K"])
model = os.environ["MODEL_NAME"]

missing = []
dst.mkdir(parents=True, exist_ok=True)

for cid in range(k):
    src_cluster = src / f"cluster_{cid:02d}"
    hw = src_cluster / "head_weights.json"
    if not hw.exists():
        missing.append(cid)
        continue
    try:
        payload = json.loads(hw.read_text())
    except Exception:
        missing.append(cid)
        continue
    if payload.get("model") != model:
        missing.append(cid)
        continue
    dst_cluster = dst / f"cluster_{cid:02d}"
    if dst_cluster.exists():
        shutil.rmtree(dst_cluster)
    shutil.copytree(src_cluster, dst_cluster)

print(",".join(str(i) for i in missing))
PY
)

if [ -n "${missing_ids}" ]; then
  IFS=',' read -r -a missing_list <<< "${missing_ids}"
  echo "[2/4] Re-detecting missing clusters: ${missing_ids}"
  for cid in "${missing_list[@]}"; do
    python "${ROOT}/preference_head/detect_cluster_heads.py" \
      --cluster_file "${cluster_file}" \
      --model_path "${MODEL_PATH}" \
      --task "${TASK}" \
      --split dev \
      --num_samples 100 \
      --save_dir "${head_dir}" \
      --top_percent 0.04 \
      --pcs_norm max \
      --pcs_power 1.0 \
      --cluster_start "${cid}" \
      --cluster_end "${cid}"
  done
else
  echo "[2/4] No missing clusters detected."
fi

echo "[3/4] Verifying all Qwen2 head files exist..."
python - <<'PY'
import json
import os
from pathlib import Path

head_dir = Path(os.environ["NEW_HEAD_DIR"])
k = int(os.environ["K"])
model = os.environ["MODEL_NAME"]

missing = []
for cid in range(k):
    hw = head_dir / f"cluster_{cid:02d}" / "head_weights.json"
    if not hw.exists():
        missing.append(cid)
        continue
    payload = json.loads(hw.read_text())
    if payload.get("model") != model:
        missing.append(cid)

if missing:
    print("Missing or mismatched clusters:", missing)
    raise SystemExit(1)
print("All clusters present.")
PY

echo "[4/4] Running weighted DPS..."
python "${ROOT}/dps/scripts/run_weighted_dps.py" \
  --task "${TASK_DECODER}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type instruct \
  --cluster_file "${cluster_file}" \
  --cluster_heads_dir "${head_dir}" \
  --embeddings_file "${emb_file}" \
  --routing soft \
  --temperature 1.0
