#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"
set -euo pipefail

# Prefetch models + eval metrics into the shared HF cache so sbatch jobs can run offline.

CACHE_ROOT="${HF_HOME:-${DPS_ROOT}/.cache/huggingface}"
export HF_HOME="$CACHE_ROOT"
export TRANSFORMERS_CACHE="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export HF_EVALUATE_CACHE="$CACHE_ROOT/evaluate"
export PYTHONNOUSERSITE=1
export HF_OFFLINE=false
export HF_HUB_OFFLINE=false
export HF_DATASETS_OFFLINE=false
export TRANSFORMERS_OFFLINE=false

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_EVALUATE_CACHE"

# Default model list (override by passing repo IDs as args)
MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen2-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Mistral-7B-v0.3"
  "microsoft/Phi-4-mini-instruct"
  "sentence-transformers/all-MiniLM-L6-v2"
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
fi

MODELS_ENV="$(IFS=';'; echo "${MODELS[*]}")"
export MODELS_ENV

python - <<'PY'
import os
from huggingface_hub import snapshot_download

models = os.environ["MODELS_ENV"].split(";")
cache_dir = os.environ["HF_HOME"]

for model in models:
    print(f"Prefetching model: {model}")
    snapshot_download(
        repo_id=model,
        cache_dir=cache_dir,
        local_files_only=False,
    )
PY

# Prefetch evaluation metrics used in LAMP
python - <<'PY'
import os
import evaluate

metrics = ["accuracy", "f1", "rouge", "meteor", "mae", "mse"]
for name in metrics:
    print(f"Prefetching metric: {name}")
    evaluate.load(name)
PY

# Prefetch LaMP/LongLaMP datasets (train + dev) into the local dataset root
export LAMP_DATA_ROOT="${LAMP_DATA_ROOT:-${DPS_ROOT}/banditpr/dataset}"
python "${ROOT}/scripts/prefetch_lamp_datasets.py" --splits train,dev

# Prefetch NLTK resources used by METEOR
python - <<'PY'
import nltk

packages = ["wordnet", "punkt", "punkt_tab", "omw-1.4"]
for pkg in packages:
    print(f"Prefetching NLTK: {pkg}")
    nltk.download(pkg, quiet=True)
PY

echo "✅ Prefetch complete. Cache at: $HF_HOME"
