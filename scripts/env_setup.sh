#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPS_ROOT="${DPS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export DPS_ROOT

DECORE_ENV="${DECORE_ENV:-${DPS_ROOT}/.venv}"
if [ -f "${DECORE_ENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${DECORE_ENV}/bin/activate"
else
  echo "[warn] DECORE_ENV not found: ${DECORE_ENV}. Set DECORE_ENV to your venv path." >&2
fi

# LaMP/LongLaMP dataset location (loader code lives in src/lamp_benchmark)
export LAMP_DATA_ROOT="${LAMP_DATA_ROOT:-/scratch/weixuz/lamp_data}"

HF_CACHE="${HF_HOME:-${DPS_ROOT}/.cache/huggingface}"
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"

export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_OFFLINE="${HF_OFFLINE:-true}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-true}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
