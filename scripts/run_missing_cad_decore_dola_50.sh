#!/bin/bash
set -euo pipefail

ROOT="/scratch/weixuz"
NUM_SAMPLES=50

source "${ROOT}/envs/decore/bin/activate"

# Cache directories
hf_cache="${ROOT}/decore/.cache/huggingface"
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

cd "${ROOT}/decore"

missing_list="$(python - <<'PY'
import json
from pathlib import Path

summary_path = Path("/scratch/weixuz/outputs/evaluation_summary_combined.json")
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

tasks = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]
models = ["LLaMA3-8b-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2-7B-Instruct"]
methods = ["ContextAwareDecoding", "DeCoReVanilla", "DoLa"]

model_alias = {
    "Mistral-7b-Instruct": "Mistral-7B-Instruct-v0.3",
    "Qwen2-7b-Instruct": "Qwen2-7B-Instruct",
}

min_samples = 50
missing = []
for task in tasks:
    for model in models:
        for method in methods:
            key = f"{task}|{model}|{method}"
            total = summary.get(key, {}).get("total_samples", 0)
            if total < min_samples:
                # Also check aliases (model name variants)
                alias_total = 0
                for alt_model, canonical in model_alias.items():
                    if canonical == model:
                        alias_key = f"{task}|{alt_model}|{method}"
                        alias_total = max(alias_total, summary.get(alias_key, {}).get("total_samples", 0))
                total = max(total, alias_total)
            if total < min_samples:
                missing.append((task, model, method))

for row in missing:
    print("|".join(row))
PY
)"

if [[ -z "${missing_list}" ]]; then
  echo "No missing CAD/DeCoRe/DoLa runs (< ${NUM_SAMPLES} samples)."
  exit 0
fi

echo "Will run the following (task | model | method):"
echo "${missing_list}"

while IFS='|' read -r task model method; do
  if [[ -z "${task}" ]]; then
    continue
  fi
  task_slug="$(echo "${task}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
  case "${model}" in
    "LLaMA3-8b-Instruct") model_slug="llama3_8b_instruct" ;;
    "Mistral-7B-Instruct-v0.3"|"Mistral-7b-Instruct") model_slug="mistral_7b_instruct" ;;
    "Qwen2-7B-Instruct"|"Qwen2-7b-Instruct") model_slug="qwen2_7b_instruct" ;;
    *) echo "Unknown model: ${model}"; continue ;;
  esac

  case "${method}" in
    "ContextAwareDecoding") decoder="context_aware_decoding" ;;
    "DeCoReVanilla") decoder="decore_vanilla" ;;
    "DoLa") decoder="dola" ;;
    *) echo "Unknown method: ${method}"; continue ;;
  esac

  experiment="${task_slug}/baseline/${model_slug}"
  if [[ "${method}" == "ContextAwareDecoding" ]]; then
    experiment="${task_slug}/context_aware_decoding/${model_slug}"
  fi

  echo "------------------------------------------------------------"
  echo "Running ${task} | ${model} | ${method} (${NUM_SAMPLES} samples)"
  echo "------------------------------------------------------------"
  python scripts/main.py \
    experiment="${experiment}" \
    decoder="${decoder}" \
    data.num_samples="${NUM_SAMPLES}"
done <<< "${missing_list}"
