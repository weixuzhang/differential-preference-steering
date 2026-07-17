#!/bin/bash
#SBATCH --job-name=baseline_sanity
#SBATCH --output=baseline_sanity_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mail-type=ALL

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
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export HF_DATASETS_SINGLE_THREAD=true
export HYDRA_FULL_ERROR=1

cd "${ROOT}/decore"

# ---- Sanity-check settings ----
# You can override these for login-node runs, e.g.:
#   TASKS_LIST="lamp_1" MODELS_LIST="llama3_8b_instruct" METHODS_LIST="baseline" \
#   NUM_SAMPLES=1 MAX_NEW_TOKENS=1 bash experiments/baseline_scripts/run_baseline_sanity_check.sh
NUM_SAMPLES="${NUM_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
NUM_RETRIEVE="${NUM_RETRIEVE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"

TASKS_LIST="${TASKS_LIST:-lamp_1 lamp_2 lamp_3 lamp_4 lamp_5 lamp_7}"
MODELS_LIST="${MODELS_LIST:-llama3_8b_instruct qwen2_7b_instruct mistral_7b_instruct}"
METHODS_LIST="${METHODS_LIST:-baseline cad decore_vanilla dola}"

TASKS_LIST="${TASKS_LIST//,/ }"
MODELS_LIST="${MODELS_LIST//,/ }"
METHODS_LIST="${METHODS_LIST//,/ }"

if [ "${SANITY_FAST:-false}" = "true" ]; then
  TASKS_LIST="lamp_1 lamp_7"
fi

read -r -a TASKS <<< "${TASKS_LIST}"
read -r -a MODELS <<< "${MODELS_LIST}"
read -r -a METHODS <<< "${METHODS_LIST}"

failures=()

run_case() {
  local task="$1"
  local model="$2"
  local method="$3"
  local experiment=""
  local label=""
  local extra=()

  case "${method}" in
    baseline)
      experiment="${task}/baseline/${model}"
      label="baseline"
      ;;
    cad)
      experiment="${task}/context_aware_decoding/${model}"
      label="cad"
      ;;
    decore_vanilla)
      experiment="${task}/baseline/${model}"
      label="decore_vanilla"
      extra=(decoder=decore_vanilla)
      ;;
    dola)
      experiment="${task}/baseline/${model}"
      label="dola"
      extra=(decoder=dola)
      ;;
    *)
      echo "Unknown method: ${method}"
      return 1
      ;;
  esac

  echo "------------------------------------------------------------"
  echo "Sanity check: ${task} | ${model} | ${label}"
  echo "------------------------------------------------------------"

  cmd=(
    python scripts/main.py
    experiment="${experiment}"
    data.num_samples="${NUM_SAMPLES}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.num_retrieve="${NUM_RETRIEVE}"
    data_loader.num_workers="${NUM_WORKERS}"
    model.configs.max_new_tokens="${MAX_NEW_TOKENS}"
    debug=true
  )
  cmd+=("${extra[@]}")

  if "${cmd[@]}"; then
    echo "✅ OK: ${task} | ${model} | ${label}"
  else
    echo "❌ FAIL: ${task} | ${model} | ${label}"
    failures+=("${task}/${model}/${label}")
  fi
}

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_case "${task}" "${model}" "${method}"
    done
  done
done

echo "============================================================"
if [ ${#failures[@]} -eq 0 ]; then
  echo "All sanity checks passed."
else
  echo "Sanity check failures (${#failures[@]}):"
  printf ' - %s\n' "${failures[@]}"
  exit 1
fi
