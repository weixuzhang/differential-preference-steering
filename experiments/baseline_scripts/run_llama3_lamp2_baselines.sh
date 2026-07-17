#!/bin/bash
#SBATCH --job-name=llama3_l2_baselines
#SBATCH --output=llama3_l2_baselines_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
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

cd "${ROOT}/decore"

TASK="lamp_2"
MODEL="llama3_8b_instruct"

failures=()

run_case() {
  local method="$1"
  local experiment=""
  local label=""
  local extra=()

  case "${method}" in
    baseline)
      experiment="${TASK}/baseline/${MODEL}"
      label="baseline"
      ;;
    cad)
      experiment="${TASK}/context_aware_decoding/${MODEL}"
      label="cad"
      ;;
    decore_vanilla)
      experiment="${TASK}/baseline/${MODEL}"
      label="decore_vanilla"
      extra=(decoder=decore_vanilla)
      ;;
    dola)
      experiment="${TASK}/baseline/${MODEL}"
      label="dola"
      extra=(decoder=dola)
      ;;
    *)
      echo "Unknown method: ${method}"
      return 1
      ;;
  esac

  echo "------------------------------------------------------------"
  echo "Baseline: ${TASK} | ${MODEL} | ${label}"
  echo "------------------------------------------------------------"

  cmd=(
    python scripts/main.py
    experiment="${experiment}"
  )
  cmd+=("${extra[@]}")

  if "${cmd[@]}"; then
    echo "✅ OK: ${TASK} | ${MODEL} | ${label}"
  else
    echo "❌ FAIL: ${TASK} | ${MODEL} | ${label}"
    failures+=("${TASK}/${MODEL}/${label}")
  fi
}

for method in baseline cad decore_vanilla dola; do
  run_case "${method}"
done

echo "============================================================"
if [ ${#failures[@]} -eq 0 ]; then
  echo "All baselines completed for ${TASK} | ${MODEL}."
else
  echo "Baseline failures (${#failures[@]}):"
  printf ' - %s\n' "${failures[@]}"
  exit 1
fi
