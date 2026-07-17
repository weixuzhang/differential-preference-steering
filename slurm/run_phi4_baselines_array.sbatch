#!/bin/bash
#SBATCH --job-name=phi4_base
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-0

set -euo pipefail

ROOT="${DPS_ROOT:-${SLURM_SUBMIT_DIR:?Set DPS_ROOT or submit from the repo root}}"
# shellcheck disable=SC1090
source "${ROOT}/scripts/env_setup.sh"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-${DPS_ROOT}/runs/phi4_mini}"
TASKS_CSV="${TASKS_CSV:-LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7}"
METHODS_CSV="${METHODS_CSV:-baseline,context_aware_decoding}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"

TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
TASK_ID="${TASK#LaMP-}"
TASK_SLUG="$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')"
DATA_CFG="lamp_${TASK_ID}"

MODEL_CFG="${MODEL_CFG:-phi4_mini_instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

echo "========================================="
echo "Phi-4 baselines"
echo "Task: ${TASK}"
echo "Methods: ${METHODS_CSV}"
echo "========================================="

for method in "${METHODS[@]}"; do
  RUN_STAMP="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  RUN_DIR="${RUN_ROOT}/baselines/${TASK_SLUG}/${method}/${RUN_STAMP}"
  mkdir -p "${RUN_DIR}"
  echo "--- ${TASK} | ${method} ---"
  python "${ROOT}/scripts/main.py" \
    model="${MODEL_CFG}" \
    data="${DATA_CFG}" \
    decoder="${method}" \
    hydra.run.dir="${RUN_DIR}" \
    model.configs.max_new_tokens="${MAX_NEW_TOKENS}"
done
