#!/bin/bash
#SBATCH --job-name=phi4_heads
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=18:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-0

set -euo pipefail

ROOT="${DPS_ROOT:-${SLURM_SUBMIT_DIR:?Set DPS_ROOT or submit from the repo root}}"
# shellcheck disable=SC1090
source "${ROOT}/scripts/env_setup.sh"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-${DPS_ROOT}/runs/phi4_mini}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/phi4_cluster_head_manifest.tsv}"

export MODEL_PATH="${MODEL_PATH:-microsoft/Phi-4-mini-instruct}"
export BASE_CLUSTER="${BASE_CLUSTER:-${RUN_ROOT}/cluster_runs}"
export BASE_OUT="${BASE_OUT:-${RUN_ROOT}/cluster_heads/phi4_mini_instruct}"
export SPLIT="${SPLIT:-dev}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export TOP_PERCENT="${TOP_PERCENT:-0.04}"
export PCS_NORM="${PCS_NORM:-max}"
export PCS_POWER="${PCS_POWER:-1.0}"
export MANIFEST

mkdir -p "${BASE_OUT}"

echo "========================================="
echo "Phi-4 cluster head detection"
echo "Manifest: ${MANIFEST}"
echo "Array index: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_PATH}"
echo "Output root: ${BASE_OUT}"
echo "========================================="

bash "${ROOT}/preference_head/run_detect_cluster_heads.sh" "${SLURM_ARRAY_TASK_ID}"
