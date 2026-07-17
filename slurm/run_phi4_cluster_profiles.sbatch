#!/bin/bash
#SBATCH --job-name=phi4_cluster
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "${ROOT}/scripts/env_setup.sh"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-${DPS_ROOT}/runs/phi4_mini}"
TASKS_CSV="${TASKS_CSV:-LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

export TARGET_GROUP="${TARGET_GROUP:-100}"
export SPLIT="${SPLIT:-dev}"
export EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
export BASE_OUT="${BASE_OUT:-${RUN_ROOT}/cluster_runs}"

mkdir -p "${RUN_ROOT}" "${BASE_OUT}"

echo "========================================="
echo "Phi-4 cluster profiling"
echo "Run root: ${RUN_ROOT}"
echo "Split: ${SPLIT}"
echo "Target group size: ${TARGET_GROUP}"
echo "Tasks: ${TASKS_CSV}"
echo "========================================="

bash "${ROOT}/preference_head/run_cluster_profiles.sh" "${TASKS[@]}"
