#!/bin/bash
#SBATCH --job-name=phi4_agg
#SBATCH --output=%x_%j.out
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT="${DPS_ROOT:-${SLURM_SUBMIT_DIR:?Set DPS_ROOT or submit from the repo root}}"
# shellcheck disable=SC1090
source "${ROOT}/scripts/env_setup.sh"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-${DPS_ROOT}/runs/phi4_mini}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/summary}"

echo "========================================="
echo "Phi-4 result aggregation"
echo "Run root: ${RUN_ROOT}"
echo "Output:   ${OUT_DIR}"
echo "========================================="

bash "${ROOT}/scripts/aggregate_phi4_results.sh" "${RUN_ROOT}" "${OUT_DIR}"
