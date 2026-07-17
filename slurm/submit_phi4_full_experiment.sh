#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "${ROOT}/scripts/env_setup.sh"
cd "${ROOT}"

ACTIVE_ENV="${DECORE_ENV:-${DPS_ROOT}/.venv}"
if [[ ! -f "${ACTIVE_ENV}/bin/activate" ]]; then
  echo "Missing Python environment: ${ACTIVE_ENV}" >&2
  echo "Export DECORE_ENV=/path/to/your/venv before submitting Phi-4 jobs." >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-${DPS_ROOT}/runs/phi4_mini}"
TASKS_CSV="${TASKS_CSV:-LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7}"
TARGET_GROUP="${TARGET_GROUP:-100}"
SPLIT="${SPLIT:-dev}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/phi4_cluster_head_manifest.tsv}"

mkdir -p "${RUN_ROOT}"

echo "Building manifest: ${MANIFEST}"
TASKS="${TASKS_CSV}" SPLIT="${SPLIT}" \
  bash "preference_head/build_cluster_heads_manifest.sh" \
  "${TARGET_GROUP}" "${CHUNK_SIZE}" "${MANIFEST}"

NUM_HEAD_JOBS="$(wc -l < "${MANIFEST}")"
IFS=',' read -r -a TASKS_ARR <<< "${TASKS_CSV}"
NUM_TASKS="${#TASKS_ARR[@]}"

if [[ "${NUM_HEAD_JOBS}" -le 0 ]]; then
  echo "Manifest is empty: ${MANIFEST}" >&2
  exit 1
fi

cluster_job="$(sbatch --parsable \
  --export=ALL,RUN_ROOT="${RUN_ROOT}",TARGET_GROUP="${TARGET_GROUP}",SPLIT="${SPLIT}",TASKS_CSV="${TASKS_CSV}" \
  "${ROOT}/slurm/run_phi4_cluster_profiles.sbatch")"

detect_job="$(sbatch --parsable \
  --dependency=afterok:${cluster_job} \
  --array=0-$((NUM_HEAD_JOBS - 1)) \
  --export=ALL,RUN_ROOT="${RUN_ROOT}",SPLIT="${SPLIT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/slurm/run_phi4_detect_cluster_heads_array.sbatch")"

dps_job="$(sbatch --parsable \
  --dependency=afterok:${detect_job} \
  --array=0-$((NUM_TASKS - 1)) \
  --export=ALL,RUN_ROOT="${RUN_ROOT}",TARGET_GROUP="${TARGET_GROUP}",SPLIT="${SPLIT}",TASKS_CSV="${TASKS_CSV}" \
  "${ROOT}/slurm/run_phi4_weighted_dps_array.sbatch")"

baseline_job="$(sbatch --parsable \
  --array=0-$((NUM_TASKS - 1)) \
  --export=ALL,RUN_ROOT="${RUN_ROOT}",TASKS_CSV="${TASKS_CSV}" \
  "${ROOT}/slurm/run_phi4_baselines_array.sbatch")"

cat <<EOF
Submitted Phi-4 full experiment.

  cluster profiles : ${cluster_job}
  head detection   : ${detect_job}
  weighted DPS     : ${dps_job}
  baselines        : ${baseline_job}

Prefetch first on the login node if Phi-4 is not cached yet:
  bash scripts/prefetch_models_and_metrics.sh microsoft/Phi-4-mini-instruct sentence-transformers/all-MiniLM-L6-v2

Notes:
  - This submission flow schedules LaMP-1/2/3/4/5/7 by default.
  - Baselines here are limited to baseline + context-aware decoding.
  - Retrieval-head baselines and DoLa are not included for Phi-4-mini in this repo path.
  - Aggregate finished results on the login node with:
      bash scripts/aggregate_phi4_results.sh "${RUN_ROOT}"
EOF
