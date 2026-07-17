#!/bin/bash
#SBATCH --job-name=phi4_dps
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-0

set -euo pipefail

ROOT="${DPS_ROOT:-${SLURM_SUBMIT_DIR:?Set DPS_ROOT or submit from the repo root}}"
# shellcheck disable=SC1090
source "${ROOT}/scripts/env_setup.sh"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-${DPS_ROOT}/runs/phi4_mini}"
TASKS_CSV="${TASKS_CSV:-LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7}"
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
TASK_ID="${TASK#LaMP-}"
TASK_SLUG="$(echo "${TASK}" | tr '[:upper:]' '[:lower:]' | tr -d '-')"

TARGET_GROUP="${TARGET_GROUP:-100}"
SPLIT="${SPLIT:-dev}"
MODEL_PATH="${MODEL_PATH:-microsoft/Phi-4-mini-instruct}"
MODEL_NAME="${MODEL_NAME:-Phi-4-Mini-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-instruct}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
ROUTING="${ROUTING:-soft}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_RETRIEVE="${NUM_RETRIEVE:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"

K="$(python "preference_head/compute_k.py" --task "${TASK}" --split "${SPLIT}" --target_group "${TARGET_GROUP}")"
CLUSTER_DIR="${RUN_ROOT}/cluster_runs/${TASK_SLUG}_k${K}"
HEADS_DIR="${RUN_ROOT}/cluster_heads/phi4_mini_instruct/${TASK_SLUG}_k${K}"
RUN_STAMP="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_DIR="${RUN_ROOT}/weighted_dps/${TASK_SLUG}/${RUN_STAMP}"

mkdir -p "${RUN_DIR}"

echo "========================================="
echo "Phi-4 weighted DPS"
echo "Task: ${TASK} (k=${K})"
echo "Cluster dir: ${CLUSTER_DIR}"
echo "Heads dir: ${HEADS_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Routing: ${ROUTING}"
echo "========================================="

bash "${ROOT}/run_weighted_dps.sh" \
  --task "${TASK}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_type "${MODEL_TYPE}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --num_samples "${NUM_SAMPLES}" \
  --retriever bm25 \
  --num_retrieve "${NUM_RETRIEVE}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --cluster_file "${CLUSTER_DIR}/clusters.json" \
  --cluster_heads_dir "${HEADS_DIR}" \
  --embeddings_file "${CLUSTER_DIR}/embeddings.npy" \
  --routing "${ROUTING}" \
  --temperature "${TEMPERATURE}" \
  --output_dir "${RUN_ROOT}/weighted_dps" \
  --run_dir "${RUN_DIR}"
