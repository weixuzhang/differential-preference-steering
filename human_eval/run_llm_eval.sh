#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before running." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_jsonl...> [--model gpt-5.2]" >&2
  echo "Example: $0 human_eval/LaMP-4_dps_vs_cad.jsonl human_eval/LaMP-7_dps_vs_cad.jsonl" >&2
  exit 1
fi

MODEL="gpt-5.2"
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      INPUTS+=("$1")
      shift
      ;;
  esac
done

OUTPUT="${ROOT}/human_eval/llm_eval_gpt5_2.jsonl"
SUMMARY="${ROOT}/human_eval/llm_eval_gpt5_2_summary.json"

rm -f "${OUTPUT}" "${SUMMARY}"
python "${ROOT}/human_eval/run_llm_eval_gpt5_2.py" \
  --inputs "${INPUTS[@]}" \
  --output "${OUTPUT}" \
  --summary "${SUMMARY}" \
  --model "${MODEL}"
