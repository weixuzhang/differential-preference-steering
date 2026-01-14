#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_config> [hydra_overrides...]" >&2
  echo "Example: $0 lamp_1/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40" >&2
  exit 1
fi

EXPERIMENT="$1"
shift

python "${ROOT}/scripts/main.py" "experiment=${EXPERIMENT}" "$@"
