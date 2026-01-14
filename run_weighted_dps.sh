#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_weighted_dps.py args...>" >&2
  echo "Example: $0 --task LAMP_1 --model_path meta-llama/Meta-Llama-3-8B-Instruct \\" >&2
  echo "  --model_name LLaMA3-8b-Instruct --cluster_file <clusters.json> \\" >&2
  echo "  --cluster_heads_dir <heads_dir> --embeddings_file <emb.npy> --routing soft" >&2
  exit 1
fi

python "${ROOT}/scripts/run_weighted_dps.py" "$@"
