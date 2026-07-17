#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

python -m pip install --upgrade \
  "huggingface-hub>=0.35.1" \
  "transformers>=4.51.3" \
  "tokenizers>=0.21.1"

echo "Phi-4-compatible HF stack installed."
