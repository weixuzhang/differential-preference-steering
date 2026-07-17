#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SETUP="${ROOT}/scripts/env_setup.sh"
# shellcheck disable=SC1090
source "${ENV_SETUP}"

RUN_ROOT="${1:-${DPS_ROOT}/runs/phi4_mini}"
OUT_DIR="${2:-${RUN_ROOT}/summary}"

python "${ROOT}/scripts/aggregate_phi4_results.py" \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUT_DIR}"
