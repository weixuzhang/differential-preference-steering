# Differential Preference Steering (DPS)

This repository contains code and scripts for **Differential Preference Steering (DPS)**, a personalization method that detects preference-sensitive attention heads and steers decoding by scaling those heads in a depersonalized pass. The project extends DeCoRe-style contrastive decoding with **preference head detection**, **cluster routing**, and **weighted head scaling**.

## What’s inside
- `src/` — core models, datasets, metrics, and decoding logic (DPS, CAD, DeCoRe, DoLa).
- `preference_head/` — profile embedding, clustering, head detection, and diagnostics.
- `configs/` — Hydra configs for experiments and decoders.
- `scripts/` — utilities for DPS runs, dataset prefetch, and efficiency plots.
- `human_eval/` — LLM evaluation scripts (no raw prompts or summaries included).

## Setup
Create a Python environment and install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment variables:
```
export DPS_ROOT=$(pwd)
export DECORE_ENV=/path/to/venv            # optional; used by shell scripts
export BANDITPR_ROOT=/path/to/banditpr     # optional; for dataset loading
export LAMP_DATA_ROOT=/path/to/dataset     # optional; overrides BANDITPR_ROOT/dataset
```

Most shell scripts source `scripts/env_setup.sh` to set cache and offline flags.

## Data
LaMP/LongLaMP datasets are **not included** in this anonymized repo. Point to your local data via:
- `LAMP_DATA_ROOT` (preferred), or
- `BANDITPR_ROOT` (expects `dataset/` under that repo).

## Quick run (example)
```
# 1) Cluster profiles (dev split)
bash preference_head/run_cluster_profiles.sh

# 2) Detect heads
bash preference_head/run_detect_cluster_heads.sh

# 3) Run DPS decoding
bash run_weighted_dps.sh
```

## Human evaluation
LLM evaluation scripts live in `human_eval/`. Raw annotation files and prompt JSONL files are excluded from this anonymized repo.

## Notes on anonymization
This repo removes:
- dataset copies, large outputs, and cache artifacts,
- local logs and notebook artifacts,
- absolute paths tied to a specific user or machine.
Runtime outputs (e.g., `output/`, `outputs/`, `results/`) are generated when you run experiments and are not included in the release copy.

If you need to reproduce the full pipeline, set `DPS_ROOT`, `DECORE_ENV`, and dataset paths, then rerun the scripts.
