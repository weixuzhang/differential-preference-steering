# Differential Preference Steering (DPS)

This repository contains code and scripts for **Differential Preference Steering (DPS)**, a personalization method that detects preference-sensitive attention heads and steers decoding by scaling those heads in a depersonalized pass. The project extends DeCoRe-style contrastive decoding with **preference head detection**, **cluster routing**, and **weighted head scaling**.

## What’s inside
- `src/` — core models, datasets, metrics, and decoding logic (DPS, CAD, DeCoRe, DoLa);
  `src/lamp_benchmark/` holds the LaMP/LongLaMP loading, prompting, and metric utilities.
- `preference_head/` — profile embedding, clustering, head detection, and diagnostics
  (code only; detection outputs live under `results/preference_head/`).
- `configs/` — Hydra configs for experiments, models, and decoders.
- `scripts/` — entry points and utilities (`main.py`, weighted DPS, prefetch, plotting).
- `experiments/` — experiment launchers: main-table runs, `baseline_scripts/`,
  `e2e_scripts/`, `hparam_scripts/`, quick smoke tests.
- `slurm/` — Slurm pipeline for the Phi-4 experiment suite.
- `human_eval/` — LLM-judge evaluation scripts, annotations, and win rates.
- `results/`, `plots/` — aggregated evaluation tables, preference-head detection
  outputs (`results/preference_head/`), and generated figures.
- `docs/` — methodology, workflow, integration and analysis write-ups
  (`docs/tempdocs/` holds historical dev notes).
- `data/`, `notebooks/`, `retrieval_heads/`, `tests/` — eval data, analysis
  notebooks, retrieval-head JSONs, integration tests.

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
export LAMP_DATA_ROOT=/path/to/dataset     # optional; LaMP/LongLaMP data location
```

Most shell scripts source `scripts/env_setup.sh` to set cache and offline flags.

## Data
LaMP/LongLaMP datasets are **not included** in the repo. Point `LAMP_DATA_ROOT`
at your local dataset directory (one subdirectory per task, e.g. `LaMP-1/`,
`LongLaMP-2/`). Loading, prompting, and metric code lives in `src/lamp_benchmark`.

## Quick run (example)
```
# 1) Cluster profiles (dev split)
bash preference_head/run_cluster_profiles.sh

# 2) Detect heads
bash preference_head/run_detect_cluster_heads.sh

# 3) Run DPS decoding
bash run_weighted_dps.sh
```

## Slurm pipeline for Phi-4-mini
For a smaller-model replacement of the Mistral runs, the repo now includes a Phi-4-mini path:
```
# Prefetch once on the login node
bash scripts/upgrade_phi4_stack.sh   # only needed if your HF stack is older
bash scripts/prefetch_models_and_metrics.sh microsoft/Phi-4-mini-instruct sentence-transformers/all-MiniLM-L6-v2

# Submit the LaMP-1/2/3/4/5/7 pipeline
bash slurm/submit_phi4_full_experiment.sh
```

This schedules:
- profile clustering,
- per-cluster preference-head detection,
- weighted DPS decoding with soft routing,
- baseline and context-aware decoding baselines.

Retrieval-head DeCoRe baselines and DoLa are not part of the Phi-4-mini Slurm path in this release copy.

## Human evaluation
LLM evaluation scripts live in `human_eval/`. Raw annotation files and prompt JSONL files are excluded from this anonymized repo.

## Notes on anonymization
This repo removes:
- dataset copies, large outputs, and cache artifacts,
- local logs and notebook artifacts,
- absolute paths tied to a specific user or machine.
Runtime outputs (e.g., `output/`, `outputs/`, `results/`) are generated when you run experiments and are not included in the release copy.

If you need to reproduce the full pipeline, set `DPS_ROOT`, `DECORE_ENV`, and dataset paths, then rerun the scripts.
