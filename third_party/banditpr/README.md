# Vendored: BanditPR `lamp` package

Source: https://github.com/haolun-wu/BanditPR (`src/lamp/` at commit `ee6c21f`,
plus local modifications to `dataset.py` and `__init__.py` — see
`docs/banditpr_local_modifications.patch` for the diff against upstream).

This is the LaMP/LongLaMP data-loading, prompting, and metric package that
`src/datasets/lamp.py` and `src/metrics/lamp.py` import
(`load_lamp_dataset`, `create_prompt_generator`, `create_metric`, `get_labels`).
Only this package is vendored; the rest of BanditPR (the bandit retriever
method itself) is not used by DPS.

`BANDITPR_ROOT` defaults to this directory (see `scripts/env_setup.sh`); the
code appends `$BANDITPR_ROOT/src` to `sys.path`. Dataset files are resolved
via `LAMP_DATA_ROOT` (default `/scratch/weixuz/lamp_data`).
