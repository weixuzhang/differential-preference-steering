# Clustering, Head Detection, and Weighted DPS (Train/Dev Split)

This doc captures the current pipeline for clustering, preference head detection, and
weighted DPS, with an explicit train/dev split.

## Split usage
- dev: profile clustering, per-cluster head detection, head weight export, evaluation

## Dataset sizes (local cache)

### LaMP
| Task | Dev | Usage |
| --- | --- | --- |
| LaMP-1 | 2500 | dev=cluster/head detect/evaluate |
| LaMP-2 | 692 | dev=cluster/head detect/evaluate |
| LaMP-3 | 2500 | dev=cluster/head detect/evaluate |
| LaMP-4 | 1925 | dev=cluster/head detect/evaluate |
| LaMP-5 | 2500 | dev=cluster/head detect/evaluate |
| LaMP-7 | 1500 | dev=cluster/head detect/evaluate |

### LongLaMP
| Task | Dev | Usage |
| --- | --- | --- |
| LongLaMP-2 | 4560 | dev=cluster/head detect/evaluate |
| LongLaMP-3 | 2452 | dev=cluster/head detect/evaluate |
| LongLaMP-4 | 1822 | dev=cluster/head detect/evaluate |

Note: LongLaMP uses the HF `test` split when `dev` is missing; the processed data
is saved under the `dev` directory (see `banditpr/src/lamp/dataset.py`).

## Clustering (profile embeddings)
- Script: `preference_head/cluster_profiles.py`
- Data: `load_lamp_dataset(task, split=dev)`
- Profile text: use `text` when present; else concatenate `title` + `abstract` + `description`
  for up to `max_profiles` (default: 5).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` with mean pooling; optional L2 normalize.
- Clustering: k-means with k-means++ init. `k` is computed per task to target
  ~100 users per cluster: `k = round(dev_size / 100)` (see
  `preference_head/compute_k.py`).
- Outputs:
  - `clusters.json`: assignments, centroids, metadata
  - `embeddings.npy`: optional (used for routing)

## Head detection (PCS)
- Script: `preference_head/preference_head_detection.py` (global), and
  `preference_head/detect_cluster_heads.py` (per-cluster)
- Score: negative log-likelihood (cross-entropy) on target tokens.
- PCS definition: `PCS = ablated_nll - baseline_nll` (higher = more important).
- Ablation: zero one attention head by hooking `o_proj` in the target layer.
- Per-cluster weights:
  - Keep top `top_percent` heads by PCS.
  - Apply `pcs_norm` (`max`/`sum`/`none`), `pcs_min`, and `pcs_power`.
  - Export `head_weights.json` + `head_weights.npy` per cluster.
- Array runs: `preference_head/run_detect_cluster_heads.sh` reads
  `preference_head/cluster_head_manifest.tsv` built by
  `preference_head/build_cluster_heads_manifest.py`.

## Weighted DPS (soft routing)
- Script: `decore/scripts/run_weighted_dps.py`
- Inputs:
  - `clusters.json` (centroids, k)
  - per-cluster `head_weights.json`
  - `embeddings.npy` aligned to the evaluation split
- Routing:
  - hard: nearest centroid (argmin distance)
  - soft: softmax of `-distance/temperature`
- Weighted heads:
  - `weighted_heads = sum_k p_k * head_weights_k`
  - `head_scale = clip(1 - weighted_heads, 0, 1)`
- Application:
  - `head_scale` is applied to `q_proj` outputs in the depersonalized pass.
  - DPS combines base vs depersonalized logits using entropy-based alpha (optionally capped).

## Split alignment note (important)
Weighted DPS requires embeddings to match the evaluation split length
(`decore/src/datasets/lamp.py` uses `dev`). Since clustering and evaluation both
use `dev`, the embeddings align, but we still write `embeddings_dev.npy` for the
router input. The dev embeddings are expected at
`/scratch/weixuz/preference_head/cluster_runs/<task>_k<computed_k>/embeddings_dev.npy`.

## Hyperparameter + ablation experiments (LaMP-1, LLaMA3-8B)
These runs are implemented as single sbatch scripts (no arrays) under
`decore/hparam_scripts/`.

### Head-count sweep (top_percent)
- Script: `decore/hparam_scripts/run_llama3_lamp1_heads_sweep.sh`
- Heads: 10/20/40/80/160; `top_percent = num_heads / 1024`
- Head dirs: `preference_head/cluster_heads/lamp1_k<K>_llama3-8b-instruct_h<num_heads>`
- Outputs: `decore/outputs/hparam/heads/h<num_heads>`

### DPS alpha sweep (adaptive vs fixed)
- Script: `decore/hparam_scripts/run_llama3_lamp1_gamma_sweep.sh`
- Adaptive alpha: entropy-based with `--scale_alpha`
- Fixed alpha: `--alpha 0.5`
- Uses 40 heads (`top_percent=0.04`) and k from `target_group=100`

### Group-size sweep (target_group)
- Script: `decore/hparam_scripts/run_llama3_lamp1_groupsize_sweep.sh`
- Target group sizes: 10/50/100/200/400 (K computed from dev size)
- For target_group=10 only:
  - `NUM_SAMPLES_SMALL=25`
  - Detection is chunked in blocks of 25 clusters using `--cluster_start/--cluster_end`

### Random-head ablations
- Script: `decore/hparam_scripts/run_llama3_lamp1_ablation_random_heads.sh`
- Randomized weights (shuffle):
  - `preference_head/randomize_cluster_head_weights.py`
  - Output dir: `..._random`
  - Run: `decore/outputs/hparam/ablation/random_heads`
- Random masks (true masking):
  - `preference_head/random_mask_cluster_head_weights.py`
  - Output dir: `..._random_mask`
  - Run: `decore/outputs/hparam/ablation/random_mask`
- Head-quality validation uses an aggregated top-heads file:
  - `preference_head/aggregate_cluster_heads.py` + `validate_preference_heads.py`
