# Main Experiment Results (Draft)

This document summarizes the **main experiments** (preference head detection + DPS inference) and the **baseline comparisons** using the latest aggregated evaluation results.

## Scope and Artifacts
- Tasks: LaMP-1/2/3/4/5/7 (dev split)
- Models: LLaMA3-8b-Instruct, Mistral-7B-Instruct-v0.3, Qwen2-7B-Instruct
- Methods: Baseline, ContextAwareDecoding (CAD), DeCoReVanilla, DPSWeightedSoft, DoLa
- Evaluation summary: `outputs/evaluation_summary_combined.json`

## Stage 0: Profile Embedding + Clustering
We embed user profiles and run k-means with a target group size of ~100 users. K is derived from dataset size per task. The table below summarizes cluster sizes used by the current runs (from `clusters.json`).

| Task | Dev samples | K (clusters) | Cluster size min/mean/max | Cluster file |
| --- | --- | --- | --- | --- |
| LaMP-1 | 2500 | 25 | 18/100.0/179 | `results/preference_head/cluster_runs/lamp1_k25/clusters.json` |
| LaMP-2 | 692 | 7 | 18/98.9/225 | `results/preference_head/cluster_runs/lamp2_k7/clusters.json` |
| LaMP-3 | 2500 | 25 | 1/100.0/242 | `results/preference_head/cluster_runs/lamp3_k25/clusters.json` |
| LaMP-4 | 1925 | 19 | 5/101.3/467 | `results/preference_head/cluster_runs/lamp4_k19/clusters.json` |
| LaMP-5 | 2500 | 25 | 18/100.0/155 | `results/preference_head/cluster_runs/lamp5_k25/clusters.json` |
| LaMP-7 | 1500 | 15 | 39/100.0/150 | `results/preference_head/cluster_runs/lamp7_k15/clusters.json` |

## Stage 1: Preference Head Detection (PCS)
We compute per-head PCS by ablation within each cluster, then select the top ~4% heads (40 heads for LLaMA3). Below are summary diagnostics for LaMP-1 head discovery (clustered and per-user analyses).

- Cluster-level head-set Jaccard overlap (LaMP-1 k): mean=0.5127, min=0.2903, max=0.7391 (`results/preference_head/visualizations/lamp1_k/headset_jaccard.csv`).
- Cluster-level PCS rank correlation (Spearman): mean=0.8833, min=0.7273, max=0.9763 (`results/preference_head/visualizations/lamp1_k/pcs_spearman.csv`).
- User-level head overlap (k=1, 50 users): min=0.0390, mean=0.1609, max=0.3559 (`results/preference_head/visualizations/lamp1_k1_users/user_jaccard_summary.txt`).
- Most frequent heads across 50 users (k=1):

| Rank | Layer | Head | Count |
| --- | --- | --- | --- |
| 1 | 28 | 18 | 50 |
| 2 | 30 | 18 | 50 |
| 3 | 29 | 18 | 47 |
| 4 | 31 | 5 | 40 |
| 5 | 2 | 15 | 39 |

### Head Detection Visualizations
- Cluster PCS heatmap grid: `results/preference_head/visualizations/lamp1_k/pcs_heatmap_grid.png`
- Cluster head-set overlap heatmap: `results/preference_head/visualizations/lamp1_k/headset_jaccard_heatmap.png`
- Cluster PCS Spearman heatmap: `results/preference_head/visualizations/lamp1_k/pcs_spearman_heatmap.png`
- User PCS grid (9 users): `results/preference_head/visualizations/lamp1_k1_users_9/pcs_heatmap_grid.png`
- User PCS Spearman (9 users): `results/preference_head/visualizations/lamp1_k1_users_9_spearman/pcs_spearman_heatmap.png`
- User PCS Spearman (50 users): `results/preference_head/visualizations/lamp1_k1_users_50_spearman/pcs_spearman_heatmap.png`

## Stage 2: DPS and Baselines (Main Results)
For each task, we report task-specific metrics. Missing entries indicate runs that did not finish or are absent from the summary. **Note:** rows with small sample counts (e.g., 50 or 2) are quick sanity/partial runs and should not be compared directly to full-dev results.

### LaMP-1
| Model | Method | Metrics | Samples |
| --- | --- | --- | --- |
| LLaMA3-8b-Instruct | Baseline | acc=0.6240, f1=0.6070 | 2500 |
| LLaMA3-8b-Instruct | ContextAwareDecoding | acc=0.6240, f1=0.6070 | 2500 |
| LLaMA3-8b-Instruct | DeCoReVanilla | acc=0.6232, f1=0.6200 | 2500 |
| LLaMA3-8b-Instruct | DPSWeightedSoft | acc=0.6356, f1=0.6288 | 2500 |
| LLaMA3-8b-Instruct | DoLa | acc=0.6156, f1=0.5961 | 2500 |
| Mistral-7B-Instruct-v0.3 | Baseline | acc=0.0004, f1=0.0008 | 2500 |
| Mistral-7B-Instruct-v0.3 | ContextAwareDecoding | acc=0.0000, f1=0.0000 | 50 |
| Mistral-7B-Instruct-v0.3 | DeCoReVanilla | acc=0.0000, f1=0.0000 | 50 |
| Mistral-7B-Instruct-v0.3 | DPSWeightedSoft | acc=0.0008, f1=0.0016 | 2500 |
| Mistral-7B-Instruct-v0.3 | DoLa | acc=0.0000, f1=0.0000 | 50 |
| Qwen2-7B-Instruct | Baseline | acc=1.0000, f1=1.0000 | 2 |
| Qwen2-7B-Instruct | ContextAwareDecoding | acc=0.6000, f1=0.6250 | 50 |
| Qwen2-7B-Instruct | DeCoReVanilla | acc=0.5400, f1=0.5891 | 50 |
| Qwen2-7B-Instruct | DPSWeightedSoft | acc=0.6932, f1=0.7078 | 2500 |
| Qwen2-7B-Instruct | DoLa | acc=0.6800, f1=0.6795 | 50 |

### LaMP-2
| Model | Method | Metrics | Samples |
| --- | --- | --- | --- |
| LLaMA3-8b-Instruct | Baseline | acc=0.4552, f1=0.3839 | 692 |
| LLaMA3-8b-Instruct | ContextAwareDecoding | acc=0.4552, f1=0.3839 | 692 |
| LLaMA3-8b-Instruct | DeCoReVanilla | acc=0.4639, f1=0.4034 | 692 |
| LLaMA3-8b-Instruct | DPSWeightedSoft | acc=0.4610, f1=0.3910 | 692 |
| LLaMA3-8b-Instruct | DoLa | acc=0.2800, f1=0.1643 | 50 |
| Mistral-7B-Instruct-v0.3 | Baseline | **missing** | - |
| Mistral-7B-Instruct-v0.3 | ContextAwareDecoding | acc=0.0200, f1=0.0222 | 50 |
| Mistral-7B-Instruct-v0.3 | DeCoReVanilla | acc=0.0200, f1=0.0222 | 50 |
| Mistral-7B-Instruct-v0.3 | DPSWeightedSoft | acc=0.0939, f1=0.1487 | 692 |
| Mistral-7B-Instruct-v0.3 | DoLa | acc=0.0200, f1=0.0222 | 50 |
| Qwen2-7B-Instruct | Baseline | acc=0.0000, f1=0.0000 | 2 |
| Qwen2-7B-Instruct | ContextAwareDecoding | acc=0.1800, f1=0.1181 | 50 |
| Qwen2-7B-Instruct | DeCoReVanilla | acc=0.2000, f1=0.1217 | 50 |
| Qwen2-7B-Instruct | DPSWeightedSoft | acc=0.3902, f1=0.3202 | 692 |
| Qwen2-7B-Instruct | DoLa | acc=0.2000, f1=0.0958 | 50 |

### LaMP-3
| Model | Method | Metrics | Samples |
| --- | --- | --- | --- |
| LLaMA3-8b-Instruct | Baseline | mae=0.4426, rmse=0.9300 | 2500 |
| LLaMA3-8b-Instruct | ContextAwareDecoding | mae=0.4426, rmse=0.9300 | 2500 |
| LLaMA3-8b-Instruct | DeCoReVanilla | mae=0.4442, rmse=0.9458 | 2500 |
| LLaMA3-8b-Instruct | DPSWeightedSoft | **missing** | - |
| LLaMA3-8b-Instruct | DoLa | mae=0.4000, rmse=0.8718 | 50 |
| Mistral-7B-Instruct-v0.3 | Baseline | **missing** | - |
| Mistral-7B-Instruct-v0.3 | ContextAwareDecoding | mae=0.5400, rmse=1.1916 | 50 |
| Mistral-7B-Instruct-v0.3 | DeCoReVanilla | mae=0.4200, rmse=0.9695 | 50 |
| Mistral-7B-Instruct-v0.3 | DPSWeightedSoft | **missing** | - |
| Mistral-7B-Instruct-v0.3 | DoLa | mae=0.5000, rmse=1.0677 | 50 |
| Qwen2-7B-Instruct | Baseline | mae=0.0000, rmse=0.0000 | 2 |
| Qwen2-7B-Instruct | ContextAwareDecoding | mae=0.3200, rmse=0.6000 | 50 |
| Qwen2-7B-Instruct | DeCoReVanilla | mae=0.3200, rmse=0.6325 | 50 |
| Qwen2-7B-Instruct | DPSWeightedSoft | mae=0.3276, rmse=0.6719 | 2500 |
| Qwen2-7B-Instruct | DoLa | mae=0.3200, rmse=0.6000 | 50 |

### LaMP-4
| Model | Method | Metrics | Samples |
| --- | --- | --- | --- |
| LLaMA3-8b-Instruct | Baseline | r1=0.1681, rL=0.1498, meteor=0.1568 | 1925 |
| LLaMA3-8b-Instruct | ContextAwareDecoding | r1=0.1681, rL=0.1498, meteor=0.1568 | 1925 |
| LLaMA3-8b-Instruct | DeCoReVanilla | r1=0.1768, rL=0.1572, meteor=0.1626 | 1925 |
| LLaMA3-8b-Instruct | DPSWeightedSoft | r1=0.1686, rL=0.1495, meteor=0.1550 | 1925 |
| LLaMA3-8b-Instruct | DoLa | r1=0.1694, rL=0.1508, meteor=0.1592 | 1925 |
| Mistral-7B-Instruct-v0.3 | Baseline | **missing** | - |
| Mistral-7B-Instruct-v0.3 | ContextAwareDecoding | r1=0.1361, rL=0.1132, meteor=0.0980 | 50 |
| Mistral-7B-Instruct-v0.3 | DeCoReVanilla | r1=0.1299, rL=0.1085, meteor=0.0908 | 50 |
| Mistral-7B-Instruct-v0.3 | DPSWeightedSoft | r1=0.1537, rL=0.1362, meteor=0.1399 | 1925 |
| Mistral-7B-Instruct-v0.3 | DoLa | r1=0.1362, rL=0.1136, meteor=0.0962 | 50 |
| Qwen2-7B-Instruct | Baseline | r1=0.0000, rL=0.0000, meteor=0.0000 | 2 |
| Qwen2-7B-Instruct | ContextAwareDecoding | r1=0.1680, rL=0.1492, meteor=0.1255 | 50 |
| Qwen2-7B-Instruct | DeCoReVanilla | r1=0.1581, rL=0.1305, meteor=0.1232 | 50 |
| Qwen2-7B-Instruct | DPSWeightedSoft | r1=0.1529, rL=0.1347, meteor=0.1318 | 1925 |
| Qwen2-7B-Instruct | DoLa | r1=0.1642, rL=0.1473, meteor=0.1272 | 50 |

### LaMP-5
| Model | Method | Metrics | Samples |
| --- | --- | --- | --- |
| LLaMA3-8b-Instruct | Baseline | r1=0.3530, rL=0.3068, meteor=0.3925 | 2500 |
| LLaMA3-8b-Instruct | ContextAwareDecoding | r1=0.3530, rL=0.3068, meteor=0.3925 | 2500 |
| LLaMA3-8b-Instruct | DeCoReVanilla | r1=0.4010, rL=0.3527, meteor=0.4004 | 2500 |
| LLaMA3-8b-Instruct | DPSWeightedSoft | r1=0.3242, rL=0.2785, meteor=0.3826 | 2500 |
| LLaMA3-8b-Instruct | DoLa | r1=0.3636, rL=0.3117, meteor=0.4079 | 973 |
| Mistral-7B-Instruct-v0.3 | Baseline | **missing** | - |
| Mistral-7B-Instruct-v0.3 | ContextAwareDecoding | r1=0.4375, rL=0.3712, meteor=0.4561 | 50 |
| Mistral-7B-Instruct-v0.3 | DeCoReVanilla | r1=0.4135, rL=0.3648, meteor=0.4419 | 50 |
| Mistral-7B-Instruct-v0.3 | DPSWeightedSoft | r1=0.3984, rL=0.3352, meteor=0.4162 | 2500 |
| Mistral-7B-Instruct-v0.3 | DoLa | r1=0.4364, rL=0.3733, meteor=0.4605 | 50 |
| Qwen2-7B-Instruct | Baseline | r1=0.1389, rL=0.1389, meteor=0.0429 | 2 |
| Qwen2-7B-Instruct | ContextAwareDecoding | r1=0.4197, rL=0.3780, meteor=0.4381 | 50 |
| Qwen2-7B-Instruct | DeCoReVanilla | r1=0.4311, rL=0.3729, meteor=0.4565 | 50 |
| Qwen2-7B-Instruct | DPSWeightedSoft | r1=0.4071, rL=0.3424, meteor=0.4230 | 2500 |
| Qwen2-7B-Instruct | DoLa | r1=0.4277, rL=0.3746, meteor=0.4596 | 50 |

### LaMP-7
| Model | Method | Metrics | Samples |
| --- | --- | --- | --- |
| LLaMA3-8b-Instruct | Baseline | r1=0.3368, rL=0.2893, meteor=0.2813 | 1500 |
| LLaMA3-8b-Instruct | ContextAwareDecoding | r1=0.3368, rL=0.2893, meteor=0.2813 | 1500 |
| LLaMA3-8b-Instruct | DeCoReVanilla | r1=0.3231, rL=0.2764, meteor=0.2729 | 1500 |
| LLaMA3-8b-Instruct | DPSWeightedSoft | r1=0.3186, rL=0.2698, meteor=0.2684 | 1500 |
| LLaMA3-8b-Instruct | DoLa | r1=0.3365, rL=0.2877, meteor=0.2795 | 1500 |
| Mistral-7B-Instruct-v0.3 | Baseline | **missing** | - |
| Mistral-7B-Instruct-v0.3 | ContextAwareDecoding | r1=0.3342, rL=0.2916, meteor=0.3097 | 50 |
| Mistral-7B-Instruct-v0.3 | DeCoReVanilla | r1=0.3407, rL=0.2927, meteor=0.2978 | 50 |
| Mistral-7B-Instruct-v0.3 | DPSWeightedSoft | r1=0.3242, rL=0.2797, meteor=0.2890 | 1500 |
| Mistral-7B-Instruct-v0.3 | DoLa | r1=0.3291, rL=0.2852, meteor=0.3026 | 50 |
| Qwen2-7B-Instruct | Baseline | r1=0.1538, rL=0.1538, meteor=0.0538 | 2 |
| Qwen2-7B-Instruct | ContextAwareDecoding | r1=0.3590, rL=0.3106, meteor=0.3384 | 50 |
| Qwen2-7B-Instruct | DeCoReVanilla | r1=0.3470, rL=0.3065, meteor=0.3173 | 50 |
| Qwen2-7B-Instruct | DPSWeightedSoft | r1=0.3432, rL=0.2978, meteor=0.3169 | 1500 |
| Qwen2-7B-Instruct | DoLa | r1=0.3524, rL=0.3046, meteor=0.3246 | 50 |

## Main Result Visualizations
Primary-metric bar charts (per task, per model) were generated from `outputs/evaluation_summary_combined.json`:
- `decore/plots/main_results/lamp1_primary_metric.png` (Accuracy)
- `decore/plots/main_results/lamp2_primary_metric.png` (Accuracy)
- `decore/plots/main_results/lamp3_primary_metric.png` (MAE; lower is better)
- `decore/plots/main_results/lamp4_primary_metric.png` (ROUGE-L)
- `decore/plots/main_results/lamp5_primary_metric.png` (ROUGE-L)
- `decore/plots/main_results/lamp7_primary_metric.png` (ROUGE-L)

## Missing Results (from summary)
- LaMP-2 | Mistral-7B-Instruct-v0.3 | Baseline
- LaMP-3 | LLaMA3-8b-Instruct | DPSWeightedSoft
- LaMP-3 | Mistral-7B-Instruct-v0.3 | Baseline
- LaMP-3 | Mistral-7B-Instruct-v0.3 | DPSWeightedSoft
- LaMP-4 | Mistral-7B-Instruct-v0.3 | Baseline
- LaMP-5 | Mistral-7B-Instruct-v0.3 | Baseline
- LaMP-7 | Mistral-7B-Instruct-v0.3 | Baseline
