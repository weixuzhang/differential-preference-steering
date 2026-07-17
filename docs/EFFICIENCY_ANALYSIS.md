# Efficiency Analysis (Draft)

This note estimates compute (FLOPs) by stage and compares DPS with baselines. Numbers are approximate and meant for relative comparisons.

## Assumptions

- Model: **LLaMA3‑8B** (L=32 layers, d=4096, d_ff=14336, heads=32).
- FLOPs counted as multiply‑adds (2 FLOPs per multiply‑add).
- Decode uses KV‑cache; prompt prefill uses full attention.
- Max prompt length P in {512, 1024, 2048}; max new tokens T=32.

## FLOPs formulas (decoder‑only transformer)

Per **layer**, per **decode token** with context length S (KV cache):
- Linear projections: `8 d^2 + 4 d d_ff`
- Attention matmuls: `4 d S`

Per **layer**, **prefill** for a full prompt of length P:
- Linear projections: `(8 d^2 + 4 d d_ff) * P`
- Attention matmuls: `4 d P^2`

Total per pass = per‑layer FLOPs * L.

## Example FLOPs (LLaMA3‑8B)

These are **single forward pass** costs for one sample:

| Prompt P | Prefill (TFlop) | Decode / token (TFlop) | Baseline total (prefill + T*decode) | DPS total (prefill + 2*T*decode) | DPS / Baseline |
| --- | --- | --- | --- | --- | --- |
| 512  | 6.18 | 0.0121 | 6.57 | 6.96 | 1.06× |
| 1024 | 12.64 | 0.0123 | 13.04 | 13.43 | 1.03× |
| 2048 | 26.39 | 0.0129 | 26.80 | 27.21 | 1.02× |

**Takeaway:** decode is 2× for DPS vs baseline, but prefill dominates when P is large. For shorter prompts or longer T, DPS overhead increases.

## Stage‑by‑stage compute (relative)

### 1) Profile embedding + clustering
- **Embedding**: one pass of MiniLM‑L6‑v2 per profile (small vs LLM).
- **Clustering**: k‑means `O(N * K * d_emb * iters)`, negligible vs LLM inference.

### 2) Preference head detection (per cluster)
From `preference_head_detection.py`:
- Baseline NLL: `N` forward passes
- Head ablation: `H * S` forward passes
  - `H = L * heads = 1024`
  - `S = min(50, N)`

Total passes per cluster: `N + 1024 * min(50, N)`.

**Example (LaMP‑1 dev, target group ~100):**
- N=100, S=50 → 51,300 forward passes per cluster.
- k≈25 clusters → ~**1.28M forward passes** for the task.

This stage dwarfs inference cost; it is the main compute bottleneck.

### 3) Routing + Weighted DPS inference (per sample)
From `run_weighted_dps.py`:
- **Prefill** once on full prompt.
- **Decode**: 2 passes per generated token (base + depersonalized).
- **Routing**: L2 distance to k centroids (`O(k * d_emb)`, tiny).
- **Head scaling**: elementwise multiply in q_proj (tiny).

### 4) Evaluation
- Metrics are CPU‑side and negligible vs model inference.

## Baseline compute comparison

**Decode‑time forward passes per generated token:**
- Baseline (greedy): **1×**
- DeCoRe‑vanilla: **2×** (base + block‑list pass)
- CAD: **2×** (base + no‑context pass) + **extra prefill** for no‑context prompt
- DPS‑Weighted: **2×** (base + head‑scaled pass) + routing overhead
- DoLa: **~1–1.5×** (single generate call, but extra layer logits; exact overhead depends on `dola_layers`)

## Performance vs overhead (empirical)

We plot a simple tradeoff view using **LLaMA3** dev results from `outputs/evaluation_summary_combined.json`. For each method, we compute the **average normalized primary metric** across LaMP‑1/2/3/4/5/7, where each task is normalized to its LLaMA3 baseline (baseline=1.0; MAE is inverted so higher is better). Compute multipliers are approximate decode‑time costs from the section above.

| Method | Relative compute | Avg normalized primary metric |
| --- | --- | --- |
| Baseline | 1.0× | 1.000 |
| ContextAwareDecoding | 2.1× | 1.000 |
| DeCoReVanilla | 2.0× | 1.028 |
| DPSWeightedSoft | 2.0× | 0.974 |
| DoLa | 1.2× | 0.954 |

Figure: `decore/plots/efficiency/efficiency_tradeoff.png`

## Observed runtime (from existing logs)

These are coarse estimates from tqdm in `.out` files (H100 nodes, LLaMA3):
- **Baseline NLL scoring** during head detection: ~**40–50 samples/s**
  - e.g. `llama3_lamp1_e2e_181404.out` shows ~47 it/s for 100 samples.
- **Head ablation**: ~**1.1 heads/s** for 50 samples per head
  - ~25s per 28‑head layer (seen in `qwen2_resume_181712.out`).

We do **not** currently log DPS decode throughput; can add timing hooks if you want (see below).

## Suggested quick timing probe (optional)

To measure per‑sample runtime on your hardware:

```
/usr/bin/time -v bash scripts/run_weighted_dps.py \
  --task LAMP_1 --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --cluster_file ... --cluster_heads_dir ... --embeddings_file ... \
  --num_samples 50 --run_dir /tmp/dps_timing
```

Then compute `elapsed_time / 50` for seconds per sample. Same idea for baselines using their run scripts.

---

If you want a scripted FLOPs calculator that auto‑reads model configs and dataset lengths, I can add it.
