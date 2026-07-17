# Cluster-Aware Preference Head Detection and Weighted Decoding with Preference Suppression

## Abstract

This document describes a methodology for mitigating over-personalization in large language models (LLMs) through cluster-aware preference head detection and weighted decoding. The approach identifies attention heads that causally encode user-specific preferences, groups users by profile similarity, and applies targeted suppression during decoding to balance personalization with factual accuracy.

---

## 1. Introduction and Motivation

Large language models fine-tuned or prompted with user profiles can exhibit *over-personalization*—generating outputs that excessively mirror user preferences at the cost of factual accuracy or diversity. This work proposes a method to:

1. Identify **preference heads**: attention heads that disproportionately encode and inject user-specific stylistic patterns
2. Cluster users by **profile similarity** to capture diverse preference patterns
3. Apply **weighted preference suppression** during decoding using cluster-specific head importance scores

The method builds on the Decoding by Contrasting Retrieval Heads (DeCoRe) framework, extending it to personalization tasks.

---

## 2. Problem Formulation

### 2.1 Setup

Let \(\mathcal{M}\) be a transformer-based LLM with:
- \(L\) layers
- \(H\) attention heads per layer
- Total heads: \(L \times H\)

Given a user \(u\) with profile \(\mathcal{P}_u = \{p_1, p_2, \ldots, p_M\}\) (a set of historical items), and a query \(q\), the model generates a response:

\[
y = \mathcal{M}(q \mid \mathcal{P}_u)
\]

### 2.2 Objective

Identify a subset of attention heads \(\mathcal{H}^* \subseteq \{(l, h) : l \in [L], h \in [H]\}\) that are primarily responsible for injecting user preferences, and selectively suppress their influence during generation to reduce over-personalization while preserving task performance.

---

## 3. Profile Embedding and User Clustering

### 3.1 Profile Text Construction

For each user sample \(i\), we construct a textual representation of their profile by concatenating content from their historical items. Given profile items \(\{p_1, \ldots, p_M\}\), we extract text using a priority hierarchy:

\[
\text{text}(p_j) = 
\begin{cases}
p_j.\texttt{text} & \text{if exists} \\
p_j.\texttt{title} \oplus p_j.\texttt{abstract} \oplus p_j.\texttt{description} \oplus p_j.\texttt{content} \oplus p_j.\texttt{summary} \oplus p_j.\texttt{reviewText} & \text{otherwise}
\end{cases}
\]

If none of the preferred fields are available, we fall back to concatenating any non-empty string fields (excluding IDs).

where \(\oplus\) denotes string concatenation. We limit to the first \(M_{\max}\) profiles (default: 5) to bound computation:

\[
\mathbf{t}_i = \text{text}(p_1) \oplus \text{text}(p_2) \oplus \cdots \oplus \text{text}(p_{M_{\max}})
\]

### 3.2 Sentence Embedding

We encode profile texts using a pre-trained sentence transformer \(\phi\) (e.g., `all-MiniLM-L6-v2`). For a text sequence \(\mathbf{t}_i\) tokenized to \(T\) tokens with hidden states \(\mathbf{H} \in \mathbb{R}^{T \times d}\) and attention mask \(\mathbf{m} \in \{0,1\}^T\):

**Mean Pooling:**
\[
\mathbf{e}_i = \frac{\sum_{t=1}^{T} \mathbf{m}_t \cdot \mathbf{h}_t}{\sum_{t=1}^{T} \mathbf{m}_t}
\]

**Optional L2 Normalization:**
\[
\tilde{\mathbf{e}}_i = \frac{\mathbf{e}_i}{\|\mathbf{e}_i\|_2}
\]

This yields an embedding matrix \(\mathbf{E} \in \mathbb{R}^{N \times d}\) for \(N\) users.

### 3.3 K-Means Clustering

We partition users into \(K\) clusters using k-means with k-means++ initialization. The number of clusters is computed per task to maintain approximately 100 users per cluster:

\[
K = \text{round}\left(\frac{N_{\text{dev}}}{100}\right)
\]

We clamp \(K\) to \([1, N_{\text{dev}}]\) to avoid degenerate values.

**K-Means++ Initialization:**

1. Select first centroid uniformly: \(\boldsymbol{\mu}_1 = \mathbf{e}_{i_1}\) where \(i_1 \sim \text{Uniform}(1, N)\)

2. For \(c = 2, \ldots, K\):
   - Compute squared distance to nearest existing centroid:
     \[
     D^2(i) = \min_{j < c} \|\mathbf{e}_i - \boldsymbol{\mu}_j\|_2^2
     \]
   - Sample next centroid with probability proportional to \(D^2(i)\):
     \[
     \boldsymbol{\mu}_c = \mathbf{e}_{i_c} \quad \text{where} \quad P(i_c = i) = \frac{D^2(i)}{\sum_{i'} D^2(i')}
     \]

**Lloyd's Algorithm:**

Iterate until convergence (centroid shift \(< \epsilon\)):

1. **Assignment:** 
   \[
   z_i = \arg\min_{k \in [K]} \|\mathbf{e}_i - \boldsymbol{\mu}_k\|_2^2
   \]

2. **Update:**
   \[
   \boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i : z_i = k} \mathbf{e}_i
   \]

where \(C_k = \{i : z_i = k\}\) is the set of users assigned to cluster \(k\).

**Outputs:**
- Cluster assignments: \(\mathbf{z} \in [K]^N\)
- Centroids: \(\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K \in \mathbb{R}^d\)

---

## 4. Preference Head Detection via Ablation

### 4.1 Preference Contribution Score (PCS)

We identify preference-encoding heads through causal intervention. The key insight is that ablating (zeroing) an important preference head should degrade the model's ability to generate personalized outputs.

**Definition (Preference Contribution Score):**

For head \((l, h)\) and sample \(i\) with prompt \(x_i\) (including profile) and target \(y_i\):

\[
\text{PCS}_{l,h}^{(i)} = \mathcal{L}_{\text{ablated}}^{(l,h)}(x_i, y_i) - \mathcal{L}_{\text{baseline}}(x_i, y_i)
\]

where:
- \(\mathcal{L}_{\text{baseline}}\) is the negative log-likelihood (NLL) with the unmodified model
- \(\mathcal{L}_{\text{ablated}}^{(l,h)}\) is the NLL when head \((l,h)\) is ablated

**Interpretation:** A positive PCS indicates that removing the head *increases* loss, meaning the head contributes positively to predicting the personalized target. Higher PCS = more important for personalization.

### 4.2 Head Ablation Mechanism

Ablation is performed by hooking the output projection (\(\mathbf{W}_O\)) of the target attention layer. For a transformer layer with multi-head attention:

\[
\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O
\]

where each head computes:
\[
\text{head}_h = \text{Softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}}\right) \mathbf{V}_h
\]

**Ablation Hook:** After the \(\mathbf{W}_O\) projection, we reshape the output to isolate heads:

\[
\mathbf{O} \in \mathbb{R}^{B \times T \times D} \rightarrow \mathbf{O}' \in \mathbb{R}^{B \times T \times H \times d_h}
\]

Then zero out head \(h^*\):
\[
\mathbf{O}'[:, :, h^*, :] = \mathbf{0}
\]

And reshape back:
\[
\mathbf{O}' \rightarrow \tilde{\mathbf{O}} \in \mathbb{R}^{B \times T \times D}
\]

### 4.3 NLL Computation

For prompt tokens \(\mathbf{x} = (x_1, \ldots, x_P)\) and target tokens \(\mathbf{y} = (y_1, \ldots, y_T)\):

\[
\mathcal{L}(\mathbf{x}, \mathbf{y}) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(y_t \mid \mathbf{x}, y_{<t})
\]

where \(P_\theta\) is the model's next-token distribution.

### 4.4 Aggregated PCS

For each head, we compute PCS across a sample set \(\mathcal{S}\) and aggregate:

\[
\overline{\text{PCS}}_{l,h} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \text{PCS}_{l,h}^{(i)}
\]

---

## 5. Per-Cluster Head Detection

### 5.1 Motivation

Different user clusters may rely on different preference heads. A head important for encoding "formal academic writing" preferences may differ from one encoding "casual conversational" preferences.

### 5.2 Cluster-Specific Detection

For each cluster \(k\):

1. **Select cluster samples:** \(\mathcal{S}_k = \{i : z_i = k\}\)
2. **Run ablation study** on \(\mathcal{S}_k\) (or a subsample for efficiency)
3. **Compute cluster-specific PCS:**
   \[
   \overline{\text{PCS}}_{l,h}^{(k)} = \frac{1}{|\mathcal{S}_k|} \sum_{i \in \mathcal{S}_k} \text{PCS}_{l,h}^{(i)}
   \]

In practice, we cap the number of samples per cluster with `--num_samples`, and
the ablation loop uses a per-head subsample of size \(\min(50, \text{num\_samples})\)
for efficiency.

### 5.3 Head Weight Computation

From PCS scores, we compute normalized head weights for each cluster.

**Step 1: Top-K Selection**

Keep only the top \(\rho\)% of heads by PCS (e.g., \(\rho = 4\%\)):
\[
\mathcal{H}_k = \text{Top}_{\lfloor \rho \cdot L \cdot H \rfloor}\left(\{(l,h) : \overline{\text{PCS}}_{l,h}^{(k)}\}\right)
\]

**Step 2: Thresholding**

Apply minimum PCS threshold \(\tau_{\min}\):
\[
w_{l,h}^{(k)} = 
\begin{cases}
\max(\overline{\text{PCS}}_{l,h}^{(k)}, 0) & \text{if } (l,h) \in \mathcal{H}_k \text{ and } \overline{\text{PCS}}_{l,h}^{(k)} \geq \tau_{\min} \\
0 & \text{otherwise}
\end{cases}
\]

**Step 3: Normalization**

Normalize weights (options: max, sum, or none):

- **Max normalization:** \(\tilde{w}_{l,h}^{(k)} = \frac{w_{l,h}^{(k)}}{\max_{l',h'} w_{l',h'}^{(k)}}\)
- **Sum normalization:** \(\tilde{w}_{l,h}^{(k)} = \frac{w_{l,h}^{(k)}}{\sum_{l',h'} w_{l',h'}^{(k)}}\)

**Step 4: Power Scaling (Optional)**

Apply power transformation for sharpening/smoothing:
\[
\hat{w}_{l,h}^{(k)} = \left(\tilde{w}_{l,h}^{(k)}\right)^\gamma
\]

where \(\gamma > 1\) sharpens (emphasizes top heads) and \(\gamma < 1\) smooths.

**Output:** Per-cluster head weight matrices \(\mathbf{W}^{(k)} \in \mathbb{R}^{L \times H}\) for \(k = 1, \ldots, K\).

---

## 6. Cluster Routing

At inference, we route each sample to clusters based on its profile embedding similarity to cluster centroids.

### 6.1 Distance Computation

For a test sample with profile embedding \(\mathbf{e}\):
\[
d_k = \|\mathbf{e} - \boldsymbol{\mu}_k\|_2 \quad \text{for } k = 1, \ldots, K
\]

### 6.2 Hard Routing

Assign to nearest cluster:
\[
p_k = 
\begin{cases}
1 & \text{if } k = \arg\min_{k'} d_{k'} \\
0 & \text{otherwise}
\end{cases}
\]

### 6.3 Soft Routing

Compute softmax probabilities over negative distances:
\[
p_k = \frac{\exp(-d_k / \tau)}{\sum_{k'=1}^{K} \exp(-d_{k'} / \tau)}
\]

where \(\tau > 0\) is a temperature parameter:
- \(\tau \to 0\): approaches hard routing
- \(\tau \to \infty\): uniform distribution over clusters

---

## 7. Weighted Head Importance

### 7.1 Aggregated Head Weights

Given cluster probabilities \(\mathbf{p} = (p_1, \ldots, p_K)\) and per-cluster weights \(\mathbf{W}^{(k)}\):

\[
\mathbf{W}_{\text{agg}} = \sum_{k=1}^{K} p_k \cdot \mathbf{W}^{(k)}
\]

This is a weighted average of cluster-specific head importance scores.

### 7.2 Head Scale Computation

Convert importance to suppression scale (higher importance = more suppression):

\[
\mathbf{S} = \text{clip}(1 - \mathbf{W}_{\text{agg}}, 0, 1)
\]

where \(S_{l,h} \in [0, 1]\):
- \(S_{l,h} = 1\): no suppression (head operates normally)
- \(S_{l,h} = 0\): full suppression (head output zeroed)

---

## 8. Weighted Decoding with Preference Suppression (Weighted DPS)

### 8.1 Dual-Pass Generation

Weighted DPS performs two forward passes per token:

1. **Base pass:** Normal generation with full model
2. **Depersonalized pass:** Generation with preference heads scaled down

### 8.2 Head Scaling Mechanism

During the depersonalized pass, we hook the query projection (\(\mathbf{W}_Q\)) output:

For layer \(l\), reshape query output:
\[
\mathbf{Q} \in \mathbb{R}^{B \times T \times D} \rightarrow \mathbf{Q}' \in \mathbb{R}^{B \times T \times H \times d_h}
\]

Apply per-head scaling:
\[
\mathbf{Q}'[:, :, h, :] = S_{l,h} \cdot \mathbf{Q}'[:, :, h, :]
\]

Reshape back:
\[
\mathbf{Q}' \rightarrow \tilde{\mathbf{Q}} \in \mathbb{R}^{B \times T \times D}
\]

This effectively reduces the attention contribution of preference-encoding heads.

### 8.3 Contrastive Decoding

At each generation step \(t\), we obtain:
- Base logits: \(\ell_t^{\text{base}} = \log P_\theta(y_t \mid \mathbf{x}, y_{<t})\)
- Depersonalized logits: \(\ell_t^{\text{dep}} = \log P_{\theta,\mathbf{S}}(y_t \mid \mathbf{x}, y_{<t})\)

**Entropy-Based Mixing:**

Compute entropy of base distribution:
\[
H_t = -\sum_{v} P_\theta^{\text{base}}(v) \log P_\theta^{\text{base}}(v)
\]

Optionally normalize by vocabulary size \(V\):
\[
\alpha_t = \frac{H_t}{\log V} \in [0, 1]
\]

Apply optional cap: \(\alpha_t = \min(\alpha_t, \alpha_{\max})\)

**Contrastive Logits:**
\[
\ell_t^{\text{final}} = (1 + \alpha_t) \cdot \ell_t^{\text{base}} - \alpha_t \cdot \ell_t^{\text{dep}}
\]

**Intuition:**
- High entropy (uncertain base prediction) → higher \(\alpha_t\) → stronger contrast
- Low entropy (confident base prediction) → lower \(\alpha_t\) → trust base model

### 8.4 Token Selection

Select next token (current implementation uses greedy decoding):
\[
y_t = \arg\max_v \ell_t^{\text{final}}(v)
\]

---

## 9. Algorithm Summary

### Algorithm 1: Cluster-Aware Preference Head Detection

```
Input: Dataset D_dev, LLM M, top_percent ρ
Output: Per-cluster head weights {W^(k)}, centroids μ

1. Embed profiles: E = Φ(D_dev)
2. Compute K = round(|D_dev| / 100)  // ~100 users per cluster (clamped to [1, N])
3. Cluster: z, μ = KMeans(E, K)
4. For each cluster k = 1 to K:
   a. S_k = {i : z_i = k}
   b. For each head (l, h):
      - Compute PCS via ablation on S_k
   c. Select top ρ% heads by PCS
   d. Normalize to get W^(k)
5. Return {W^(k)}, μ
```

### Algorithm 2: Weighted DPS Inference

```
Input: Query x, profile P, centroids μ (from dev), weights {W^(k)}, 
       dev embeddings E_dev, sample index i, LLM M
Output: Generated text y

1. Look up pre-computed dev embedding: e = E_dev[i]
2. Compute cluster probs: p_k = softmax(-||e - μ_k|| / τ)
3. Aggregate weights: W_agg = Σ_k p_k · W^(k)
4. Compute head scale: S = clip(1 - W_agg, 0, 1)
5. For each token t:
   a. Base forward: ℓ_base = M(x, y_{<t})
   b. Scaled forward: ℓ_dep = M_S(x, y_{<t})  // with head scaling
   c. Compute α_t from entropy of ℓ_base
   d. Contrast: ℓ_t = (1 + α_t)·ℓ_base - α_t·ℓ_dep
   e. Select: y_t = argmax(ℓ_t)
6. Return y
```

**Note:** We pre-compute `embeddings_dev.npy` for routing; clustering and evaluation both use the dev split.

---

## 10. Pipeline Outputs

### 10.1 Clustering Stage

| Output | Format | Description |
|--------|--------|-------------|
| `clusters.json` | JSON | Cluster assignments, centroids, metadata (task, k, embedding model) |
| `embeddings.npy` | NumPy | Dev profile embeddings (used for clustering) |
| `embeddings_dev.npy` | NumPy | Dev profile embeddings (computed separately for routing) |

### 10.2 Head Detection Stage (per cluster)

| Output | Format | Description |
|--------|--------|-------------|
| `cluster_XX/head_weights.json` | JSON | Normalized head weights for cluster XX |
| `cluster_XX/head_weights.npy` | NumPy | Weight matrix \(\mathbf{W}^{(k)} \in \mathbb{R}^{L \times H}\) |
| `cluster_XX/*_pcs.json` | JSON | Raw PCS scores per head |
| `cluster_XX/*_ranked.json` | JSON | Heads ranked by average PCS |

### 10.3 Inference Stage

| Output | Format | Description |
|--------|--------|-------------|
| `pred_<task>_<model>__<method>.json` | JSONL | Per-sample predictions with alphas |

---

## 11. Computational Complexity

| Component | Time Complexity | Notes |
|-----------|-----------------|-------|
| Profile embedding | \(O(N \cdot T \cdot d)\) | One-time, parallelizable |
| K-means clustering | \(O(N \cdot K \cdot d \cdot I)\) | \(I\) = iterations |
| PCS detection (per cluster) | \(O(L \cdot H \cdot S \cdot F)\) | \(S\) = samples, \(F\) = forward cost |
| Weighted DPS inference | \(O(2 \cdot T \cdot F)\) | 2× forward passes per token |

---

## 12. Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Number of clusters | \(K\) | \(\text{round}(N_{\text{dev}} / 100)\) | Computed per task to target ~100 users per cluster (clamped to [1, N]) |
| Top percent heads | \(\rho\) | 0.04 | Fraction of heads to keep |
| Min PCS threshold | \(\tau_{\min}\) | 0.0 | Minimum PCS for inclusion |
| PCS power | \(\gamma\) | 1.0 | Sharpening exponent |
| Routing temperature | \(\tau\) | 1.0 | Softness of cluster routing |
| Alpha cap | \(\alpha_{\max}\) | None | Maximum contrastive weight |
| Max profiles | \(M_{\max}\) | 5 | Profiles per user for embedding |

### 12.1 Adaptive Cluster Count

The number of clusters \(K\) is computed dynamically based on dev set size:

\[
K = \text{round}\left(\frac{N_{\text{dev}}}{100}\right)
\]

This ensures approximately 100 users per cluster regardless of dataset size, providing:
- Sufficient samples per cluster for reliable PCS estimation
- Balanced granularity across tasks of different scales

### 12.2 Experimental Sweeps (LaMP-1, LLaMA3-8B)

These ablations are implemented as single sbatch scripts under
`experiments/hparam_scripts/` and use the dev split throughout.

- **Head-count sweep** (`run_llama3_lamp1_heads_sweep.sh`):
  - Heads: 10/20/40/80/160
  - Implementation: \(\rho = \text{num\_heads} / 1024\)
  - Outputs: `outputs/hparam/heads/h<num_heads>`
- **DPS alpha sweep** (`run_llama3_lamp1_gamma_sweep.sh`):
  - Adaptive alpha: entropy-based with `scale_alpha`
  - Fixed alpha: \(\alpha = 0.5\)
  - 40-head detection (`top_percent=0.04`)
- **Group-size sweep** (`run_llama3_lamp1_groupsize_sweep.sh`):
  - Target group sizes: 10/50/100/200/400
  - For target_group=10 only: reduced `num_samples=25` and chunked head detection
    (25 clusters per chunk) to keep runtime manageable.
- **Random-head ablations** (`run_llama3_lamp1_ablation_random_heads.sh`):
  - Randomized weights: shuffle head weights per cluster
  - Random masks (true masking): binary masks with the same head count
  - Outputs: `outputs/hparam/ablation/random_heads` and
    `outputs/hparam/ablation/random_mask`
  - Validation uses aggregated cluster heads + `validate_preference_heads.py`

---

## 13. Dataset Split Usage

| Split | Usage |
|-------|-------|
| **Dev/Test** | Profile clustering, per-cluster head detection, head weight computation, evaluation |

### 13.1 Dataset Sizes and Computed \(K\)

#### LaMP Benchmark

| Task | Dev | \(K = \text{round}(N/100)\) |
|------|-----|----------------------------|
| LaMP-1 | 2,500 | 25 |
| LaMP-2 | 692 | 7 |
| LaMP-3 | 2,500 | 25 |
| LaMP-4 | 1,925 | 19 |
| LaMP-5 | 2,500 | 25 |
| LaMP-7 | 1,500 | 15 |

#### LongLaMP Benchmark

| Task | Dev | \(K = \text{round}(N/100)\) |
|------|-----|----------------------------|
| LongLaMP-2 | 4,560 | 46 |
| LongLaMP-3 | 2,452 | 25 |
| LongLaMP-4 | 1,822 | 18 |

**Note:** LongLaMP uses the HuggingFace `test` split as `dev` when the dev split is unavailable.

### 13.2 Split Alignment Requirement

**Critical:** The evaluation embeddings must match the evaluation split length. Since clustering is performed on dev:

1. Compute **dev embeddings** → used for clustering and centroid computation
2. Write `embeddings_dev.npy` (can be re-computed or reused from clustering) → used for routing
3. Use dev `clusters.json` (with centroids) and per-cluster head weights
4. Pass dev embeddings via `--embeddings_file` during weighted DPS inference

The dev embeddings file is expected at:
```
cluster_runs/<task>_k<K>/embeddings_dev.npy
```

---

## 14. Relationship to DeCoRe

This method extends **DeCoRe** (Decoding by Contrasting Retrieval Heads) which was designed for factual retrieval:

| Aspect | DeCoRe | This Work |
|--------|--------|-----------|
| Target | Retrieval heads (factual grounding) | Preference heads (personalization) |
| Head detection | Based on retrieval attention patterns | Based on NLL ablation (PCS) |
| Clustering | None | User profile clustering |
| Head weights | Binary (use/don't use) | Continuous, cluster-specific |
| Routing | None | Soft/hard cluster routing |

The contrastive decoding mechanism is shared, but adapted for personalization suppression.

---

## 15. References

- DeCORE: Decoding by Contrasting Retrieval Heads to Mitigate Hallucination
- LaMP: Language Model Personalization Benchmark
- K-Means++ Initialization (Arthur & Vassilvitskii, 2007)
- Contrastive Decoding (Li et al., 2022)
