# Project: Detecting Preference Heads in LLMs

## Goal
Identify **Preference Heads** — attention heads in a transformer that causally encode and inject **user-specific preferences** (style, tone, vocabulary) into model outputs.  
This supports **Differential Preference Steering (DPS)** for personalized text generation.

---

## Methodology

### 1. Dataset
- Use prompts with **(user profile, query, reference completion)**.
- Example source: **LaMP benchmark** or any dataset with personalized references.

### 2. Baseline Score
- Define a **personalization metric**:
  - Negative log-likelihood of the reference (lower loss = better).
  - Or style similarity metrics (BLEU/ROUGE/METEOR, classifier-based style match).
- Compute baseline score `S(base)` on the dataset.

### 3. Head-wise Ablation (PCS)
For each layer `L` and head `H`:
1. Hook into the head’s **output projection (`o_proj`)**.
2. Zero out that head’s contribution during forward pass.
3. Recompute personalization score `S(M_{-h})`.
4. Define **Preference Contribution Score (PCS)**:
PCS(L,H) = S(base) - S(M_{-h})
(If using loss, flip sign so larger PCS = bigger harm when head removed.)

5. Save PCS for all heads.

### 4. Candidate Selection
- Rank heads by PCS.
- Select **top 2–6%** of heads (or those above a threshold).
- This gives the **global candidate set of preference heads**.

### 5. Activation Patching (Causal Test)
- Run model on:
- **Clean prompt** (with profile).
- **Corrupted prompt** (without profile).
- Cache head activations on clean run.
- In corrupted run, replace one head’s activation with cached clean value.
- If personalization improves (higher style similarity, Δlogits toward profile-consistent tokens), that head is causally carrying preference information.

### 6. Validation
- Mask only identified preference heads → personalization drops significantly.
- Mask random heads → minimal effect.
- Mask known retrieval/induction heads → little effect on personalization.
- Confirms **preference heads are distinct**.

### 7. Diagnostics (Optional)
- Track **Δlogits** for stylistic tokens during generation:
- Δ = logit_with_profile – logit_without_profile.
- Preference heads should modulate these differences.

---

## Implementation Notes
- Use **PyTorch forward hooks** on `self_attn.o_proj`.
- For ablation: reshape output to `(batch, seq_len, num_heads, head_dim)` and zero the target head.
- For patching: inject cached clean activation slice for the target head.
- Evaluate across **multiple users**:
- Stage A: Global discovery (≈400–800 examples across users).
- Stage B: Per-user calibration (≈20–50 examples per user).

---

## Deliverables
- PCS heatmap (layers × heads).
- Ranked list of candidate preference heads.
- Validation results: performance drop when masking candidate heads vs random heads.
- (Optional) Δlogits plots showing profile influence.

