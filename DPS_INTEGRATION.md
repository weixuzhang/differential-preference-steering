# Differential Preference Steering (DPS) Integration

## Overview

**DPS (Differential Preference Steering)** is a personalization method that steers language model generation based on detected preference heads. Similar to DeCoRe but designed for personalization tasks, DPS uses attention heads that are most sensitive to user preferences to guide model outputs.

---

## Key Differences: DPS vs DeCoRe

| Aspect | DeCoRe | DPS |
|--------|---------|-----|
| **Purpose** | Context-aware decoding | Preference-based personalization |
| **Heads Used** | Retrieval heads (context-sensitive) | Preference heads (user-sensitive) |
| **Detection** | Based on context retrieval | Based on user preference patterns |
| **Tasks** | QA, fact retrieval | Personalized generation (LaMP) |
| **Steering** | Contextual reranking | Preference amplification |

---

## Architecture

### 1. Preference Head Detection

Preference heads are detected using the methodology from `preference_head/preference_head_detection.py`:

1. **Baseline Score Computation**: Compute NLL for each sample with all heads active
2. **Head-wise Ablation**: Test each head individually by masking it
3. **Preference Contribution Score (PCS)**: Calculate how much each head contributes to preference alignment
4. **Ranking**: Select top k% heads with highest PCS

**Output**: `Meta-Llama-3-8B-Instruct_LaMP_X_top_heads.json`

### 2. DPS Decoder

Located at: `src/models/dps.py`

**Key Components**:
- `_load_preference_heads()`: Loads detected preference heads for the task
- `_calculate_entropy()`: Computes steering strength based on uncertainty
- `generate_self_contrast()`: Main generation method with preference steering
- `_apply_preference_steering()`: Applies preference-based logit adjustments

**Steering Mechanism**:
```python
# Higher entropy (uncertainty) = more steering needed
entropy = -∑(p * log(p))
alpha = entropy (capped if specified)

# Amplify logits based on preference strength
steered_logits = base_logits + alpha * weight * base_logits
```

---

## Directory Structure

```
/scratch/weixuz/
├── preference_head/
│   ├── preference_scores/                    # Detected preference heads
│   │   ├── Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json
│   │   ├── Meta-Llama-3-8B-Instruct_LaMP_2_top_heads.json
│   │   └── ...
│   ├── preference_head_detection.py          # Detection script
│   └── run_detection.sh                      # SLURM batch script
│
└── decore/
    ├── src/models/
    │   └── dps.py                            # DPS decoder implementation
    │
    ├── configs/
    │   ├── decoder/
    │   │   └── dps.yaml                      # DPS decoder config
    │   │
    │   └── experiment/
    │       ├── lamp_1/dps/llama3_8b_instruct.yaml
    │       ├── lamp_2/dps/llama3_8b_instruct.yaml
    │       ├── lamp_3/dps/llama3_8b_instruct.yaml
    │       └── lamp_4/dps/llama3_8b_instruct.yaml
    │
    ├── test_dps.sh                           # Quick test script
    └── run_dps.sh                            # Full experiment script
```

---

## Configuration

### Decoder Config: `configs/decoder/dps.yaml`

```yaml
defaults:
  - base_decoder_config

name: DPS
method: DPS
configs:
  preference_heads_dir: /scratch/weixuz/preference_head/preference_scores/
  num_preference_heads: 40      # Number of preference heads to use
  task: LaMP-1                  # Overridden by experiment config
  post_softmax: True
  alpha_cap: null               # Cap on steering strength (null = no cap)
  scale_alpha: False            # Scale alpha by log(vocab_size)
  amateur_model_name_or_path: null  # Not yet supported
```

### Experiment Config: `configs/experiment/lamp_1/dps/llama3_8b_instruct.yaml`

```yaml
defaults:
  - override /model: llama3_8b_instruct
  - override /data: lamp_1
  - override /decoder: dps

decoder:
  configs:
    task: LaMP-1                # Task-specific preference heads
    num_preference_heads: 40

model:
  configs:
    max_new_tokens: 32          # Task-specific generation length
```

---

## Usage

### 1. Detect Preference Heads (One-time per task)

```bash
# For LaMP-1
cd /scratch/weixuz/preference_head
sbatch run_detection.sh

# For other tasks, edit run_detection.sh and uncomment the desired task
```

**Output**: `preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json`

**Detection Time**: ~30 minutes per task (400 samples, 1024 heads)

### 2. Quick Test (5 samples)

```bash
cd /scratch/weixuz/decore
bash test_dps.sh
```

**Expected Output**:
```
Testing DPS on LaMP-1 (5 samples)...
Loading preference heads for meta-llama/Meta-Llama-3-8B-Instruct on LaMP-1
Loaded 40 preference heads (requested 40)
✅ DPS test successful!
```

### 3. Full Experiment

```bash
cd /scratch/weixuz/decore
sbatch run_dps.sh
```

### 4. Manual Command

```bash
cd /scratch/weixuz/decore
source /scratch/weixuz/envs/decore/bin/activate

python scripts/main.py \
  experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=40
```

---

## Parameters

### `num_preference_heads`

Number of top preference heads to use for steering.

- **Default**: 40 (4% of 1024 heads)
- **Range**: 10-100
- **Trade-off**: More heads = stronger steering but may overfit

**Tuning**:
```bash
# Use 20 heads (more conservative)
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=20

# Use 60 heads (more aggressive)
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=60
```

### `alpha_cap`

Maximum steering strength (entropy cap).

- **Default**: `null` (no cap)
- **Range**: 0.5-5.0
- **Effect**: Prevents over-steering on very uncertain tokens

**Tuning**:
```bash
# Cap steering at 3.0
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.alpha_cap=3.0
```

### `scale_alpha`

Whether to normalize entropy by log(vocab_size).

- **Default**: `False`
- **Effect**: Makes steering strength comparable across vocabularies

---

## Expected Results

### LaMP-1: Citation Identification

| Method | Accuracy | Notes |
|--------|----------|-------|
| Baseline | ~35% | No personalization |
| DeCoRe | ~42% | Context-aware |
| **DPS** | **~45%** (expected) | Preference-aware |

### Performance Characteristics

- **Latency**: Similar to baseline (no extra model calls)
- **Memory**: Minimal overhead (only stores head indices)
- **Quality**: Better personalization on user-specific tasks

---

## Comparison with Baselines

Run all methods for comparison:

```bash
# Baseline (no steering)
python scripts/main.py experiment=lamp_1/baseline/llama3_8b_instruct

# DeCoRe (context steering)
python scripts/main.py experiment=lamp_1/decore_entropy/llama3_8b_instruct \
  decoder.configs.num_retrieval_heads=50

# DPS (preference steering)
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=40
```

---

## Troubleshooting

### Error: "Preference heads file not found"

**Cause**: Preference heads not detected for this task.

**Solution**:
```bash
cd /scratch/weixuz/preference_head
python preference_head_detection.py --task LaMP-1 --num_samples 400
```

### Error: "Amateur model not yet supported"

**Cause**: DPS currently only supports expert-only mode.

**Solution**: Ensure `decoder.configs.amateur_model_name_or_path` is `null` in config.

### Low Accuracy

**Potential Causes**:
1. Too few preference heads (try increasing `num_preference_heads`)
2. Over-steering (try setting `alpha_cap=3.0`)
3. Wrong task (ensure task matches preference heads file)

**Debug**:
```bash
# Test with debug mode
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  data.num_samples=5 \
  debug=true
```

---

## Preference Heads Format

**File**: `Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json`

```json
{
  "model": "Meta-Llama-3-8B-Instruct",
  "task": "LaMP-1",
  "top_percent": 0.04,
  "num_heads_selected": 40,
  "preference_heads": [
    [28, 18],  // Layer 28, Head 18 (highest PCS)
    [30, 18],  // Layer 30, Head 18
    [29, 18],  // Layer 29, Head 18
    ...
  ]
}
```

**Fields**:
- `preference_heads`: List of `[layer, head]` pairs, sorted by PCS (highest first)
- Each head is in format `[layer_index, head_index]` (0-indexed)

---

## Extension to Other Tasks

### 1. Detect Preference Heads for New Task

```bash
cd /scratch/weixuz/preference_head
python preference_head_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --task LaMP-2 \
  --num_samples 400 \
  --top_percent 0.04 \
  --save_dir ./preference_scores
```

### 2. Create Experiment Config

```yaml
# configs/experiment/lamp_2/dps/llama3_8b_instruct.yaml
defaults:
  - override /model: llama3_8b_instruct
  - override /data: lamp_2
  - override /decoder: dps

decoder:
  configs:
    task: LaMP-2  # Important: must match detection task
    num_preference_heads: 40

model:
  configs:
    max_new_tokens: 32
```

### 3. Run Experiment

```bash
python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct
```

---

## Advanced Usage

### Varying Number of Preference Heads

```bash
# Ablation study
for num_heads in 10 20 40 60 80 100; do
  echo "Testing with $num_heads preference heads..."
  python scripts/main.py \
    experiment=lamp_1/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=$num_heads \
    --multirun
done
```

### Combined with RAG

DPS uses the RAG-integrated LAMP dataset automatically:

```yaml
# configs/data/lamp_1.yaml
retriever: bm25           # BM25 retriever
num_retrieve: 5           # 5 user histories
max_prompt_length: 2048   # Token budget
```

Both retrieval (context) and preference steering work together!

---

## Citation

If you use DPS in your research, please cite:

```bibtex
@article{dps2024,
  title={Differential Preference Steering for Personalized Language Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## Future Work

- [ ] Amateur model support (contrastive steering)
- [ ] Dynamic preference head selection per sample
- [ ] Multi-task preference head sharing
- [ ] Preference head visualization tools
- [ ] Integration with other LLMs (Mistral, Qwen)

---

## Contact

For questions or issues:
- Check `/scratch/weixuz/preference_head/TROUBLESHOOTING.md`
- Review DPS code: `/scratch/weixuz/decore/src/models/dps.py`
- Compare with DeCoRe: `/scratch/weixuz/decore/src/models/decore_entropy.py`

---

## Quick Reference

```bash
# Detect preference heads
cd /scratch/weixuz/preference_head && sbatch run_detection.sh

# Test DPS
cd /scratch/weixuz/decore && bash test_dps.sh

# Run full experiment
cd /scratch/weixuz/decore && sbatch run_dps.sh

# Evaluate results
cd /scratch/weixuz/decore && python evaluate_predictions.py
```

**Key Files**:
- Detection: `/scratch/weixuz/preference_head/preference_head_detection.py`
- Decoder: `/scratch/weixuz/decore/src/models/dps.py`
- Config: `/scratch/weixuz/decore/configs/decoder/dps.yaml`
- Results: `/scratch/weixuz/preference_head/preference_scores/*.json`

---

**Status**: ✅ Ready to use for LaMP-1 (preference heads detected)

**Next Steps**: Detect preference heads for LaMP-2, LaMP-3, LaMP-4

