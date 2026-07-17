# Preference Head Detection

Detect **Preference Heads** — attention heads in transformers that causally encode and inject user-specific preferences (style, tone, vocabulary) into model outputs.

This code is adapted from the retrieval head detection methodology to identify heads that specialize in personalization rather than retrieval.

## Overview

**Preference heads** are attention mechanisms that:
- Encode user-specific preferences from profile/history
- Inject stylistic and tonal information into generations
- Enable personalized text generation

This is different from **retrieval heads** which focus on copying/retrieving factual information.

---

## Methodology

### 1. **Preference Contribution Score (PCS)**

For each attention head `(layer, head)`:

1. **Baseline**: Compute personalization score `S(base)` with full model
2. **Ablation**: Zero out the head and compute score `S(M_{-h})`
3. **PCS**: `PCS(L,H) = S(M_{-h}) - S(base)`

Higher PCS → head is more important for personalization (removing it hurts performance)

### 2. **Personalization Metrics**

- **Primary**: Negative log-likelihood (NLL) of reference completion
  - Lower NLL = better match to user's style
- **Alternative**: Style similarity metrics (BLEU, ROUGE, METEOR)

### 3. **Validation**

- **Masking**: Verify that masking preference heads hurts personalization more than random heads
- **Activation Patching**: Inject clean (with-profile) activations into corrupted (no-profile) runs
  - If personalization improves, head is causally carrying preference info

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch transformers datasets
pip install numpy tqdm

# For LAMP dataset integration
cd /scratch/weixuz/dps
# (ensure LAMP_DATA_ROOT points at the dataset)
```

### Environment

- **PyTorch**: 2.0+
- **Transformers**: 4.37+
- **GPU**: Recommended (80GB for large models)

---

## Usage

### 1. **Detect Preference Heads**

```bash
python preference_head_detection.py \
  --model_path /path/to/llama3-8b-instruct \
  --task LaMP-1 \
  --num_samples 400 \
  --top_percent 0.04 \
  --save_dir ./preference_scores
```

**Arguments:**
- `--model_path`: Path to model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--task`: LaMP task (LaMP-1, LaMP-2, ..., LaMP-7)
- `--num_samples`: Number of samples for detection (default: 400)
- `--top_percent`: Percentage of heads to select (default: 0.04 = 4%)
- `--save_dir`: Where to save results

**Output Files:**
- `{model}_{task}_pcs.json`: Full PCS scores for all heads
- `{model}_{task}_ranked.json`: Heads ranked by PCS
- `{model}_{task}_top_heads.json`: Top preference heads (for use in DPS)

### 2. **Validate Preference Heads**

```bash
python validate_preference_heads.py \
  --model_path /path/to/llama3-8b-instruct \
  --preference_heads_file ./preference_scores/llama3_LaMP_1_top_heads.json \
  --task LaMP-1 \
  --num_samples 100
```

**Validation Methods:**
1. **Masking**: Compare performance drop when masking preference vs random heads
2. **Activation Patching**: Test causal effect by patching activations

---

## Results Format

### Top Heads File (`*_top_heads.json`)

```json
{
  "model": "Meta-Llama-3-8B-Instruct",
  "task": "LaMP-1",
  "top_percent": 0.04,
  "num_heads_selected": 50,
  "preference_heads": [
    [15, 30],
    [24, 27],
    [16, 20],
    ...
  ]
}
```

Each `[layer, head]` pair represents a detected preference head.

### Ranked File (`*_ranked.json`)

```json
{
  "model": "Meta-Llama-3-8B-Instruct",
  "task": "LaMP-1",
  "num_samples": 400,
  "ranked_heads": [
    {
      "layer": 15,
      "head": 30,
      "avg_pcs": 0.0234,
      "rank": 1
    },
    ...
  ]
}
```

---

## Integration with Differential Preference Steering (DPS)

Once preference heads are detected, use them for personalization:

```python
import json

# Load detected preference heads
with open('preference_scores/llama3_LaMP_1_top_heads.json', 'r') as f:
    heads_data = json.load(f)
    preference_heads = heads_data['preference_heads']

# Use in DPS (similar to DeCoRe)
# 1. Extract attention from preference heads during generation
# 2. Modulate outputs based on user preference signal
# 3. Steer generation toward user-specific style
```

---

## Comparison: Preference Heads vs Retrieval Heads

| Aspect | Retrieval Heads | Preference Heads |
|--------|----------------|------------------|
| **Purpose** | Copy/retrieve factual info | Inject user preferences |
| **Detection** | Needle-in-haystack + attention tracking | Profile ablation + NLL |
| **Metric** | Retrieval accuracy | Personalization score (NLL) |
| **Dataset** | Generic text + needle | User-profiled data (LaMP) |
| **Layer Distribution** | Often mid-to-late layers | TBD (likely distributed) |

---

## Example: Quick Start

```bash
# 1. Detect preference heads for LaMP-1 (Citation Identification)
python preference_head_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --task LaMP-1 \
  --num_samples 200 \
  --top_percent 0.05

# 2. Validate detected heads
python validate_preference_heads.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --preference_heads_file preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json \
  --task LaMP-1 \
  --num_samples 50

# 3. Use heads in DPS (next step)
# See ../decore for integration example
```

---

## Expected Results

### Hypothesis

- **Top 2-6%** of heads should account for most personalization effect
- **Masking** preference heads should significantly increase NLL (worse personalization)
- **Patching** preference head activations should restore personalization in corrupted inputs

### Interpretation

- **High PCS** (> 0.01): Strong preference contributor
- **Medium PCS** (0.001 - 0.01): Moderate effect
- **Low PCS** (< 0.001): Minimal personalization role

---

## Advanced Usage

### Multi-Task Detection

Detect preference heads across multiple LaMP tasks:

```bash
for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4; do
  python preference_head_detection.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --task $task \
    --num_samples 400 \
    --save_dir ./preference_scores
done
```

### Per-User Calibration

After global detection, fine-tune on specific users:

```python
# Stage A: Global detection (400-800 examples across users)
# Stage B: Per-user calibration (20-50 examples per user)
# Adjust weights based on user-specific PCS
```

---

## Troubleshooting

### OOM (Out of Memory)

- Reduce `--num_samples`
- Use smaller model or quantization
- Process in batches

### Slow Detection

- Detection is compute-intensive (requires N_layers × N_heads forward passes per sample)
- For quick testing: use `--num_samples 50 --top_percent 0.1`
- For production: use full `--num_samples 400+`

### No Clear Preference Heads

- Check if task actually has personalization signal
- Verify LaMP dataset has user profiles
- Try different tasks (LaMP-1, LaMP-3 work well)

---

## Citation

If you use this code, please cite:

```bibtex
@article{decore2024,
  title={DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucinations},
  author={...},
  year={2024}
}
```

And the LaMP benchmark:

```bibtex
@inproceedings{lamp2023,
  title={LaMP: When Large Language Models Meet Personalization},
  author={...},
  year={2023}
}
```

---

## Contact

For questions or issues, please open an issue in the repository.

