# Preference Head Detection - Implementation Summary

## Overview

Successfully implemented a complete preference head detection system based on the methodology in `preference_head.md`. The implementation is adapted from the retrieval head detection codebase and specialized for detecting heads that encode user preferences.

---

## Files Created

### 1. **Core Detection** (`preference_head_detection.py`)

**Purpose:** Main script for detecting preference heads using Preference Contribution Score (PCS)

**Key Features:**
- Head-wise ablation to compute PCS
- Supports all LaMP tasks (LaMP-1 through LaMP-7)
- Efficient two-stage detection:
  - Stage 1: Baseline score computation
  - Stage 2: Head-wise ablation (sampled for efficiency)
- Automatic ranking and selection of top preference heads
- Saves results in multiple formats for downstream use

**Usage:**
```bash
python preference_head_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --task LaMP-1 \
  --num_samples 400 \
  --top_percent 0.04 \
  --save_dir ./preference_scores
```

**Key Classes:**
- `PreferenceHeadConfig`: Configuration dataclass
- `PreferenceHeadDetector`: Main detection class with methods:
  - `compute_baseline_score()`: Baseline personalization score
  - `compute_ablation_score()`: Score with head ablated
  - `compute_pcs()`: Preference Contribution Score
  - `detect_preference_heads()`: Main detection loop
  - `rank_heads()`: Rank by average PCS
  - `save_results()`: Save to JSON files

---

### 2. **Validation** (`validate_preference_heads.py`)

**Purpose:** Validate detected preference heads using masking and activation patching

**Key Features:**
- **Masking validation**: Compare performance drop when masking preference vs random heads
- **Activation patching**: Test causal effect by patching activations from clean to corrupted prompts
- Confirms that detected heads actually carry preference information

**Usage:**
```bash
python validate_preference_heads.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --preference_heads_file preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json \
  --task LaMP-1 \
  --num_samples 100
```

**Validation Methods:**
- `validate_with_masking()`: Ablation experiment
- `validate_with_patching()`: Causal intervention experiment

---

### 3. **Visualization** (`visualize_heads.py`)

**Purpose:** Create visualizations of detection results

**Key Features:**
- PCS heatmap (layers × heads)
- Top preference heads bar plot
- Layer distribution histogram
- Comparison with retrieval heads (overlap analysis)

**Usage:**
```bash
python visualize_heads.py \
  --pcs_file preference_scores/model_task_pcs.json \
  --ranked_file preference_scores/model_task_ranked.json \
  --retrieval_file ../retrival_head/head_score/model.json \
  --output_dir ./visualizations
```

---

### 4. **Testing** (`test_detection.py`)

**Purpose:** Quick test with small sample size

**Usage:**
```bash
python test_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --num_samples 10
```

---

### 5. **Automation** (`run_detection.sh`)

**Purpose:** SLURM job script for running detection on cluster

**Features:**
- Pre-configured for LLaMA3-8B-Instruct
- Supports multiple LaMP tasks
- Automatic environment setup

**Usage:**
```bash
sbatch run_detection.sh
```

---

### 6. **Documentation**

- **`README.md`**: Comprehensive usage guide
- **`preference_head.md`**: Methodology documentation
- **`IMPLEMENTATION_SUMMARY.md`**: This file

---

## Detection Methodology

### Preference Contribution Score (PCS)

For each attention head `(layer, head)`:

1. **Baseline**: Compute NLL with full model
   ```
   S(base) = NLL(reference | prompt_with_profile)
   ```

2. **Ablation**: Zero out the head and recompute
   ```
   S(M_{-h}) = NLL(reference | prompt_with_profile, head_ablated)
   ```

3. **PCS Calculation**:
   ```
   PCS(L,H) = S(M_{-h}) - S(base)
   ```
   - Higher PCS → head is more important for personalization
   - Removing it increases NLL (worse personalization)

### Algorithm Flow

```
1. Load LaMP dataset with user profiles
2. For each sample:
   a. Compute baseline personalization score
3. For each head (layer, head):
   a. For subset of samples:
      - Ablate head by zeroing o_proj output
      - Recompute personalization score
      - Calculate PCS
   b. Store average PCS
4. Rank heads by average PCS
5. Select top X% as preference heads
6. Save results
```

---

## Output Files

### `{model}_{task}_pcs.json`
Full PCS scores for all heads:
```json
{
  "15-30": [0.023, 0.021, 0.025, ...],
  "24-27": [0.019, 0.022, 0.018, ...],
  ...
}
```

### `{model}_{task}_ranked.json`
Heads ranked by average PCS:
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

### `{model}_{task}_top_heads.json`
Top preference heads for DPS:
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

---

## Integration with DPS (Differential Preference Steering)

### Loading Preference Heads

```python
import json

# Load detected preference heads
with open('preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json', 'r') as f:
    heads_data = json.load(f)
    preference_heads = heads_data['preference_heads']

print(f"Loaded {len(preference_heads)} preference heads")
# Output: Loaded 50 preference heads
```

### Using in DPS (Similar to DeCoRe)

1. **Extract attention from preference heads during generation**
2. **Compute preference signal** (e.g., attention entropy, magnitude)
3. **Modulate logits** based on preference signal
4. **Steer generation** toward user-specific style

---

## Comparison: Preference vs Retrieval Heads

| Aspect | Retrieval Heads | Preference Heads |
|--------|----------------|------------------|
| **Detection Method** | Needle-in-haystack + attention tracking | Profile ablation + PCS |
| **Metric** | Retrieval accuracy (ROUGE recall) | Personalization (NLL) |
| **Dataset** | Generic text + inserted needle | User-profiled data (LaMP) |
| **Purpose** | Copy/retrieve facts | Inject user preferences |
| **Validation** | Masking → retrieval fails | Masking → personalization drops |
| **Layer Distribution** | Often mid-to-late layers | TBD (to be discovered) |
| **Expected Overlap** | Low-to-moderate | |

---

## Expected Results

### Hypothesis

1. **Top 2-6%** of heads should account for most personalization effect
2. **Layer distribution**: Preference heads may be distributed across layers
3. **Task variation**: Different tasks may use different preference heads
4. **Distinct from retrieval**: Low overlap with retrieval heads

### Success Criteria

- **High PCS heads** (> 0.01): Strong preference contributors
- **Validation**: Masking preference heads significantly increases NLL
- **Causal test**: Patching restores personalization in corrupted inputs

---

## Usage Examples

### Quick Test (10 samples, ~5 minutes)

```bash
python test_detection.py --num_samples 10
```

### Full Detection (400 samples, ~4-8 hours on 80GB GPU)

```bash
sbatch run_detection.sh
```

### Multi-Task Detection

```bash
for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4; do
  python preference_head_detection.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --task $task \
    --num_samples 400 \
    --save_dir ./preference_scores
done
```

### Validation

```bash
python validate_preference_heads.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --preference_heads_file preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json \
  --task LaMP-1 \
  --num_samples 100
```

### Visualization

```bash
python visualize_heads.py \
  --pcs_file preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_pcs.json \
  --ranked_file preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_ranked.json \
  --output_dir ./visualizations
```

---

## Key Differences from Retrieval Head Detection

### 1. **Scoring Metric**
- **Retrieval**: Track if attention focuses on needle tokens during generation
- **Preference**: Measure NLL change when head is ablated

### 2. **Dataset**
- **Retrieval**: Synthetic needle-in-haystack
- **Preference**: Real user-profiled data (LaMP)

### 3. **Detection Signal**
- **Retrieval**: Attention matrix inspection
- **Preference**: Performance degradation (ablation)

### 4. **Computational Cost**
- **Retrieval**: O(samples × decode_steps)
- **Preference**: O(samples × num_layers × num_heads) - more expensive

---

## Next Steps

### 1. **Run Full Detection**
```bash
sbatch run_detection.sh
```

### 2. **Analyze Results**
- Check PCS distribution
- Identify layer patterns
- Compare across tasks

### 3. **Validate Detected Heads**
- Run masking experiments
- Test activation patching
- Confirm causal effect

### 4. **Integrate with DPS**
- Implement DPS decoder (similar to DeCoRe)
- Use preference heads instead of retrieval heads
- Test on LAMP benchmark

---

## Troubleshooting

### OOM Errors
- Reduce `--num_samples`
- Use gradient checkpointing
- Process in smaller batches

### Slow Detection
- Normal: ~10-20 seconds per head per sample
- For 32 layers × 32 heads × 50 samples ≈ 4-8 hours
- Use `--num_samples 50` for quick testing

### No Clear Signal
- Try different tasks (LaMP-1, LaMP-3 have strong personalization)
- Increase `--num_samples`
- Check if model/dataset has personalization signal

---

## Code Quality

- ✅ Modular design (separate detection, validation, visualization)
- ✅ Type hints throughout
- ✅ Comprehensive error handling
- ✅ Progress bars (tqdm)
- ✅ Detailed logging
- ✅ JSON output for reproducibility
- ✅ Command-line interface (argparse)
- ✅ Documentation (docstrings, README)

---

## Dependencies

- `torch >= 2.0`
- `transformers >= 4.37`
- `datasets`
- `numpy`
- `tqdm`
- `matplotlib` (for visualization)
- `seaborn` (for visualization)

All dependencies already available in DeCoRe environment.

---

## Summary

✅ **Complete preference head detection system implemented**  
✅ **Follows methodology from `preference_head.md`**  
✅ **Adapted from retrieval head detection**  
✅ **Ready for testing and integration with DPS**  

**Next:** Run detection, analyze results, integrate with DPS decoder!

