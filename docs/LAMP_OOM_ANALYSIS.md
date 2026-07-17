# LAMP Dataset OOM Analysis & Solutions

## Problem Summary

### Job Failures

1. **decore_lamp2_decore.out (Job 32472)**: CUDA Out of Memory Error
   - Error: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 50.89 GiB`
   - GPU: 79.19 GiB total, only 15.93 GiB free
   - Failed at line 1671 in `modelling_llama.py`: `logits = logits.float()`

2. **decore_lamp1_base.out (Job 32475)**: Missing Configuration File
   - Error: `Could not find 'experiment/lamp_1/base_model/llama3_8b_instruct'`
   - **STATUS: FIXED** - Created missing baseline configs

---

## Root Cause: LAMP Dataset Has EXTREMELY Long Inputs

### Dataset Analysis (LAMP-3)

```
Profile token statistics (first 100 samples):
  Mean:   41,931 tokens  (10.2x the 4096 limit!)
  Median: 13,039 tokens  (3.2x the 4096 limit!)
  Min:     2,611 tokens
  Max:   443,044 tokens  (108x the 4096 limit!)

Distribution:
  - 96/100 samples exceed 4096 tokens
  - 74/100 samples exceed 8192 tokens
```

**The LAMP datasets include extensive user profile history**, which creates inputs that are:
- 10-100x longer than typical QA datasets
- Far beyond the 4096 max_seq_len configured
- Incompatible with current memory constraints

### Why DeCoRe Makes It Worse

DeCoRe performs **TWO forward passes** per generation step:
1. **Base model** (normal attention)
2. **Hallucinated model** (blocked retrieval heads)

Each maintains separate KV caches, effectively **doubling memory requirements**.

### Memory Breakdown

For LLaMA-3-8B (vocab_size = 128,256):

| Sequence Length | Logits Memory | With DeCoRe (2x) + KV Cache |
|-----------------|---------------|------------------------------|
| 4,096 tokens    | 1.96 GB       | ~8-10 GB                     |
| 10,000 tokens   | 4.78 GB       | ~20-25 GB                    |
| 20,000 tokens   | 9.56 GB       | ~40-50 GB                    |
| 40,000 tokens   | 19.11 GB      | **~80-100 GB (OOM!)**        |

The attempted 50.89 GB allocation suggests the input was **~20K+ tokens**.

---

## Solutions

### Solution 1: **Truncate Input Sequences** ⭐ RECOMMENDED

Modify the LAMP dataset loader to truncate profiles to a manageable size.

**Implementation:**

Edit `/scratch/weixuz/dps/src/datasets/lamp.py`:

```python
def _get_instruction(self, task: str, profiles: List[Dict], query: str, corpus: List[str]) -> str:
    """Generate instruction based on LAMP task and profiles."""
    
    # SOLUTION: Truncate profile text to fit within token budget
    MAX_PROFILE_TOKENS = 2048  # Adjust based on GPU memory
    CHARS_PER_TOKEN = 4  # Rough estimate
    max_profile_chars = MAX_PROFILE_TOKENS * CHARS_PER_TOKEN
    
    profile_texts = [p.get('text', '') for p in profiles if p.get('text')]
    full_profile_text = "\n".join(profile_texts)
    
    # Truncate if too long
    if len(full_profile_text) > max_profile_chars:
        full_profile_text = full_profile_text[:max_profile_chars] + "..."
    
    if full_profile_text:
        profile_section = f"User Profile:\n{full_profile_text}\n\n"
    else:
        profile_section = ""
    
    # Rest of the function remains the same...
```

**Benefits:**
- Simple to implement
- Works with existing GPU
- Can tune truncation based on available memory

### Solution 2: **Use Gradient Checkpointing**

Enable gradient checkpointing to trade computation for memory.

Edit model config or add to experiment configs:

```yaml
model:
  configs:
    max_seq_len: 8192  # Can increase if using checkpointing
    max_new_tokens: 16
    gradient_checkpointing: true
```

**Benefits:**
- Reduces peak memory by ~30-40%
- Allows longer sequences

**Drawbacks:**
- Slower inference (~20-30% overhead)

### Solution 3: **Request Larger GPU**

The current job used a GPU with 79.19 GiB. For LAMP with full profiles:

| GPU Type           | Memory | Can Handle Avg LAMP-3? | Can Handle Max LAMP-3? |
|--------------------|--------|------------------------|------------------------|
| A100 (current)     | 80 GB  | ❌ No (~42K tokens)    | ❌ No                  |
| A100 80GB (shared) | 80 GB  | ❌ No                  | ❌ No                  |
| H100               | 80 GB  | ❌ No                  | ❌ No                  |
| **Multiple GPUs**  | 160GB+ | ✅ Maybe               | ❌ No                  |

**Reality Check:** Even with the largest single GPU (H100 80GB), the **average** LAMP-3 sample with DeCoRe will likely OOM. The max sample (443K tokens) is impossible without major changes.

### Solution 4: **Use Flash Attention 2**

Already enabled in the log but deprecated. Update to proper implementation:

```python
# In model loading code
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # New syntax
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

**Benefits:**
- Reduces attention memory from O(n²) to O(n)
- 20-30% memory savings

### Solution 5: **Reduce Batch Size** ✅ Already Done

Current config has `batch_size: 1`, which is optimal.

### Solution 6: **Use Smaller Model or Quantization**

Options:
- **8-bit quantization**: Saves ~50% memory
- **4-bit quantization (QLoRA)**: Saves ~75% memory
- **Smaller model**: Use LLaMA-3-7B instead of 8B (though you already are)

---

## Recommended Approach

**Short-term (Immediate Fix):**

1. ✅ **Truncate profiles to 2048 tokens** (Solution 1)
2. ✅ **Add max_seq_len validation** in dataset loader
3. ✅ **Enable expandable segments**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Medium-term:**

4. Enable **gradient checkpointing** (Solution 2)
5. Update to proper **Flash Attention 2** syntax (Solution 4)
6. Consider **8-bit quantization** for 2x memory reduction

**Long-term:**

7. Implement **profile selection** instead of using all profiles
8. Use **retrieval** to select most relevant profile snippets

---

## Why Other Datasets Work

Datasets like NQ, TriviaQA, PopQA have:
- Much shorter contexts (typically < 2K tokens)
- Single passage retrieval (not entire user histories)
- Fits comfortably within 4096 token budget

LAMP is fundamentally different - it's a **personalization** dataset with extensive user history.

---

## Immediate Action Items

1. ✅ **Config files created** for baseline experiments
2. **Implement profile truncation** in `src/datasets/lamp.py`
3. **Update run.sh** to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. **Test with truncated profiles** on a single sample first
5. **Monitor GPU memory** during runs

---

## GPU Recommendations

For LAMP datasets with DeCoRe:

| Approach                          | Minimum GPU Memory | Recommended |
|-----------------------------------|-------------------|-------------|
| Truncated (2K tokens)             | 40 GB             | 80 GB       |
| Truncated (4K tokens)             | 60 GB             | 80 GB       |
| Truncated (8K tokens)             | 100 GB            | 2x A100     |
| Full profiles (avg 42K)           | 200+ GB           | **Infeasible** |

**Conclusion:** You MUST truncate profiles. There's no GPU configuration that can handle full LAMP profiles with DeCoRe.
