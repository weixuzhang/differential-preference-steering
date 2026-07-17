# Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: FileNotFoundError - Dataset Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: './dataset/LaMP-1/dev_questions.json'
```

**Cause:**
The `load_lamp_dataset()` function from BanditPR uses relative paths (`./dataset/`) and expects to be run from the `/scratch/weixuz/banditpr/` directory. When running from `/scratch/weixuz/preference_head/`, it can't find the dataset files.

**Solution:**
The code now automatically changes to the correct directory when loading the dataset:

```python
# In preference_head_detection.py and validate_preference_heads.py
original_cwd = os.getcwd()
try:
    os.chdir('/scratch/weixuz/banditpr')
    dataset = load_lamp_dataset(task, split='dev')
finally:
    os.chdir(original_cwd)
```

**Status:** ✅ Fixed in latest version

---

### Issue 2: HuggingFace Token Required

**Error:**
```
OSError: We couldn't connect to 'https://huggingface.co' to load the files
```

**Cause:**
Model is not cached locally and `HF_OFFLINE=true` is set.

**Solution:**

Option 1: Run in online mode first to download the model:
```bash
# In run_detection.sh, change:
export HF_OFFLINE=false  # instead of true
export HF_TOKEN="your_token_here"
```

Option 2: Use already cached model:
```bash
# Make sure model is in cache
ls /scratch/weixuz/decore/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct
```

Option 3: Point to local model:
```bash
python preference_head_detection.py \
  --model_path /scratch/weixuz/decore/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/...
```

---

### Issue 3: CUDA Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Cause:**
- Model + ablation requires significant memory
- Processing too many samples at once

**Solution:**

1. **Reduce samples per head:**
```python
# In preference_head_detection.py, line ~260
samples_per_head = min(20, num_samples)  # Reduce from 50 to 20
```

2. **Use smaller batch processing:**
```python
# Process samples in smaller chunks
for i in range(0, len(sample_indices), batch_size):
    batch = sample_indices[i:i+batch_size]
    # Process batch
```

3. **Use gradient checkpointing:**
```python
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

4. **Request larger GPU:**
```bash
# In run_detection.sh
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G  # Request more memory
```

---

### Issue 4: Detection Takes Too Long

**Cause:**
- Normal behavior: O(num_layers × num_heads × num_samples) forward passes
- For LLaMA3-8B: 32 layers × 32 heads × 50 samples = 51,200 forward passes

**Solution:**

1. **Quick testing:**
```bash
python test_detection.py --num_samples 5  # Very fast (~10 min)
```

2. **Reduced detection:**
```bash
python preference_head_detection.py \
  --num_samples 100 \  # Instead of 400
  --top_percent 0.1    # Select top 10% instead of 4%
```

3. **Subset of layers:**
Modify code to only test certain layers:
```python
# In detect_preference_heads()
for layer_idx in range(16, 32):  # Only last half of layers
    # ... detection code
```

---

### Issue 5: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'lamp'
```

**Cause:**
BanditPR path not in Python path.

**Solution:**
The code already handles this:
```python
sys.path.append('/scratch/weixuz/banditpr/src')
```

If still failing, verify BanditPR is installed:
```bash
ls /scratch/weixuz/banditpr/src/lamp/
# Should show: __init__.py, dataset.py, metric.py, etc.
```

---

### Issue 6: Wrong Task Format

**Error:**
```
ValueError: Invalid task: lamp-1
```

**Cause:**
Task name must be capitalized correctly.

**Solution:**
Use correct format:
- ✅ `LaMP-1` (capital L, P, hyphen)
- ❌ `lamp-1`, `LAMP-1`, `LaMP_1`

---

### Issue 7: Permission Denied

**Error:**
```
Permission denied: ./preference_scores/
```

**Cause:**
Directory doesn't exist or no write permissions.

**Solution:**
```bash
mkdir -p ./preference_scores
chmod 755 ./preference_scores
```

Or specify different directory:
```bash
python preference_head_detection.py --save_dir ~/my_results
```

---

## Quick Diagnostics

### Test 1: Verify Environment
```bash
cd /scratch/weixuz/preference_head
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Test 2: Verify BanditPR Dataset
```bash
python -c "
import sys
sys.path.append('/scratch/weixuz/banditpr/src')
import os
os.chdir('/scratch/weixuz/banditpr')
from lamp import load_lamp_dataset
dataset = load_lamp_dataset('LaMP-1', 'dev')
print(f'Dataset loaded: {len(dataset)} samples')
"
```

### Test 3: Verify Model Access
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    local_files_only=True
)
print('Model found in cache!')
"
```

### Test 4: Run Minimal Test
```bash
python test_detection.py --num_samples 2
```

---

## Performance Optimization

### For Quick Testing (< 30 minutes)
```bash
python preference_head_detection.py \
  --num_samples 50 \
  --top_percent 0.1 \
  --device cuda
```

### For Production (4-8 hours, best results)
```bash
sbatch run_detection.sh  # Uses 400 samples, top 4%
```

### For Multiple Tasks (overnight)
```bash
for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4; do
  sbatch --job-name=pref_${task} run_detection.sh
done
```

---

## Getting Help

1. **Check logs:**
```bash
tail -50 preference_head_detection.out
```

2. **Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. **Test with small model:**
```bash
python preference_head_detection.py \
  --model_path gpt2 \  # Smaller model
  --num_samples 10
```

---

## Expected Runtime

| Configuration | Samples | Time (80GB GPU) |
|--------------|---------|-----------------|
| Quick test | 10 | ~5-10 minutes |
| Fast detection | 50 | ~30-60 minutes |
| Standard | 200 | ~2-4 hours |
| Production | 400 | ~4-8 hours |

**Note:** Time scales approximately linearly with:
- Number of samples
- Number of layers × heads
- Model size

---

## Contact

If you encounter issues not covered here, check:
1. This file (`TROUBLESHOOTING.md`)
2. Main README (`README.md`)
3. Implementation summary (`IMPLEMENTATION_SUMMARY.md`)

