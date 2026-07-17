# Path Configuration for Preference Head Detection

This document explains the directory structure and path configuration used in the preference head detection system.

---

## Directory Structure

```
/scratch/weixuz/
├── decore/
│   ├── .cache/huggingface/          # Shared HuggingFace cache
│   │   ├── models--meta-llama--Meta-Llama-3-8B-Instruct/
│   │   ├── datasets/
│   │   ├── evaluate/
│   │   └── metrics/
│   ├── outputs/                      # DeCoRe experiment outputs
│   └── ...
│
├── lamp_data/
│   ├── dataset/                      # LaMP datasets
│   │   ├── LaMP-1/
│   │   │   ├── dev_questions.json
│   │   │   └── dev_outputs.json
│   │   ├── LaMP-2/
│   │   └── ...
│   └── src/lamp/                     # LAMP data loading utilities
│
└── preference_head/
    ├── preference_scores/            # Detection results (output)
    │   ├── Meta-Llama-3-8B-Instruct_LaMP_1_pcs.json
    │   ├── Meta-Llama-3-8B-Instruct_LaMP_1_ranked.json
    │   └── Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json
    │
    ├── test_preference_scores/       # Test run results
    │
    ├── visualizations/               # Visualization outputs
    │
    └── *.py                          # Detection scripts
```

---

## Path Configuration

### 1. **HuggingFace Cache**

**Path:** `/scratch/weixuz/decore/.cache/huggingface`

**Why this path?**
- Shared with DeCoRe to avoid duplicating large model files
- LLaMA3-8B-Instruct is already cached here (~16GB)
- Saves disk space and download time

**Environment Variables:**
```bash
export TRANSFORMERS_CACHE="/scratch/weixuz/decore/.cache/huggingface"
export HF_HOME="/scratch/weixuz/decore/.cache/huggingface"
export HF_OFFLINE=true  # Use cached models only
```

### 2. **LaMP Dataset**

**Path:** `/scratch/weixuz/lamp_data/LaMP-X/`

**Why this path?**
- lamp_benchmark's `load_lamp_dataset()` uses relative paths from this location
- Dataset already downloaded and processed here
- Code temporarily changes directory to load data:
  ```python
  os.chdir('/scratch/weixuz/dps')
  dataset = load_lamp_dataset('LaMP-1', 'dev')
  ```

### 3. **Output Directory**

**Path:** `/scratch/weixuz/preference_head/preference_scores/`

**Why this path?**
- Keeps results organized within preference_head project
- Absolute path prevents confusion about working directory
- Easy to find and version control

**Output Files:**
- `{model}_{task}_pcs.json` - Full PCS scores
- `{model}_{task}_ranked.json` - Ranked heads
- `{model}_{task}_top_heads.json` - Top preference heads

### 4. **Working Directory**

**Script Location:** `/scratch/weixuz/preference_head/`

**Execution:**
```bash
cd /scratch/weixuz/preference_head
sbatch run_detection.sh
# or
python preference_head_detection.py --model_path ...
```

---

## Configuration in Scripts

### `run_detection.sh`

```bash
# Cache (shared with decore)
hf_cache="/scratch/weixuz/decore/.cache/huggingface"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

# Output directory (absolute path)
SAVE_DIR="/scratch/weixuz/preference_head/preference_scores"

# Model (uses cached version)
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
```

### `test_detection.py`

```python
config = PreferenceHeadConfig(
    model_path="meta-llama/Meta-Llama-3-8B-Instruct",
    save_dir="/scratch/weixuz/preference_head/test_preference_scores",
    ...
)
```

### `preference_head_detection.py`

```python
# Dataset loading (handles path internally)
os.chdir('/scratch/weixuz/dps')
dataset = load_lamp_dataset(task, split='dev')
os.chdir(original_cwd)

# Model loading (uses HF_HOME cache)
self.model = AutoModelForCausalLM.from_pretrained(
    config.model_path,  # Uses TRANSFORMERS_CACHE
    ...
)
```

---

## Advantages of This Setup

### 1. **Resource Sharing**
- ✅ Model cache shared with DeCoRe (saves ~16GB disk space)
- ✅ Dataset shared with lamp_benchmark (saves ~2GB)
- ✅ No duplication of large files

### 2. **Consistency**
- ✅ Same environment as DeCoRe
- ✅ Same model versions
- ✅ Same cache locations

### 3. **Offline Operation**
- ✅ `HF_OFFLINE=true` ensures no unexpected downloads
- ✅ All required files already cached
- ✅ Faster execution (no network delays)

### 4. **Organization**
- ✅ Results clearly separated by project
- ✅ Easy to backup/archive
- ✅ Clear directory structure

---

## Verification Commands

### Check HuggingFace Cache

```bash
# Verify model is cached
ls /scratch/weixuz/decore/.cache/huggingface/models--meta-llama--Meta-Llama-3-8B-Instruct/

# Check cache size
du -sh /scratch/weixuz/decore/.cache/huggingface/
```

### Check LaMP Dataset

```bash
# Verify dataset exists
ls /scratch/weixuz/lamp_data/LaMP-1/

# Check dataset files
ls -lh /scratch/weixuz/lamp_data/LaMP-1/*.json
```

### Check Output Directory

```bash
# Create if doesn't exist
mkdir -p /scratch/weixuz/preference_head/preference_scores

# Check permissions
ls -ld /scratch/weixuz/preference_head/preference_scores
```

---

## Environment Setup

### Method 1: Using run_detection.sh (Recommended)

```bash
sbatch run_detection.sh
```

All paths and environment variables are automatically configured.

### Method 2: Manual Setup

```bash
# Activate environment
source /scratch/weixuz/envs/decore/bin/activate

# Set cache
export TRANSFORMERS_CACHE="/scratch/weixuz/decore/.cache/huggingface"
export HF_HOME="/scratch/weixuz/decore/.cache/huggingface"
export HF_OFFLINE=true

# Run detection
cd /scratch/weixuz/preference_head
python preference_head_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --task LaMP-1 \
  --save_dir /scratch/weixuz/preference_head/preference_scores
```

---

## Troubleshooting Paths

### Issue: Model Not Found

**Error:**
```
OSError: Can't load model 'meta-llama/Meta-Llama-3-8B-Instruct'
```

**Solution:**
```bash
# Check if model is cached
ls /scratch/weixuz/decore/.cache/huggingface/models--meta-llama--Meta-Llama-3-8B-Instruct/

# If not, ensure cache path is correct
echo $TRANSFORMERS_CACHE
# Should output: /scratch/weixuz/decore/.cache/huggingface
```

### Issue: Dataset Not Found

**Error:**
```
FileNotFoundError: ./dataset/LaMP-1/dev_questions.json
```

**Solution:**
```bash
# Verify dataset exists
ls /scratch/weixuz/lamp_data/LaMP-1/

# Code should handle this automatically by changing directory
# If issue persists, check that path in preference_head_detection.py line ~107
```

### Issue: Permission Denied on Output

**Error:**
```
Permission denied: /scratch/weixuz/preference_head/preference_scores/
```

**Solution:**
```bash
# Create directory with correct permissions
mkdir -p /scratch/weixuz/preference_head/preference_scores
chmod 755 /scratch/weixuz/preference_head/preference_scores
```

---

## Path Summary Table

| Resource | Path | Type | Size |
|----------|------|------|------|
| Model Cache | `/scratch/weixuz/decore/.cache/huggingface/` | Shared | ~16GB |
| LaMP Dataset | `/scratch/weixuz/lamp_data/` | Shared | ~2GB |
| Detection Scripts | `/scratch/weixuz/preference_head/` | Local | ~100KB |
| Output Results | `/scratch/weixuz/preference_head/preference_scores/` | Local | ~10MB |
| Test Results | `/scratch/weixuz/preference_head/test_preference_scores/` | Local | ~1MB |
| Visualizations | `/scratch/weixuz/preference_head/visualizations/` | Local | ~5MB |

---

## Best Practices

1. ✅ **Always use absolute paths** in configuration files
2. ✅ **Share cache directories** across projects (decore, preference_head)
3. ✅ **Keep outputs organized** within project directories
4. ✅ **Set HF_OFFLINE=true** to avoid unexpected downloads
5. ✅ **Verify paths** before long-running jobs (use test_detection.py)

---

## Quick Reference

```bash
# Environment
DECORE_ENV="/scratch/weixuz/envs/decore"
HF_CACHE="/scratch/weixuz/decore/.cache/huggingface"

# Data
LAMP_DATA="/scratch/weixuz/lamp_data"
LAMP_TASK="LaMP-1"

# Project
PREF_HEAD_DIR="/scratch/weixuz/preference_head"
OUTPUT_DIR="${PREF_HEAD_DIR}/preference_scores"

# Model
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
```

---

This configuration ensures:
- ✅ No duplicate downloads
- ✅ Consistent environment with DeCoRe
- ✅ Clear organization
- ✅ Offline operation
- ✅ Easy debugging

