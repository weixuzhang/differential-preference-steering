# DPS Extension to All LaMP Tasks - Complete Guide

## Overview

This guide shows how to extend DPS (Differential Preference Steering) to all LaMP tasks: LaMP-1, LaMP-2, LaMP-3, and LaMP-4.

**Current Status**:
- ✅ LaMP-1: Preference heads detected (40 heads) - **Ready to run**
- ⏳ LaMP-2: Need detection (~30 minutes)
- ⏳ LaMP-3: Need detection (~30 minutes)
- ⏳ LaMP-4: Need detection (~30 minutes)

---

## Step-by-Step Workflow

### Step 1: Detect Preference Heads for All Tasks

You have two options:

#### Option A: Detect All Tasks at Once (Recommended)

**Batch job** (~2 hours total for all 4 tasks):

```bash
cd /scratch/weixuz/dps-dev-dev/preference_head
sbatch run_detection_all_tasks.sh
```

**What this does**:
- Detects preference heads for LaMP-1, 2, 3, 4 sequentially
- Uses 400 samples per task
- Selects top 4% (40 heads) per task
- Saves results to `preference_scores/`

**Monitor progress**:
```bash
# Check job status
squeue -u $USER

# Watch output (updates in real-time)
tail -f preference_head_detection_all_*.out

# Check completed tasks
ls -lh preference_scores/*_top_heads.json
```

#### Option B: Detect Tasks Individually

If you want more control or need to run one task at a time:

**LaMP-2 (Movie Tagging)**:
```bash
cd /scratch/weixuz/dps-dev-dev/preference_head
python preference_head_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --task LaMP-2 \
  --num_samples 400 \
  --top_percent 0.04 \
  --save_dir ./preference_scores
```

**LaMP-3 (Score Prediction)**:
```bash
python preference_head_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --task LaMP-3 \
  --num_samples 400 \
  --top_percent 0.04 \
  --save_dir ./preference_scores
```

**LaMP-4 (News Headline Generation)**:
```bash
python preference_head_detection.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --task LaMP-4 \
  --num_samples 400 \
  --top_percent 0.04 \
  --save_dir ./preference_scores
```

**Or submit as SLURM jobs**:
```bash
# Edit run_detection.sh to uncomment the desired task
nano run_detection.sh

# Submit
sbatch run_detection.sh
```

---

### Step 2: Verify Detection Results

After detection completes, verify all files are created:

```bash
cd results/preference_head/preference_scores
ls -lh *_top_heads.json
```

**Expected output**:
```
Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json  (~1.4 KB)
Meta-Llama-3-8B-Instruct_LaMP_2_top_heads.json  (~1.4 KB)
Meta-Llama-3-8B-Instruct_LaMP_3_top_heads.json  (~1.4 KB)
Meta-Llama-3-8B-Instruct_LaMP_4_top_heads.json  (~1.4 KB)
```

**Check top heads for each task**:
```bash
# LaMP-1
python3 -c "import json; data=json.load(open('Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json')); print(f'LaMP-1: {data[\"num_heads_selected\"]} heads'); print('Top 5:', data['preference_heads'][:5])"

# LaMP-2
python3 -c "import json; data=json.load(open('Meta-Llama-3-8B-Instruct_LaMP_2_top_heads.json')); print(f'LaMP-2: {data[\"num_heads_selected\"]} heads'); print('Top 5:', data['preference_heads'][:5])"

# LaMP-3
python3 -c "import json; data=json.load(open('Meta-Llama-3-8B-Instruct_LaMP_3_top_heads.json')); print(f'LaMP-3: {data[\"num_heads_selected\"]} heads'); print('Top 5:', data['preference_heads'][:5])"

# LaMP-4
python3 -c "import json; data=json.load(open('Meta-Llama-3-8B-Instruct_LaMP_4_top_heads.json')); print(f'LaMP-4: {data[\"num_heads_selected\"]} heads'); print('Top 5:', data['preference_heads'][:5])"
```

---

### Step 3: Run DPS on All Tasks

Once all preference heads are detected, run DPS experiments.

#### Option A: Run All Tasks at Once (Recommended)

**Batch job** (~3 hours total for all 4 tasks):

```bash
cd /scratch/weixuz/dps-dev
sbatch experiments/run_dps_all_tasks.sh
```

**What this does**:
- Automatically checks which tasks have preference heads
- Runs DPS only on tasks that are ready
- Evaluates all predictions at the end
- Saves results to `outputs/`

#### Option B: Run Tasks Individually

**LaMP-1**:
```bash
cd /scratch/weixuz/dps-dev
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct
```

**LaMP-2**:
```bash
python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct
```

**LaMP-3**:
```bash
python scripts/main.py experiment=lamp_3/dps/llama3_8b_instruct
```

**LaMP-4**:
```bash
python scripts/main.py experiment=lamp_4/dps/llama3_8b_instruct
```

**With custom parameters**:
```bash
# More preference heads
python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=60

# With alpha cap
python scripts/main.py experiment=lamp_3/dps/llama3_8b_instruct \
  decoder.configs.alpha_cap=3.0

# Different sample size
python scripts/main.py experiment=lamp_4/dps/llama3_8b_instruct \
  data.num_samples=100
```

---

### Step 4: Evaluate Results

After all experiments complete:

```bash
cd /scratch/weixuz/dps-dev
python evaluate_predictions.py
```

**View results**:
```bash
# Check prediction files
ls -lh outputs/*dps*.json

# View evaluation summary
cat outputs/evaluation_summary.json
```

---

## Task-Specific Details

### LaMP-1: Citation Identification
- **Task**: Predict which paper to cite based on user's writing history
- **Output**: Citation ID (short text, ~32 tokens)
- **Metric**: Accuracy
- **Status**: ✅ Ready (preference heads detected)

```bash
# Already detected, run experiments:
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct
```

### LaMP-2: Movie Tagging
- **Task**: Predict movie tags based on user's rating history
- **Output**: Tag list (short text, ~32 tokens)
- **Metric**: Multi-label F1 score
- **Status**: ⏳ Need detection

```bash
# Step 1: Detect preference heads
cd /scratch/weixuz/dps-dev-dev/preference_head
python preference_head_detection.py --task LaMP-2 --num_samples 400

# Step 2: Run DPS
cd /scratch/weixuz/dps-dev
python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct
```

### LaMP-3: Score Prediction
- **Task**: Predict rating score based on user's review history
- **Output**: Score (longer text, ~128 tokens)
- **Metric**: Mean Absolute Error (MAE)
- **Status**: ⏳ Need detection

```bash
# Step 1: Detect preference heads
cd /scratch/weixuz/dps-dev-dev/preference_head
python preference_head_detection.py --task LaMP-3 --num_samples 400

# Step 2: Run DPS
cd /scratch/weixuz/dps-dev
python scripts/main.py experiment=lamp_3/dps/llama3_8b_instruct
```

### LaMP-4: News Headline Generation
- **Task**: Generate headline based on user's article history
- **Output**: Headline (medium text, ~64 tokens)
- **Metric**: ROUGE-L
- **Status**: ⏳ Need detection

```bash
# Step 1: Detect preference heads
cd /scratch/weixuz/dps-dev-dev/preference_head
python preference_head_detection.py --task LaMP-4 --num_samples 400

# Step 2: Run DPS
cd /scratch/weixuz/dps-dev
python scripts/main.py experiment=lamp_4/dps/llama3_8b_instruct
```

---

## Expected Timeline

### Detection Phase
| Task | Time | GPU Memory | Output |
|------|------|------------|--------|
| LaMP-1 | ~30 min | ~16 GB | ✅ Done |
| LaMP-2 | ~30 min | ~16 GB | ⏳ Pending |
| LaMP-3 | ~30 min | ~16 GB | ⏳ Pending |
| LaMP-4 | ~30 min | ~16 GB | ⏳ Pending |
| **Total** | **~2 hours** | | **1 job** |

### Experiment Phase
| Task | Samples | Time | Output |
|------|---------|------|--------|
| LaMP-1 | ~2,500 | ~45 min | predictions.json |
| LaMP-2 | ~2,000 | ~35 min | predictions.json |
| LaMP-3 | ~2,000 | ~35 min | predictions.json |
| LaMP-4 | ~2,000 | ~35 min | predictions.json |
| **Total** | **~8,500** | **~3 hours** | **4 files** |

---

## Quick Commands Reference

### Detection
```bash
# All tasks at once
cd /scratch/weixuz/dps-dev-dev/preference_head
sbatch run_detection_all_tasks.sh

# Check progress
tail -f preference_head_detection_all_*.out

# Verify results
ls -lh preference_scores/*_top_heads.json
```

### Experiments
```bash
# All tasks at once
cd /scratch/weixuz/dps-dev
sbatch experiments/run_dps_all_tasks.sh

# Check progress
tail -f dps_all_tasks_*.out

# Verify results
ls -lh outputs/*dps*.json
```

### Evaluation
```bash
cd /scratch/weixuz/dps-dev
python evaluate_predictions.py
cat outputs/evaluation_summary.json
```

---

## Comparison Experiments

To compare DPS with baselines across all tasks:

### Run Baseline on All Tasks
```bash
# LaMP-1
python scripts/main.py experiment=lamp_1/baseline/llama3_8b_instruct

# LaMP-2
python scripts/main.py experiment=lamp_2/baseline/llama3_8b_instruct

# LaMP-3
python scripts/main.py experiment=lamp_3/baseline/llama3_8b_instruct

# LaMP-4
python scripts/main.py experiment=lamp_4/baseline/llama3_8b_instruct
```

### Run DeCoRe on All Tasks
```bash
# LaMP-1
python scripts/main.py experiment=lamp_1/decore_entropy/llama3_8b_instruct \
  decoder.configs.num_retrieval_heads=50

# LaMP-2
python scripts/main.py experiment=lamp_2/decore_entropy/llama3_8b_instruct \
  decoder.configs.num_retrieval_heads=50

# LaMP-3
python scripts/main.py experiment=lamp_3/decore_entropy/llama3_8b_instruct \
  decoder.configs.num_retrieval_heads=50

# LaMP-4
python scripts/main.py experiment=lamp_4/decore_entropy/llama3_8b_instruct \
  decoder.configs.num_retrieval_heads=50
```

### Compare Results
```bash
python evaluate_predictions.py --compare baseline decore dps
```

---

## Expected Results

### Accuracy Improvements (Expected)

| Task | Baseline | DeCoRe | DPS | Improvement |
|------|----------|--------|-----|-------------|
| LaMP-1 | ~35% | ~42% | **~45%** | +10% |
| LaMP-2 | ~30% F1 | ~35% F1 | **~38% F1** | +8% |
| LaMP-3 | 1.2 MAE | 1.0 MAE | **0.9 MAE** | -25% |
| LaMP-4 | ~25 ROUGE | ~28 ROUGE | **~30 ROUGE** | +5 points |

---

## Troubleshooting

### Issue: Detection fails for a task

**Error**: `FileNotFoundError: ./dataset/LaMP-X/dev_questions.json`

**Solution**: Ensure you're running from the correct directory
```bash
cd /scratch/weixuz/dps-dev-dev/preference_head
python preference_head_detection.py --task LaMP-X ...
```

### Issue: DPS experiment fails with "Preference heads file not found"

**Error**: `FileNotFoundError: .../Meta-Llama-3-8B-Instruct_LaMP_X_top_heads.json`

**Solution**: Run detection first
```bash
cd /scratch/weixuz/dps-dev-dev/preference_head
python preference_head_detection.py --task LaMP-X --num_samples 400
```

### Issue: Out of memory during detection

**Error**: `CUDA out of memory`

**Solution**: Reduce number of samples
```bash
python preference_head_detection.py --task LaMP-X --num_samples 200
```

### Issue: Different tasks need different preference heads

**This is expected!** Each task has different preference patterns:
- LaMP-1: Citation preferences → different heads than...
- LaMP-2: Movie taste preferences → different heads than...
- LaMP-3: Rating patterns → different heads than...
- LaMP-4: News/writing style → different heads

---

## Advanced: Parameter Tuning

### Number of Preference Heads

Test different numbers of heads to find optimal performance:

```bash
# Ablation study for LaMP-2
for num_heads in 20 40 60 80; do
  echo "Testing with $num_heads preference heads..."
  python scripts/main.py \
    experiment=lamp_2/dps/llama3_8b_instruct \
    decoder.configs.num_preference_heads=$num_heads \
    data.num_samples=500
done
```

### Alpha Cap

Test different steering strengths:

```bash
# Test alpha caps for LaMP-3
for alpha_cap in 1.0 3.0 5.0; do
  echo "Testing with alpha_cap=$alpha_cap..."
  python scripts/main.py \
    experiment=lamp_3/dps/llama3_8b_instruct \
    decoder.configs.alpha_cap=$alpha_cap \
    data.num_samples=500
done
```

---

## File Structure After Detection

```
/scratch/weixuz/
├── preference_head/
│   └── preference_scores/
│       ├── Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json    ✅
│       ├── Meta-Llama-3-8B-Instruct_LaMP_1_pcs.json
│       ├── Meta-Llama-3-8B-Instruct_LaMP_1_ranked.json
│       ├── Meta-Llama-3-8B-Instruct_LaMP_2_top_heads.json    ⏳
│       ├── Meta-Llama-3-8B-Instruct_LaMP_2_pcs.json
│       ├── Meta-Llama-3-8B-Instruct_LaMP_2_ranked.json
│       ├── Meta-Llama-3-8B-Instruct_LaMP_3_top_heads.json    ⏳
│       ├── Meta-Llama-3-8B-Instruct_LaMP_3_pcs.json
│       ├── Meta-Llama-3-8B-Instruct_LaMP_3_ranked.json
│       ├── Meta-Llama-3-8B-Instruct_LaMP_4_top_heads.json    ⏳
│       ├── Meta-Llama-3-8B-Instruct_LaMP_4_pcs.json
│       └── Meta-Llama-3-8B-Instruct_LaMP_4_ranked.json
│
└── decore/
    └── outputs/
        ├── lamp_1_dps_predictions_*.json
        ├── lamp_2_dps_predictions_*.json
        ├── lamp_3_dps_predictions_*.json
        ├── lamp_4_dps_predictions_*.json
        └── evaluation_summary.json
```

---

## Summary Checklist

### Detection Phase
- [ ] Run `sbatch run_detection_all_tasks.sh`
- [ ] Wait ~2 hours for all tasks
- [ ] Verify 4 `*_top_heads.json` files created
- [ ] Check each file has 40 preference heads

### Experiment Phase
- [ ] Run `sbatch experiments/run_dps_all_tasks.sh`
- [ ] Wait ~3 hours for all tasks
- [ ] Verify 4 prediction JSON files created
- [ ] Run evaluation script

### Analysis Phase
- [ ] Compare DPS vs Baseline vs DeCoRe
- [ ] Analyze per-task improvements
- [ ] Identify which tasks benefit most from personalization
- [ ] Document findings

---

## Next Steps After All Tasks Complete

1. **Analyze Results**
   - Which task benefits most from DPS?
   - Are preference heads similar or different across tasks?
   - Does DPS work better on some types of personalization?

2. **Paper/Report**
   - Document methodology
   - Create comparison tables
   - Visualize preference heads
   - Analyze performance gains

3. **Further Experiments**
   - Try amateur model contrast
   - Experiment with different alpha caps
   - Test with different numbers of heads
   - Combine with other techniques

---

**Ready to Start?**

```bash
# Step 1: Detect preference heads for all remaining tasks
cd /scratch/weixuz/dps-dev-dev/preference_head
sbatch run_detection_all_tasks.sh

# Step 2: Once detection completes, run DPS on all tasks
cd /scratch/weixuz/dps-dev
sbatch experiments/run_dps_all_tasks.sh

# Step 3: Analyze results
python evaluate_predictions.py
```

**Estimated Total Time**: ~5-6 hours for complete pipeline

**Good luck! 🚀**

