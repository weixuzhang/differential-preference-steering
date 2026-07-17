# DPS All Tasks - Quick Start

## TL;DR

Run DPS on all LaMP tasks in 2 steps:

### Step 1: Detect Preference Heads (~2 hours)

```bash
cd /scratch/weixuz/preference_head
sbatch run_detection_all_tasks.sh
```

### Step 2: Run DPS Experiments (~3 hours)

```bash
cd /scratch/weixuz/decore
sbatch run_dps_all_tasks.sh
```

---

## What Was Created

### New Scripts

1. **`/scratch/weixuz/preference_head/run_detection_all_tasks.sh`**
   - Detects preference heads for LaMP-1, 2, 3, 4
   - Runs sequentially (48h time limit)
   - Outputs: 4 `*_top_heads.json` files

2. **`/scratch/weixuz/decore/run_dps_all_tasks.sh`**
   - Runs DPS on all tasks with detected heads
   - Automatically skips tasks without heads
   - Evaluates all results at the end

### Documentation

3. **`/scratch/weixuz/DPS_ALL_TASKS_GUIDE.md`**
   - Complete guide with all details
   - Troubleshooting section
   - Expected results
   - Parameter tuning tips

4. **`/scratch/weixuz/QUICK_START_ALL_TASKS.md`**
   - This file - quick reference

---

## Current Status

| Task | Preference Heads | Config | Ready for DPS |
|------|------------------|--------|---------------|
| LaMP-1 | ✅ Detected (40) | ✅ Yes | ✅ **Run now** |
| LaMP-2 | ❌ Not yet | ✅ Yes | ⏳ After detection |
| LaMP-3 | ❌ Not yet | ✅ Yes | ⏳ After detection |
| LaMP-4 | ❌ Not yet | ✅ Yes | ⏳ After detection |

---

## Commands

### Detect Preference Heads

**All at once (recommended)**:
```bash
cd /scratch/weixuz/preference_head
sbatch run_detection_all_tasks.sh
```

**Check progress**:
```bash
tail -f preference_head_detection_all_*.out
```

**Verify completion**:
```bash
ls -lh preference_scores/*_top_heads.json
```

### Run DPS Experiments

**All tasks**:
```bash
cd /scratch/weixuz/decore
sbatch run_dps_all_tasks.sh
```

**Single task** (if you only want to run one):
```bash
# LaMP-1 (already has heads)
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct

# LaMP-2 (after detection)
python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct

# LaMP-3 (after detection)
python scripts/main.py experiment=lamp_3/dps/llama3_8b_instruct

# LaMP-4 (after detection)
python scripts/main.py experiment=lamp_4/dps/llama3_8b_instruct
```

### Evaluate Results

```bash
cd /scratch/weixuz/decore
python evaluate_predictions.py
```

---

## Timeline

```
Now
 │
 ├─ Run detection (2h) ─────────────┐
 │                                   │
 │                              Detection
 │                              Complete
 │                                   │
 └─ Run DPS experiments (3h) ───────┤
                                     │
                                Results
                                Ready!
                                     │
                               Analyze
                              (5 min)

Total: ~5-6 hours
```

---

## Expected Results

| Task | Baseline | DeCoRe | DPS | Metric |
|------|----------|--------|-----|--------|
| LaMP-1 | ~35% | ~42% | **~45%** | Accuracy |
| LaMP-2 | ~30% | ~35% | **~38%** | F1 |
| LaMP-3 | 1.2 | 1.0 | **0.9** | MAE ↓ |
| LaMP-4 | ~25 | ~28 | **~30** | ROUGE-L |

---

## Files to Check

### After Detection
```bash
/scratch/weixuz/preference_head/preference_scores/
├── Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json  ✅
├── Meta-Llama-3-8B-Instruct_LaMP_2_top_heads.json  ⏳
├── Meta-Llama-3-8B-Instruct_LaMP_3_top_heads.json  ⏳
└── Meta-Llama-3-8B-Instruct_LaMP_4_top_heads.json  ⏳
```

### After Experiments
```bash
/scratch/weixuz/decore/outputs/
├── lamp_1_dps_predictions_*.json
├── lamp_2_dps_predictions_*.json
├── lamp_3_dps_predictions_*.json
├── lamp_4_dps_predictions_*.json
└── evaluation_summary.json
```

---

## Troubleshooting

### Problem: Job fails during detection

**Check**:
```bash
# View error
tail -50 preference_head_detection_all_*.out

# Common causes:
# - Out of memory: reduce --num_samples to 200
# - Dataset not found: check /scratch/weixuz/banditpr/dataset/
```

### Problem: DPS experiment skips a task

**Check**:
```bash
# Verify preference heads exist
ls /scratch/weixuz/preference_head/preference_scores/*_top_heads.json

# If missing, run detection for that task
cd /scratch/weixuz/preference_head
python preference_head_detection.py --task LaMP-X --num_samples 400
```

---

## Full Documentation

For detailed information, see:
- **Complete Guide**: `/scratch/weixuz/DPS_ALL_TASKS_GUIDE.md`
- **DPS Success**: `/scratch/weixuz/DPS_SUCCESS.md`
- **Integration Details**: `/scratch/weixuz/decore/DPS_INTEGRATION.md`

---

## Ready to Start?

```bash
# Step 1: Detect heads for LaMP-2, 3, 4 (~2 hours)
cd /scratch/weixuz/preference_head
sbatch run_detection_all_tasks.sh

# Step 2: Once done, run DPS on all tasks (~3 hours)
cd /scratch/weixuz/decore
sbatch run_dps_all_tasks.sh

# Step 3: View results
python evaluate_predictions.py
cat outputs/evaluation_summary.json
```

**Total Time**: ~5-6 hours
**Result**: DPS results for all 4 LaMP tasks! 🎉

