# DPS Quick Start Guide

## What is DPS?

**DPS (Differential Preference Steering)** personalizes LLM outputs by steering generation through detected "preference heads" - attention heads that are most sensitive to user preferences.

---

## Quick Start (3 steps)

### 1. ✅ Preference heads already detected for LaMP-1!

Located at: `/scratch/weixuz/dps/preference_head/preference_scores/Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json`

### 2. Test DPS (5 samples, ~2 minutes)

```bash
cd /scratch/weixuz/dps
bash experiments/test_dps.sh
```

### 3. Run full experiment (SLURM, ~30 minutes)

```bash
cd /scratch/weixuz/dps
sbatch experiments/run_dps.sh
```

---

## What files were created?

### Core DPS Implementation
- `/scratch/weixuz/dps/src/models/dps.py` - DPS decoder
- `/scratch/weixuz/dps/src/models/__init__.py` - Updated to import DPS

### Configuration Files
- `/scratch/weixuz/dps/configs/decoder/dps.yaml` - Base DPS config
- `/scratch/weixuz/dps/configs/experiment/lamp_1/dps/llama3_8b_instruct.yaml` - LaMP-1 config
- `/scratch/weixuz/dps/configs/experiment/lamp_2/dps/llama3_8b_instruct.yaml` - LaMP-2 config
- `/scratch/weixuz/dps/configs/experiment/lamp_3/dps/llama3_8b_instruct.yaml` - LaMP-3 config
- `/scratch/weixuz/dps/configs/experiment/lamp_4/dps/llama3_8b_instruct.yaml` - LaMP-4 config

### Scripts
- `/scratch/weixuz/dps/test_dps.sh` - Quick test script
- `/scratch/weixuz/dps/run_dps.sh` - Full experiment script (SLURM)

### Documentation
- `/scratch/weixuz/dps/DPS_INTEGRATION.md` - Comprehensive documentation
- `/scratch/weixuz/dps/DPS_QUICK_START.md` - This file

---

## How to run DPS on different tasks?

### LaMP-1 (Ready Now ✅)
```bash
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct
```

### LaMP-2, LaMP-3, LaMP-4 (Detect heads first)

**Step 1**: Detect preference heads for the task
```bash
cd /scratch/weixuz/dps/preference_head
# Edit run_detection.sh: uncomment the LaMP-2/3/4 section
sbatch run_detection.sh
```

**Step 2**: Run DPS experiment
```bash
cd /scratch/weixuz/dps
python scripts/main.py experiment=lamp_2/dps/llama3_8b_instruct
```

---

## Key Parameters

### Number of Preference Heads
```bash
# Default: 40 heads
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct

# Conservative: 20 heads
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=20

# Aggressive: 60 heads
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=60
```

### Steering Strength Cap
```bash
# No cap (default)
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct

# Cap at 3.0 (prevent over-steering)
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.alpha_cap=3.0
```

---

## Compare with Baselines

```bash
# Baseline (no steering)
python scripts/main.py experiment=lamp_1/baseline/llama3_8b_instruct

# DeCoRe (context steering)
python scripts/main.py experiment=lamp_1/decore_entropy/llama3_8b_instruct \
  decoder.configs.num_retrieval_heads=50

# DPS (preference steering) ⭐
python scripts/main.py experiment=lamp_1/dps/llama3_8b_instruct \
  decoder.configs.num_preference_heads=40

# Evaluate all
python evaluate_predictions.py
```

---

## Troubleshooting

### "Preference heads file not found"
You need to detect preference heads first:
```bash
cd /scratch/weixuz/dps/preference_head
python preference_head_detection.py --task LaMP-X
```

### Check if preference heads exist
```bash
ls -lh /scratch/weixuz/dps/preference_head/preference_scores/
```

Should see: `Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json`

---

## File Locations Summary

| File Type | Location |
|-----------|----------|
| Preference Heads | `/scratch/weixuz/dps/preference_head/preference_scores/` |
| DPS Decoder | `/scratch/weixuz/dps/src/models/dps.py` |
| DPS Configs | `/scratch/weixuz/dps/configs/experiment/lamp_*/dps/` |
| Test Script | `/scratch/weixuz/dps/test_dps.sh` |
| Run Script | `/scratch/weixuz/dps/run_dps.sh` |
| Output | `/scratch/weixuz/dps/outputs/` |

---

## Integration Status

| Task | Preference Heads | Config | Status |
|------|------------------|--------|--------|
| LaMP-1 | ✅ Detected | ✅ Created | ✅ Ready |
| LaMP-2 | ❌ Not yet | ✅ Created | ⚠️ Need detection |
| LaMP-3 | ❌ Not yet | ✅ Created | ⚠️ Need detection |
| LaMP-4 | ❌ Not yet | ✅ Created | ⚠️ Need detection |

---

## Next Steps

1. **Test on LaMP-1**: `bash experiments/test_dps.sh`
2. **Run full LaMP-1 experiment**: `sbatch experiments/run_dps.sh`
3. **Detect heads for other tasks**: Edit `/scratch/weixuz/dps/preference_head/run_detection.sh`
4. **Compare with baselines**: Run baseline + DeCoRe + DPS

---

## Full Documentation

See `/scratch/weixuz/dps/DPS_INTEGRATION.md` for:
- Architecture details
- Parameter tuning guide
- Advanced usage examples
- Extension to other tasks
- Performance benchmarks

---

## Questions?

Check these files:
- `DPS_INTEGRATION.md` - Full documentation
- `src/models/dps.py` - Implementation
- `configs/decoder/dps.yaml` - Configuration
- `/scratch/weixuz/dps/preference_head/preference_head.md` - Detection methodology

