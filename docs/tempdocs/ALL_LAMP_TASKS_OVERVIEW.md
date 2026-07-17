# Complete LaMP & LongLaMP Tasks Overview

## Summary

There are **11 total personalization tasks** available:
- **7 LaMP tasks** (LaMP-1 through LaMP-7)
- **4 LongLaMP tasks** (LongLaMP-1 through LongLaMP-4)

Currently, **DPS is only implemented for LaMP-1, 2, 3, 4** (4 out of 11 tasks).

---

## LaMP Tasks (7 tasks)

### ✅ LaMP-1: Citation Identification
- **Type**: Classification (Binary)
- **Task**: Predict which paper to cite based on user's writing history
- **Output**: Citation ID `[1]` or `[2]`
- **Metric**: Accuracy, F1
- **Status**: ✅ **DPS Implemented & Tested**
- **Dataset**: `/scratch/weixuz/lamp_data/LaMP-1/`

### ✅ LaMP-2: Movie Tagging  
- **Type**: Multi-label Classification
- **Task**: Predict movie tags based on user's rating history
- **Output**: Tags (sci-fi, comedy, action, horror, romance, thriller, drama, social commentary, violence, true story)
- **Metric**: Multi-label F1, Accuracy
- **Status**: ✅ **DPS Config Ready** (needs preference head detection)
- **Dataset**: `/scratch/weixuz/lamp_data/LaMP-2/`

### ✅ LaMP-3: Score Prediction
- **Type**: Regression
- **Task**: Predict rating score (1-5) based on user's review history
- **Output**: Score (1, 2, 3, 4, 5)
- **Metric**: MAE, RMSE
- **Status**: ✅ **DPS Config Ready** (needs preference head detection)
- **Dataset**: `/scratch/weixuz/lamp_data/LaMP-3/`

### ✅ LaMP-4: News Headline Generation
- **Type**: Text Generation
- **Task**: Generate headline based on user's article history
- **Output**: Headline text (~64 tokens)
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ✅ **DPS Config Ready** (needs preference head detection)
- **Dataset**: `/scratch/weixuz/lamp_data/LaMP-4/`

### ❌ LaMP-5: Scholarly Title Generation
- **Type**: Text Generation
- **Task**: Generate paper title based on user's publication history
- **Output**: Title text
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ❌ **Not Implemented**
- **Dataset**: `/scratch/weixuz/lamp_data/LaMP-5/`

### ❌ LaMP-6: Email Subject Generation (Avocado)
- **Type**: Text Generation
- **Task**: Generate email subject based on user's email history
- **Output**: Subject line
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ❌ **Not Implemented** (requires Avocado dataset access)
- **Dataset**: `/scratch/weixuz/lamp_data/LaMP-6/`
- **Note**: Avocado dataset is not publicly accessible (requires LDC license)

### ❌ LaMP-7: Tweet Paraphrasing
- **Type**: Text Generation
- **Task**: Paraphrase tweet based on user's tweet history
- **Output**: Paraphrased tweet
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ❌ **Not Implemented**
- **Dataset**: `/scratch/weixuz/lamp_data/LaMP-7/`

---

## LongLaMP Tasks (4 tasks)

LongLaMP extends LaMP with **longer user profiles** and **longer generation tasks**.

### ❌ LongLaMP-1: Email Generation
- **Type**: Long-form Text Generation
- **Task**: Generate full email based on user's email history
- **Output**: Complete email (longer than LaMP-6)
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ❌ **Not Implemented**
- **Dataset**: HuggingFace `LongLaMP/LongLaMP` (email_generation_user)

### ❌ LongLaMP-2: Abstract Generation
- **Type**: Long-form Text Generation
- **Task**: Generate paper abstract based on user's publication history
- **Output**: Abstract text
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ❌ **Not Implemented**
- **Dataset**: HuggingFace `LongLaMP/LongLaMP` (abstract_generation_user)

### ❌ LongLaMP-3: Topic Writing
- **Type**: Long-form Text Generation
- **Task**: Write on a topic based on user's writing history
- **Output**: Essay/article text
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ❌ **Not Implemented**
- **Dataset**: HuggingFace `LongLaMP/LongLaMP` (topic_writing_user)

### ❌ LongLaMP-4: Product Review Generation
- **Type**: Long-form Text Generation
- **Task**: Generate product review based on user's review history
- **Output**: Review text
- **Metric**: ROUGE-1, ROUGE-L, METEOR
- **Status**: ❌ **Not Implemented**
- **Dataset**: HuggingFace `LongLaMP/LongLaMP` (product_review_user)

---

## Task Type Summary

| Type | Tasks | Status |
|------|-------|--------|
| **Classification** | LaMP-1, LaMP-2 | ✅ 2/2 configs ready |
| **Regression** | LaMP-3 | ✅ 1/1 config ready |
| **Short Generation** | LaMP-4, LaMP-5, LaMP-6, LaMP-7 | ✅ 1/4 config ready |
| **Long Generation** | LongLaMP-1, 2, 3, 4 | ❌ 0/4 implemented |
| **TOTAL** | 11 tasks | ✅ 4/11 (36%) |

---

## Implementation Status

### ✅ Currently Implemented (4 tasks)
1. **LaMP-1**: Citation ID - DPS working
2. **LaMP-2**: Movie Tags - Config ready, needs detection
3. **LaMP-3**: Score Prediction - Config ready, needs detection  
4. **LaMP-4**: News Headlines - Config ready, needs detection

### ⏳ Easy to Add (3 tasks)
These tasks have datasets available and follow similar patterns:

5. **LaMP-5**: Paper Title Generation
   - Similar to LaMP-4 (generation task)
   - Dataset available in `/scratch/weixuz/lamp_data/LaMP-5/`
   - **Effort**: ~2 hours (create config + detect heads)

6. **LaMP-7**: Tweet Paraphrasing
   - Similar to LaMP-4 (generation task)
   - Dataset available in `/scratch/weixuz/lamp_data/LaMP-7/`
   - **Effort**: ~2 hours (create config + detect heads)

### 🔒 Requires Special Access (1 task)
7. **LaMP-6**: Email Subject (Avocado)
   - Requires LDC license for Avocado dataset
   - Dataset not publicly accessible
   - **Effort**: N/A (blocked by data access)

### 📦 Requires New Integration (4 tasks)
LongLaMP tasks require loading from HuggingFace datasets:

8. **LongLaMP-1**: Email Generation
9. **LongLaMP-2**: Abstract Generation
10. **LongLaMP-3**: Topic Writing
11. **LongLaMP-4**: Product Review

- **Effort per task**: ~4 hours (dataset integration + config + detection)
- **Total effort**: ~16 hours for all 4 LongLaMP tasks

---

## Priority Recommendations

### High Priority (Quick Wins)
Add LaMP-5 and LaMP-7 since:
- ✅ Datasets already available locally
- ✅ Follow same pattern as LaMP-4
- ✅ Minimal code changes needed
- ⏱️ ~4 hours total for both

### Medium Priority (Research Value)
Add LongLaMP tasks since:
- 📊 Tests DPS on longer contexts
- 🔬 More challenging personalization
- 📈 Shows scalability of preference heads
- ⏱️ ~16 hours for all 4 tasks

### Low Priority
LaMP-6 (Avocado):
- 🔒 Requires dataset access
- ⏸️ Skip unless you have LDC license

---

## Next Steps to Complete All Tasks

### Phase 1: Complete Standard LaMP (LaMP-5, LaMP-7)
**Goal**: 6/7 LaMP tasks (86% coverage)

```bash
# 1. Create configs
cp /scratch/weixuz/dps/configs/data/lamp_4.yaml \
   /scratch/weixuz/dps/configs/data/lamp_5.yaml

cp /scratch/weixuz/dps/configs/data/lamp_4.yaml \
   /scratch/weixuz/dps/configs/data/lamp_7.yaml

# 2. Update task names in configs
# Edit lamp_5.yaml: name: LAMP_5
# Edit lamp_7.yaml: name: LAMP_7

# 3. Create experiment configs
mkdir -p /scratch/weixuz/dps/configs/experiment/lamp_5/dps
mkdir -p /scratch/weixuz/dps/configs/experiment/lamp_7/dps

# 4. Detect preference heads
cd /scratch/weixuz/dps/preference_head
python preference_head_detection.py --task LaMP-5 --num_samples 400
python preference_head_detection.py --task LaMP-7 --num_samples 400

# 5. Run DPS experiments
cd /scratch/weixuz/dps
python scripts/main.py experiment=lamp_5/dps/llama3_8b_instruct
python scripts/main.py experiment=lamp_7/dps/llama3_8b_instruct
```

**Time**: ~4-6 hours

### Phase 2: Add LongLaMP Support
**Goal**: 10/11 tasks (91% coverage)

1. **Update dataset loader** to support LongLaMP
   - Modify `/scratch/weixuz/dps/src/datasets/lamp.py`
   - Add LongLaMP dataset loading from HuggingFace

2. **Create configs** for LongLaMP-1, 2, 3, 4
   - Similar to LaMP configs but with longer context

3. **Detect preference heads** for each LongLaMP task

4. **Run experiments**

**Time**: ~16-20 hours

---

## File Structure After Full Implementation

```
decore/
├── configs/
│   ├── data/
│   │   ├── lamp_1.yaml  ✅
│   │   ├── lamp_2.yaml  ✅
│   │   ├── lamp_3.yaml  ✅
│   │   ├── lamp_4.yaml  ✅
│   │   ├── lamp_5.yaml  ⏳ TODO
│   │   ├── lamp_7.yaml  ⏳ TODO
│   │   ├── longlamp_1.yaml  ⏳ TODO
│   │   ├── longlamp_2.yaml  ⏳ TODO
│   │   ├── longlamp_3.yaml  ⏳ TODO
│   │   └── longlamp_4.yaml  ⏳ TODO
│   └── experiment/
│       ├── lamp_1/dps/  ✅
│       ├── lamp_2/dps/  ✅
│       ├── lamp_3/dps/  ✅
│       ├── lamp_4/dps/  ✅
│       ├── lamp_5/dps/  ⏳ TODO
│       ├── lamp_7/dps/  ⏳ TODO
│       ├── longlamp_1/dps/  ⏳ TODO
│       ├── longlamp_2/dps/  ⏳ TODO
│       ├── longlamp_3/dps/  ⏳ TODO
│       └── longlamp_4/dps/  ⏳ TODO

preference_head/
└── preference_scores/
    ├── Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json  ✅
    ├── Meta-Llama-3-8B-Instruct_LaMP_2_top_heads.json  ⏳
    ├── Meta-Llama-3-8B-Instruct_LaMP_3_top_heads.json  ⏳
    ├── Meta-Llama-3-8B-Instruct_LaMP_4_top_heads.json  ⏳
    ├── Meta-Llama-3-8B-Instruct_LaMP_5_top_heads.json  ⏳ TODO
    ├── Meta-Llama-3-8B-Instruct_LaMP_7_top_heads.json  ⏳ TODO
    ├── Meta-Llama-3-8B-Instruct_LongLaMP_1_top_heads.json  ⏳ TODO
    ├── Meta-Llama-3-8B-Instruct_LongLaMP_2_top_heads.json  ⏳ TODO
    ├── Meta-Llama-3-8B-Instruct_LongLaMP_3_top_heads.json  ⏳ TODO
    └── Meta-Llama-3-8B-Instruct_LongLaMP_4_top_heads.json  ⏳ TODO
```

---

## Estimated Timeline for Full Coverage

| Phase | Tasks | Time | Completion |
|-------|-------|------|------------|
| **Current** | LaMP-1 | Done | ✅ 100% |
| **Phase 1a** | LaMP-2, 3, 4 detection | 2h | ⏳ In progress |
| **Phase 1b** | LaMP-2, 3, 4 experiments | 3h | ⏳ Pending |
| **Phase 2** | LaMP-5, 7 (full) | 6h | ⏳ TODO |
| **Phase 3** | LongLaMP-1,2,3,4 (full) | 20h | ⏳ TODO |
| **TOTAL** | 10/11 tasks | **~31h** | **91% coverage** |

---

## Research Questions by Task Type

### Classification Tasks (LaMP-1, 2)
- Do preference heads differ for citation vs. movie preferences?
- Are classification tasks easier/harder for DPS?

### Regression Task (LaMP-3)
- How do preference heads encode rating scales?
- Does DPS improve score calibration?

### Short Generation (LaMP-4, 5, 7)
- Do preference heads capture writing style?
- How do heads differ across domains (news, papers, tweets)?

### Long Generation (LongLaMP-1, 2, 3, 4)
- Do preference heads scale to longer contexts?
- Are more heads needed for longer generation?
- Does personalization quality degrade with length?

---

## Key Insights

1. **LaMP-1 through LaMP-4** are the core tasks most papers evaluate on
2. **LaMP-5 and LaMP-7** are easy additions (same infrastructure)
3. **LaMP-6** is blocked by data access (skip for now)
4. **LongLaMP** tasks are research extensions (longer contexts)
5. **lamp_benchmark already supports all tasks** - just need to add configs!

---

## Quick Command Reference

### Check available datasets
```bash
ls /scratch/weixuz/lamp_data/
```

### Add new task (template)
```bash
# 1. Copy config
cp configs/data/lamp_4.yaml configs/data/lamp_X.yaml

# 2. Edit config (change name to LAMP_X)
nano configs/data/lamp_X.yaml

# 3. Create experiment config
mkdir -p configs/experiment/lamp_X/dps
cp configs/experiment/lamp_4/dps/llama3_8b_instruct.yaml \
   configs/experiment/lamp_X/dps/llama3_8b_instruct.yaml

# 4. Detect heads
cd /scratch/weixuz/dps/preference_head
python preference_head_detection.py --task LaMP-X --num_samples 400

# 5. Run experiment
cd /scratch/weixuz/dps
python scripts/main.py experiment=lamp_X/dps/llama3_8b_instruct
```

---

**Current Status**: 4/11 tasks (36%) - **LaMP-1, 2, 3, 4**  
**Next Goal**: 6/11 tasks (55%) - **Add LaMP-5, 7**  
**Final Goal**: 10/11 tasks (91%) - **Add all LongLaMP tasks**
