## DPS Ablation Study Guide
# Number of Preference Heads

## 📊 **Overview**

This ablation study investigates the effect of **different numbers of preference heads** on DPS performance.

### **Research Question**
*How does the number of preference heads affect personalization performance?*

### **Experimental Design**
- **Variable**: Number of preference heads
- **Values tested**: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
- **Model**: LLaMA-3-8B-Instruct
- **Tasks**: All 10 tasks (LaMP-1,2,3,4,5,7 + LongLaMP-1,2,3,4)
- **Total experiments**: 100 (10 head counts × 10 tasks)

---

## 🎯 **Hypotheses**

### **H1: Optimal Range**
There exists an optimal range of preference heads (likely 30-60) that balances:
- **Too few heads** (< 20): Insufficient personalization signal
- **Too many heads** (> 80): Noise and overfitting

### **H2: Task Dependency**
Different task types may require different numbers of heads:
- **Classification**: Fewer heads (20-40) for clearer decision boundaries
- **Generation**: More heads (40-80) for richer stylistic control

### **H3: Diminishing Returns**
Performance improvement will plateau or decline after a certain threshold.

---

## 📁 **What Was Created**

### **Experiment Configs** (100 files)
```
For each of 10 tasks:
  configs/experiment/{task}/dps_ablation/
  ├── llama3_8b_instruct_heads10.yaml
  ├── llama3_8b_instruct_heads20.yaml
  ├── llama3_8b_instruct_heads30.yaml
  ├── llama3_8b_instruct_heads40.yaml
  ├── llama3_8b_instruct_heads50.yaml
  ├── llama3_8b_instruct_heads60.yaml
  ├── llama3_8b_instruct_heads70.yaml
  ├── llama3_8b_instruct_heads80.yaml
  ├── llama3_8b_instruct_heads90.yaml
  └── llama3_8b_instruct_heads100.yaml
```

### **Scripts** (3 new)
- ✅ `decore/run_ablation_study.sh` - Run complete ablation study
- ✅ `decore/analyze_ablation.py` - Analyze results
- ✅ `decore/test_ablation.sh` - Quick test

### **Documentation** (1 new)
- ✅ `ABLATION_STUDY_GUIDE.md` - This file

---

## 🚀 **How to Run**

### **Option 1: Complete Ablation Study** (~30-40 hours)

Run all 100 experiments:

```bash
cd /scratch/weixuz/decore
sbatch run_ablation_study.sh
```

**What this does**:
- Tests 10 different head counts on all 10 tasks
- Automatically analyzes results
- Generates summary and recommendations

**Time**: ~30-40 hours (100 experiments × ~20-30 min each)

---

### **Option 2: Single Task Ablation** (~3-5 hours)

Test all head counts on one task:

```bash
cd /scratch/weixuz/decore
TASK="lamp_1"
MODEL="llama3_8b_instruct"

for num_heads in 10 20 30 40 50 60 70 80 90 100; do
  python scripts/main.py \
    experiment=${TASK}/dps_ablation/${MODEL}_heads${num_heads}
done

# Analyze results
python analyze_ablation.py
```

**Time**: ~3-5 hours (10 experiments × ~20-30 min each)

---

### **Option 3: Quick Test** (~5 minutes)

Test a few head counts with small samples:

```bash
cd /scratch/weixuz/decore
bash test_ablation.sh
```

**What this does**:
- Tests 10, 40, and 100 heads on LaMP-1
- Uses only 5 samples per experiment
- Verifies configs work correctly

**Time**: ~5 minutes

---

### **Option 4: Custom Ablation**

Test specific head counts on specific tasks:

```bash
cd /scratch/weixuz/decore

# Example: Test 30, 40, 50 heads on LaMP-1 and LaMP-2
for task in lamp_1 lamp_2; do
  for num_heads in 30 40 50; do
    python scripts/main.py \
      experiment=${task}/dps_ablation/llama3_8b_instruct_heads${num_heads}
  done
done
```

---

## 📊 **Expected Results**

### **Performance Curves**

We expect to see different patterns for different task types:

#### **Classification Tasks (LaMP-1, LaMP-2)**
```
Accuracy
  ^
  |     ___________
  |    /           \___
  |   /                \
  |  /                  \
  | /                    \
  +-----------------------> # Heads
  0  20  40  60  80  100

Optimal: 30-50 heads
```

#### **Generation Tasks (LaMP-4,5,7, LongLaMP)**
```
ROUGE-L
  ^
  |        ___________
  |       /           \
  |      /             \
  |     /               \__
  |    /                   
  +-----------------------> # Heads
  0  20  40  60  80  100

Optimal: 40-70 heads
```

#### **Regression Task (LaMP-3)**
```
MAE
  ^
  |                    __
  |                  /
  |                /
  |  \___________ /
  |              
  +-----------------------> # Heads
  0  20  40  60  80  100

Optimal: 30-50 heads (lower is better)
```

---

## 📈 **Analysis Outputs**

After running the ablation study, you'll get:

### **1. Ablation Analysis JSON**
```
outputs/ablation_analysis.json
```

Contains:
- Optimal number of heads for each task
- Performance across all head counts
- Summary statistics
- Recommendations

### **2. Console Summary**
```
ABLATION STUDY SUMMARY: Number of Preference Heads
================================================================================

Classification Tasks:
--------------------------------------------------------------------------------
  LaMP-1          - Optimal:  40 heads (accuracy=0.4523)
    Performance: 10h:0.401  20h:0.428  30h:0.445  40h:0.452⭐ 50h:0.448  ...

  LaMP-2          - Optimal:  50 heads (f1=0.5234)
    Performance: 10h:0.478  20h:0.502  30h:0.515  40h:0.520  50h:0.523⭐ ...

Regression Task:
--------------------------------------------------------------------------------
  LaMP-3          - Optimal:  30 heads (mae=0.8234)
    Performance: 10h:0.912  20h:0.856  30h:0.823⭐ 40h:0.831  50h:0.845  ...

...

RECOMMENDATIONS
================================================================================
  Average optimal number of heads: 45.2
  Range: 30 - 60 heads

  Recommendation: Use 45 preference heads for best overall performance
```

### **3. Prediction Files**
```
outputs/
├── pred_LAMP_1_LLaMA3-8b-Instruct__DPS_heads10.json
├── pred_LAMP_1_LLaMA3-8b-Instruct__DPS_heads20.json
├── ...
└── pred_LongLaMP_4_LLaMA3-8b-Instruct__DPS_heads100.json
```

---

## 🔬 **Research Questions to Answer**

### **Primary Questions**

1. **What is the optimal number of preference heads?**
   - Overall optimal
   - Task-specific optimal
   - Task-type patterns

2. **How does performance change with head count?**
   - Linear, logarithmic, or plateau?
   - Diminishing returns threshold?
   - Performance degradation point?

3. **Do different tasks need different head counts?**
   - Classification vs generation
   - Short vs long generation
   - Task complexity effects

### **Secondary Questions**

4. **Is there a universal optimal?**
   - Can one head count work well for all tasks?
   - Trade-off between task-specific and universal?

5. **What's the sensitivity to head count?**
   - How much does performance vary?
   - Robustness to suboptimal choices?

6. **Computational trade-offs?**
   - Speed vs performance
   - Memory vs accuracy

---

## 📊 **Analysis Methods**

### **1. Performance Curves**
Plot performance vs number of heads for each task:
```python
import matplotlib.pyplot as plt
import json

with open('outputs/ablation_analysis.json') as f:
    data = json.load(f)

for task, results in data['detailed_results'].items():
    heads = sorted(results['results_by_num_heads'].keys())
    values = [results['results_by_num_heads'][h]['metric_value'] for h in heads]
    
    plt.plot(heads, values, marker='o', label=task)

plt.xlabel('Number of Preference Heads')
plt.ylabel('Performance')
plt.legend()
plt.savefig('ablation_curves.png')
```

### **2. Optimal Head Distribution**
Analyze distribution of optimal head counts:
```python
optimal_counts = [
    opt['optimal_num_heads'] 
    for opt in data['optimal_per_task'].values()
]

plt.hist(optimal_counts, bins=10)
plt.xlabel('Optimal Number of Heads')
plt.ylabel('Number of Tasks')
plt.savefig('optimal_distribution.png')
```

### **3. Task Type Comparison**
Compare optimal heads by task type:
```python
classification = ['LaMP-1', 'LaMP-2']
regression = ['LaMP-3']
short_gen = ['LaMP-4', 'LaMP-5', 'LaMP-7']
long_gen = ['LongLaMP-1', 'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4']

for task_type, tasks in [
    ('Classification', classification),
    ('Regression', regression),
    ('Short Gen', short_gen),
    ('Long Gen', long_gen)
]:
    optimal = [
        data['optimal_per_task'][t]['optimal_num_heads']
        for t in tasks if t in data['optimal_per_task']
    ]
    print(f"{task_type}: avg={np.mean(optimal):.1f}, range={min(optimal)}-{max(optimal)}")
```

---

## ⏱️ **Timeline**

| Scope | Experiments | Time per Exp | Total Time |
|-------|-------------|--------------|------------|
| Quick test | 3 | ~2 min | ~5 min |
| Single task | 10 | ~20-30 min | ~3-5 hours |
| All tasks | 100 | ~20-30 min | ~30-40 hours |

**Recommendation**: Start with single task to verify, then run full study.

---

## 💡 **Best Practices**

### **1. Start Small**
Always test with a small sample first:
```bash
python scripts/main.py \
  experiment=lamp_1/dps_ablation/llama3_8b_instruct_heads40 \
  data.num_samples=10 \
  debug=true
```

### **2. Monitor Progress**
Check job status and output:
```bash
# Check SLURM job
squeue -u $USER

# Monitor output
tail -f dps_ablation_*.out

# Count completed experiments
ls outputs/pred_*__DPS_heads*.json | wc -l
```

### **3. Incremental Analysis**
Analyze results as they complete:
```bash
# Run analysis on partial results
python analyze_ablation.py

# Check specific task
grep "LaMP-1" outputs/ablation_analysis.json
```

### **4. Prioritize Tasks**
If time is limited, prioritize diverse task types:
```bash
# Run on representative tasks first
for task in lamp_1 lamp_3 lamp_4 longlamp_1; do
  for num_heads in 10 20 30 40 50 60 70 80 90 100; do
    python scripts/main.py \
      experiment=${task}/dps_ablation/llama3_8b_instruct_heads${num_heads}
  done
done
```

---

## 🎯 **Expected Findings**

Based on the DPS methodology, we expect:

### **1. Optimal Range: 30-60 heads**
- **Rationale**: Balances signal and noise
- **Evidence**: Top 4% of ~1000 heads ≈ 40 heads

### **2. Task-Specific Variation**
- **Classification**: 30-50 heads (clearer patterns)
- **Short generation**: 40-60 heads (style control)
- **Long generation**: 50-70 heads (richer context)

### **3. Diminishing Returns**
- **Threshold**: ~60-70 heads
- **Beyond 80**: Performance plateau or decline

### **4. Robustness**
- **±10 heads**: Minimal performance change
- **±20 heads**: Noticeable but acceptable
- **±40 heads**: Significant degradation

---

## ✅ **Success Criteria**

### **Technical Success**
- ✅ All 100 experiments complete successfully
- ✅ Analysis script runs without errors
- ✅ Clear performance curves generated
- ✅ Optimal heads identified for each task

### **Research Success**
- ✅ Clear optimal range identified (e.g., 30-60 heads)
- ✅ Task-type patterns observed
- ✅ Diminishing returns threshold found
- ✅ Recommendations for future experiments

---

## 📚 **Quick Commands**

### **Run Full Study**
```bash
cd /scratch/weixuz/decore
sbatch run_ablation_study.sh
```

### **Test First**
```bash
cd /scratch/weixuz/decore
bash test_ablation.sh
```

### **Analyze Results**
```bash
cd /scratch/weixuz/decore
python analyze_ablation.py
```

### **Check Status**
```bash
# Count completed
ls outputs/pred_*__DPS_heads*.json | wc -l

# View analysis
cat outputs/ablation_analysis.json | jq '.optimal_per_task'
```

---

## 🎉 **Summary**

### **What You Have**
- ✅ 100 ablation experiment configs
- ✅ Automated experiment runner
- ✅ Analysis and visualization scripts
- ✅ Comprehensive documentation

### **What You'll Learn**
- 📊 Optimal number of preference heads
- 📈 Performance sensitivity to head count
- 🎯 Task-specific requirements
- 💡 Best practices for DPS configuration

### **Research Value**
- **Methodological**: Validates DPS design choices
- **Practical**: Guides future experiments
- **Theoretical**: Insights into preference encoding
- **Publication**: Strong ablation study for paper

---

**Ready to start?**

```bash
cd /scratch/weixuz/decore && bash test_ablation.sh
```

**Good luck! 🎓**

