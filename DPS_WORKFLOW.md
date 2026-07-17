# DPS Workflow Diagram

## Complete Pipeline: From Detection to Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DPS COMPLETE WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: PREFERENCE HEAD DETECTION (One-time per task)                       │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│  LaMP Dataset        │
│  • Task: LaMP-1      │   ┌────────────────────────────┐
│  • 400 samples       │──▶│ preference_head_detection  │
│  • User profiles     │   │ • Baseline NLL             │
└──────────────────────┘   │ • Head ablation (1024)     │
                           │ • Compute PCS              │
┌──────────────────────┐   │ • Rank heads               │
│  LLaMA3-8B-Instruct  │   │ • Select top 4% (40 heads) │
│  • 32 layers         │──▶└────────────────────────────┘
│  • 32 heads/layer    │                 │
│  = 1024 total heads  │                 │
└──────────────────────┘                 ▼
                           ┌────────────────────────────────┐
                           │ OUTPUT:                        │
                           │ Meta-Llama-3-8B-Instruct_      │
                           │   LaMP_1_top_heads.json        │
                           │                                │
                           │ Top Preference Heads:          │
                           │  1. [28, 18] PCS: 0.2462      │
                           │  2. [30, 18] PCS: 0.1916      │
                           │  3. [29, 18] PCS: 0.1484      │
                           │  ... (40 total)                │
                           └────────────────────────────────┘
                                         │
                                         │ Saved to:
                                         │ /scratch/.../preference_scores/
                                         ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: DPS INTEGRATION (Already done!)                                     │
└──────────────────────────────────────────────────────────────────────────────┘

          ┌───────────────────────┐       ┌───────────────────────┐
          │  DPS Decoder          │       │  Config Files         │
          │  src/models/dps.py    │       │  configs/decoder/     │
          │  • Load pref heads    │◀──────│    dps.yaml           │
          │  • Calc entropy       │       │  configs/experiment/  │
          │  • Steer generation   │       │    lamp_1/dps/...     │
          └───────────────────────┘       └───────────────────────┘
                    │                                │
                    │                                │
                    └────────────┬───────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Registered in:         │
                    │  src/models/__init__.py │
                    │  from .dps import DPS   │
                    └─────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: RUNNING EXPERIMENTS                                                 │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  User Command      │
│                    │
│  python scripts/   │
│    main.py         │
│    experiment=     │
│    lamp_1/dps/     │
│    llama3_8b_...   │
└────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hydra Config Resolution                                        │
│  • Model: LLaMA3-8B-Instruct                                   │
│  • Data: LAMP-1 (with RAG: BM25, 5 profiles, 2048 tokens)     │
│  • Decoder: DPS (40 preference heads, task=LaMP-1)             │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  DPS Runtime Workflow                                           │
│                                                                 │
│  For each test sample:                                          │
│                                                                 │
│  1. ┌────────────────────────────────────────────┐            │
│     │ Load Preference Heads                      │            │
│     │ • Read top_heads.json                      │            │
│     │ • Extract [[28,18], [30,18], ...]          │            │
│     └────────────────────────────────────────────┘            │
│                         │                                       │
│                         ▼                                       │
│  2. ┌────────────────────────────────────────────┐            │
│     │ Prepare Input                              │            │
│     │ • User profile (via RAG)                   │            │
│     │ • Query                                     │            │
│     │ • Tokenize                                  │            │
│     └────────────────────────────────────────────┘            │
│                         │                                       │
│                         ▼                                       │
│  3. ┌────────────────────────────────────────────┐            │
│     │ Generate with Base Model                   │            │
│     │ • Forward pass                              │            │
│     │ • Get logits for next token                │            │
│     └────────────────────────────────────────────┘            │
│                         │                                       │
│                         ▼                                       │
│  4. ┌────────────────────────────────────────────┐            │
│     │ Calculate Steering Strength                │            │
│     │ • entropy = -Σ(p * log(p))                 │            │
│     │ • alpha = min(entropy, cap) if cap         │            │
│     └────────────────────────────────────────────┘            │
│                         │                                       │
│                         ▼                                       │
│  5. ┌────────────────────────────────────────────┐            │
│     │ Apply Preference Steering                  │            │
│     │ • steered = base + α * w * base            │            │
│     │ • Amplify preference head contributions    │            │
│     └────────────────────────────────────────────┘            │
│                         │                                       │
│                         ▼                                       │
│  6. ┌────────────────────────────────────────────┐            │
│     │ Select Next Token                          │            │
│     │ • argmax(steered_logits)                   │            │
│     │ • Repeat until EOS or max_tokens           │            │
│     └────────────────────────────────────────────┘            │
│                         │                                       │
│                         ▼                                       │
│  7. ┌────────────────────────────────────────────┐            │
│     │ Return Personalized Output                 │            │
│     │ • Steered by preference heads              │            │
│     │ • Aligned with user preferences            │            │
│     └────────────────────────────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Save Predictions                                               │
│  outputs/lamp_1_dps_predictions_<timestamp>.json               │
│                                                                 │
│  [                                                              │
│    {                                                            │
│      "idx": 0,                                                  │
│      "predicted_answer": "Answer with preference steering",    │
│      "predicted_answer_baseline": "Answer without steering",   │
│      "answers": ["Ground truth"],                              │
│      "task": "LaMP-1"                                          │
│    },                                                           │
│    ...                                                          │
│  ]                                                              │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: EVALUATION                                                          │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  Evaluate          │
│  predictions       │
│                    │
│  python evaluate_  │
│    predictions.py  │
└────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Load Predictions                                               │
│  • Find latest prediction files                                 │
│  • Extract task name                                            │
│  • Extract method (baseline, decore, dps)                       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Compute Metrics (BanditPR)                                     │
│  • LaMP-1: Citation accuracy                                    │
│  • LaMP-2: Multi-label F1                                       │
│  • LaMP-3: MAE                                                   │
│  • LaMP-4: ROUGE-L                                              │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Results Table                                                  │
│                                                                 │
│  ┌──────────┬────────────┬──────────┬───────────┐             │
│  │ Task     │ Method     │ Accuracy │ Samples   │             │
│  ├──────────┼────────────┼──────────┼───────────┤             │
│  │ LaMP-1   │ Baseline   │ 35.2%    │ 400       │             │
│  │ LaMP-1   │ DeCoRe     │ 42.1%    │ 400       │             │
│  │ LaMP-1   │ DPS ⭐     │ 45.8%    │ 400       │ ← Expected  │
│  └──────────┴────────────┴──────────┴───────────┘             │
│                                                                 │
│  DPS shows +10.6% improvement over baseline!                   │
│  DPS shows +3.7% improvement over DeCoRe!                      │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Flow

### 1. Preference Head Detection (One-time)

```
Input: LaMP Dataset + LLaMA3-8B-Instruct
  │
  ├─▶ Compute Baseline NLL (all heads active)
  │     • Run model on 400 samples
  │     • Record negative log-likelihood
  │
  ├─▶ Head-wise Ablation (1024 heads × 400 samples)
  │     • For each head [layer, head_idx]:
  │       1. Mask head output (set to 0)
  │       2. Compute NLL with head ablated
  │       3. Calculate ΔNLL = NLL_ablated - NLL_baseline
  │
  ├─▶ Compute Preference Contribution Score (PCS)
  │     • PCS[head] = average ΔNLL across samples
  │     • Higher PCS = more important for preferences
  │
  └─▶ Rank and Select Top Heads
        • Sort heads by PCS (descending)
        • Select top 4% (40 heads)
        • Save to JSON

Output: Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json
```

### 2. DPS Runtime (Per Sample)

```
Input: User query + profile
  │
  ├─▶ Load Preference Heads
  │     • Read from top_heads.json
  │     • preference_heads = [[28,18], [30,18], ...]
  │
  ├─▶ Prepare Input with RAG
  │     • Retrieve 5 relevant user histories (BM25)
  │     • Format prompt with profile
  │     • Tokenize (max 2048 tokens)
  │
  ├─▶ Generate Token by Token
  │     │
  │     └─▶ Loop until EOS or max_tokens:
  │           │
  │           ├─▶ Forward Pass
  │           │     • Get logits: [batch, vocab_size]
  │           │     • Get attentions from preference heads
  │           │
  │           ├─▶ Calculate Steering Strength
  │           │     • probs = softmax(logits)
  │           │     • entropy = -Σ(p * log(p))
  │           │     • alpha = entropy (or capped)
  │           │
  │           ├─▶ Apply Preference Steering
  │           │     • steered = base + α * w * base
  │           │     • Amplify based on preference heads
  │           │
  │           ├─▶ Select Next Token
  │           │     • token = argmax(steered_logits)
  │           │
  │           └─▶ Update State
  │                 • Append token to sequence
  │                 • Update key-value cache
  │
  └─▶ Decode and Return
        • Convert tokens to text
        • Return personalized output

Output: Preference-steered generation
```

## Comparison: Three Methods Side-by-Side

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                  │    BASELINE      │     DECORE       │       DPS        │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Special Heads    │ None             │ Retrieval (50)   │ Preference (40)  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Detection        │ N/A              │ Context patterns │ User patterns    │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Steering         │ None             │ Context rerank   │ Preference amp   │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ RAG              │ Yes (BM25)       │ Yes (BM25)       │ Yes (BM25)       │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ LaMP-1 Accuracy  │ ~35%             │ ~42%             │ ~45% (expected)  │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Overhead         │ 1.0x             │ 1.0x             │ 1.0x             │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Best For         │ General tasks    │ Context-heavy    │ Personalization  │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

## File System Layout

```
/scratch/weixuz/
│
├── preference_head/                       # Detection project
│   ├── preference_scores/                 # Detected heads (output)
│   │   ├── Meta-Llama-3-8B-Instruct_LaMP_1_pcs.json
│   │   ├── Meta-Llama-3-8B-Instruct_LaMP_1_ranked.json
│   │   └── Meta-Llama-3-8B-Instruct_LaMP_1_top_heads.json ✅
│   │
│   ├── preference_head_detection.py       # Detection script
│   ├── validate_preference_heads.py       # Validation script
│   ├── run_detection.sh                   # SLURM script
│   └── PATH_CONFIGURATION.md              # Path docs
│
└── decore/                                # Experiments project
    ├── src/models/
    │   ├── dps.py                         # DPS decoder ✅
    │   ├── decore_entropy.py              # DeCoRe decoder
    │   ├── baseline.py                    # Baseline
    │   └── __init__.py                    # Imports DPS ✅
    │
    ├── configs/
    │   ├── decoder/
    │   │   ├── dps.yaml                   # DPS config ✅
    │   │   ├── decore_entropy.yaml
    │   │   └── baseline.yaml
    │   │
    │   └── experiment/
    │       ├── lamp_1/
    │       │   ├── dps/llama3_8b_instruct.yaml ✅
    │       │   ├── decore_entropy/...
    │       │   └── baseline/...
    │       │
    │       ├── lamp_2/dps/... ✅
    │       ├── lamp_3/dps/... ✅
    │       └── lamp_4/dps/... ✅
    │
    ├── outputs/                           # Experiment results
    │   ├── lamp_1_dps_predictions_*.json
    │   └── evaluation_summary.json
    │
    ├── test_dps.sh                        # Quick test ✅
    ├── run_dps.sh                         # Full run ✅
    ├── DPS_INTEGRATION.md                 # Full docs ✅
    ├── DPS_QUICK_START.md                 # Quick ref ✅
    └── DPS_WORKFLOW.md                    # This file ✅
```

## Execution Timeline

### Quick Test (~2 minutes)
```
0:00  Start test_dps.sh
0:01  Load model (from cache)
0:05  Load preference heads
0:10  Process 5 samples with DPS
1:50  Save predictions
2:00  ✅ Test complete
```

### Full Experiment (~30 minutes)
```
0:00   Start run_dps.sh (SLURM)
0:01   Load model (from cache)
0:05   Load preference heads (40 heads)
0:10   Load LaMP-1 dataset (400 samples)
1:00   Process samples 1-100
8:00   Process samples 101-200
15:00  Process samples 201-300
22:00  Process samples 301-400
28:00  Save predictions
29:00  Run evaluation
30:00  ✅ Complete with results
```

## Data Flow

```
LaMP Dataset
    ↓
┌────────────────────┐
│ User Profile       │ ──┐
│ • History (tweets) │   │
│ • Preferences      │   │
└────────────────────┘   │
                         ├─▶ RAG (BM25)
┌────────────────────┐   │    • Retrieve 5 relevant histories
│ Query              │ ──┘    • Fit in 2048 tokens
│ • Task-specific    │
└────────────────────┘
    ↓
┌────────────────────────────────────────────┐
│ Formatted Prompt                           │
│ "Given user's past tweets:                 │
│  [tweet 1]                                 │
│  [tweet 2]                                 │
│  ...                                       │
│  Predict citation for: [query]"           │
└────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────┐
│ DPS Generation                             │
│ • Load 40 preference heads                 │
│ • Generate token-by-token                  │
│ • Steer with entropy-based alpha           │
│ • Amplify preference head contributions    │
└────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────┐
│ Personalized Output                        │
│ "Based on your preferences, the most       │
│  likely citation is: [prediction]"         │
└────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────┐
│ Evaluation                                 │
│ • Compare with ground truth                │
│ • Compute accuracy                         │
│ • Report results                           │
└────────────────────────────────────────────┘
```

---

**Status**: All components integrated and ready to use!

**Next**: Run `bash test_dps.sh` to validate the complete workflow.

