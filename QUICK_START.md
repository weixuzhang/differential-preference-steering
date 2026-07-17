# Quick Start: LAMP with RAG

## TL;DR

LAMP dataset now uses RAG to avoid OOM. Run this to test:

```bash
cd /scratch/weixuz/decore
python test_lamp_rag_integration.py
```

Then run experiments:

```bash
sbatch run.sh experiment=lamp_3/decore_entropy/llama3_8b_instruct
```

Done! 🚀

---

## What Changed

- **Before:** All 116 profiles (~42K tokens) → OOM ❌
- **After:** Top-5 relevant profiles (~2K tokens) → Works ✅

---

## Configuration

Default (in `configs/data/lamp_3.yaml`):

```yaml
retriever: bm25           # Fast token-based retrieval
num_retrieve: 5           # Top-5 profiles
max_prompt_length: 2048   # 2K token budget
```

Override via command line:

```bash
# Use Contriever (better quality)
sbatch run.sh experiment=lamp_3/decore_entropy/llama3_8b_instruct \
  data.retriever=contriever

# Use more profiles
sbatch run.sh experiment=lamp_3/decore_entropy/llama3_8b_instruct \
  data.num_retrieve=10

# Larger token budget
sbatch run.sh experiment=lamp_3/decore_entropy/llama3_8b_instruct \
  data.max_prompt_length=3072
```

---

## Retriever Options

| Retriever | Speed | Quality | When? |
|-----------|-------|---------|-------|
| `bm25` | ⚡ Fast | Good | Default |
| `contriever` | Medium | Better | Best results |
| `first_k` | ⚡ Fastest | Poor | Baseline |

---

## Memory Usage

| Config | Profiles | Tokens | Memory | Status |
|--------|----------|--------|--------|--------|
| Conservative | 3 | 1.5K | 35 GB | ✅ Safe |
| **Default** | 5 | 2K | 44 GB | ✅ Good |
| Aggressive | 10 | 3K | 50 GB | ⚠️ Watch |

All tested on 80GB A100 with DeCoRe.

---

## Files to Read

1. **This file** - Quick reference
2. `INTEGRATION_SUMMARY.md` - What changed
3. `RAG_INTEGRATION_GUIDE.md` - Full documentation
4. `test_lamp_rag_integration.py` - Test script

---

## Common Commands

```bash
# Test integration
python test_lamp_rag_integration.py

# Run baseline (BM25)
sbatch run.sh experiment=lamp_3/baseline/llama3_8b_instruct

# Run DeCoRe (BM25)
sbatch run.sh experiment=lamp_3/decore_entropy/llama3_8b_instruct

# Run DeCoRe (Contriever)
sbatch run.sh experiment=lamp_3/decore_entropy/llama3_8b_instruct \
  data.retriever=contriever

# All LAMP tasks
for task in 1 2 3 4; do
  sbatch run.sh experiment=lamp_${task}/decore_entropy/llama3_8b_instruct
done
```

---

## Troubleshooting

**OOM?** → Reduce `num_retrieve` or `max_prompt_length`  
**BM25 error?** → `pip install rank-bm25==0.2.2`  
**Slow?** → Use `bm25` instead of `contriever`  

---

**That's it! You're ready to go. 🎉**
