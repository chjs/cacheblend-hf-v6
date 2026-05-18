# Phase 4 — MuSiQue RAG comparison

Model: `mistralai/Mistral-7B-Instruct-v0.2`
dtype: `bfloat16`
input: `/workspace/data/musique_ans_v1.0_train.jsonl`
num_examples_requested: 5
num_examples_used: 5
num_examples_skipped: 0
seed: 42
blend_special_str: `# #`
embedding_model: `sentence-transformers/all-MiniLM-L6-v2`
embedding_normalize: false
max_new_tokens: 32

| Method | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |
|--------|---------|--------|-----------------|----------------|
| Full recompute | 0.8000 | 1.0000 | 288.55 | 189.67 |
| Full KV reuse | 0.6000 | 1.0000 | 254.49 | 235.34 |
| CacheBlend | 0.6000 | 1.0000 | 131.15 | 120.49 |

## Skip reasons

(no skips)

## Quality gaps
- CacheBlend minus Full recompute: -0.2000
- Full KV reuse minus Full recompute: -0.2000
- CacheBlend minus Full KV reuse: +0.0000

## Latency ratios
- Full KV reuse / Full recompute: 0.882
- CacheBlend / Full recompute:   0.455
- CacheBlend / Full KV reuse:    0.515
