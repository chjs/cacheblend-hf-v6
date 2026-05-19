# Phase 4 — Loong RAG comparison

Model: `mistralai/Mistral-7B-Instruct-v0.2`
dtype: `bfloat16`
dataset: `loong`
input: `/workspace/data/loong_financial.jsonl`
num_examples_requested: 50
num_examples_used: 50
num_examples_skipped: 0
seed: 42
blend_special_str: `# #`
loong_num_chunks: 3
loong_first_order: original
loong_on_extra_chunks: first
loong_on_fewer_chunks: use_all
max_model_len: 32768
max_new_tokens: 32
safety_margin: 128
prompt_token_budget: 32608
dummy_warmup_query: `This is a cache warmup query. Do not answer.`
cacheblend_recompute_ratios: `0.00,0.05,0.15,0.30,0.50,1.00`

| Method | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |
|--------|---------|--------|-----------------|----------------|
| Full recompute | 0.0828 | 0.0000 | 1782.43 | 1894.00 |
| Full KV reuse | 0.1318 | 0.0000 | 1814.08 | 1921.64 |
| CacheBlend r=0.15 | 0.1128 | 0.0000 | 464.75 | 481.11 |

## CacheBlend ratio sweep

| Ratio | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |
|-------|---------|--------|-----------------|----------------|
| r=0.00 | 0.1335 | 0.0000 | 249.58 | 255.60 |
| r=0.05 | 0.1478 | 0.0000 | 315.79 | 325.94 |
| r=0.15 | 0.1128 | 0.0000 | 464.75 | 481.11 |
| r=0.30 | 0.0707 | 0.0000 | 689.75 | 723.51 |
| r=0.50 | 0.0816 | 0.0000 | 1016.70 | 1067.94 |
| r=1.00 | 0.0695 | 0.0000 | 1813.77 | 1925.54 |

## Failure-only subset

Examples where Full KV reuse F1 < Full recompute F1. This is the subset where naive reuse loses quality, so it is where we expect CacheBlend's selective recomputation to help.

- 갯수: 7
- Full recompute F1 mean (subset): 0.2574
- Full KV reuse F1 mean (subset): 0.0119
- CacheBlend r=0.00 F1 mean (subset): 0.0249
- CacheBlend r=0.05 F1 mean (subset): 0.1820
- CacheBlend r=0.15 F1 mean (subset): 0.1820
- CacheBlend r=0.30 F1 mean (subset): 0.0712
- CacheBlend r=0.50 F1 mean (subset): 0.2439
- CacheBlend r=1.00 F1 mean (subset): 0.2574
- Best CacheBlend ratio (subset): `cacheblend_r1.00` @ 0.2574

## Prompt length buckets

| Bucket | n | Full F1 | Reuse F1 | r=0.15 F1 | Full ms | Reuse ms | r=0.15 ms |
|--------|---|---------|----------|-----------|---------|----------|-----------|
| 0-8k | 50 | 0.0828 | 0.1318 | 0.1128 | 1782.4 | 1814.1 | 464.8 |

## Segment cache diagnostics

- evaluated examples: 50
- all chunks cache-hit: 50
- real question segment cache-hit: 0
- prefix cache-hit: 50

## Sanity checks

- CacheBlend r=0.00 vs Full KV reuse F1 gap: +0.0017
- CacheBlend r=1.00 vs Full recompute F1 gap: -0.0133

## Quality gaps (overall)
- CacheBlend r=0.15 minus Full recompute: +0.0299
- Full KV reuse minus Full recompute:     +0.0490
- CacheBlend r=0.15 minus Full KV reuse:  -0.0191

## Latency ratios (overall)
- Full KV reuse / Full recompute: 1.018
- CacheBlend r=0.00 / Full recompute: 0.140
- CacheBlend r=0.05 / Full recompute: 0.177
- CacheBlend r=0.15 / Full recompute: 0.261
- CacheBlend r=0.30 / Full recompute: 0.387
- CacheBlend r=0.50 / Full recompute: 0.570
- CacheBlend r=1.00 / Full recompute: 1.018

## Skip reasons

(no skips)
