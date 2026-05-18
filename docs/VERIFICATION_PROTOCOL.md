# Verification Protocol

Each phase has explicit pass/fail criteria. A phase is **not done** until its
criteria pass on both target models (Mistral-7B and Llama-3.1-8B), unless
noted otherwise.

For all dtype tolerances below, use `torch.allclose(a, b, rtol=R, atol=A)`.

## Phase 1 — Layerwise forward skeleton

Tests live in `tests/test_phase1_layerwise.py`.

1. `compute_layer` runs to completion for both target models with a 64-token
   random prompt without raising.
2. The generator yields exactly `num_hidden_layers` times.
3. After the generator is exhausted, the final hidden state has the expected
   shape `(num_tokens, hidden_size)`.
4. Memory does not grow unboundedly across iterations (verify by running
   `compute_layer` twice on the same input and checking peak memory delta
   stays within ~1.5× the first-run peak; treat this as a smoke test, not a
   strict bound — eager attention has cuBLAS workspace fluctuations).
5. The stub blender is wired in: `process_qkv` is called once per layer
   with the expected argument shapes. Returned q, k have RoPE applied
   (since Phase 1 stub now includes RoPE — see PHASE1_PROMPT.md "Review
   notes"); k differs from the input only by the rotation.

These tests use only Phase 1 code (no CacheBlend logic, no segment DB).

## Phase 2 — Equivalence with stock HF forward

Tests live in `tests/test_phase2_equivalence.py`.

Setup for each model and dtype:
- Load model with `torch_dtype=<dtype>`, `attn_implementation="eager"`.
- Generate a fixed-seed 128-token random prompt.
- Run stock HF forward in prefill mode with `use_cache=True`,
  `output_hidden_states=True`; capture
  - `outputs.past_key_values` — a `Cache` object in transformers ≥4.36
    (a `DynamicCache` in 5.x). Use the `extract_kv` helper from
    `PHASE2_PROMPT.md` to retrieve per-layer K, V.
  - `outputs.hidden_states[-1]` (final hidden state, **after**
    `model.norm`).
- Run `compute_layer` to completion via a thin driver that, with the stub
  blender wrapped in a recording proxy, mirrors what real `blend_layer`
  would do but without any KV cache lookup. Apply `model.model.norm` to
  the final hidden state before comparing to `outputs.hidden_states[-1]`,
  since `compute_layer` does not run the final norm.

Compare:

| dtype | rtol  | atol  | applies to                       |
|-------|-------|-------|----------------------------------|
| fp32  | 1e-5  | 1e-5  | hidden states, K, V              |
| fp16  | 1e-3  | 1e-3  | hidden states, K, V              |
| bf16  | 1e-2  | 1e-2  | hidden states, K, V              |

Pass criteria:
1. fp16 **and** bf16 pass `allclose` for hidden states and K, V on
   Mistral-7B (real model).
2. fp16 **and** bf16 pass `allclose` for hidden states and K, V on
   Llama-3.1-8B (real model).
3. fp32 passes for hidden states and K, V on the **tiny CPU model**
   (`TestTinyModelEquivalence`). Real-model fp32 is deliberately
   excluded: Mistral-7B-Instruct-v0.2 in fp32 is ~29 GB and
   Meta-Llama-3.1-8B-Instruct in fp32 is ~32 GB; neither fits on a
   24 GB GPU. The tiny model exercises the same fp32 numerical
   tolerance (1e-5) on the same code path.
4. The max absolute error per layer is logged for inspection.

## Phase 3 — CacheBlend port correctness

Tests live in `tests/test_phase3_blender.py`.

### 3.1 `SegmentTokenDatabase`

- Given a prompt
  ```
  "SYS" + SEP + "CHUNK_A" + SEP + "CHUNK_B" + SEP + "Q"
  ```
  tokenized with each target model's tokenizer, `process_tokens` yields
  exactly 4 chunks (SYS, CHUNK_A, CHUNK_B, Q) with correct `(start, end)`
  spans that do not overlap with separator tokens.
- Hash keys for `CHUNK_A` are identical across two independent calls with
  the same tokens (determinism).
- Hash keys for the same chunk text but a different surrounding context
  are identical (no prefix chaining in SegmentTokenDatabase).

### 3.2 `FusedRope` round-trip

- For random K, applying RoPE at positions `p1` and then `FusedRope` from
  `p1` → `p2` produces the same tensor as applying RoPE directly at `p2`,
  within fp32 atol 1e-5.

### 3.3 HKVD selection determinism

- With fixed seeds, calling `process_qkv` on the check layer with the same
  inputs produces the same `top_indices` and `topk_num` across runs.
- `topk_num == max(int(total_len * recomp_ratios[0]), 1)`.
- `top_indices` is sorted ascending.

### 3.4 KV merge correctness

- After `process_qkv` on the check layer:
  - `old_k[top_indices]` equals the post-RoPE freshly computed K at those
    indices (exact equality, same dtype).
  - `old_k` rows at indices not in `top_indices` are byte-identical to the
    pre-call values (no unintended writes).
  - Same for V.

### 3.5 End-to-end blending: 100% cache hit, single chunk

When the input prompt is a single chunk whose KV was stored on a previous
call, and `check_layers=[1]`, `recomp_ratios=[1.0]` (force recompute of
all tokens):

- The fused KV cache equals the full-prefill KV cache within
  fp32 atol 1e-4 (eager attention numerical drift only).

### 3.6 End-to-end blending: realistic ratio

With the standard config (`check_layers=[1]`, `recomp_ratios=[0.15]`) and
a 4-chunk prompt where every chunk has cached KV:

- `blend()` runs to completion without error.
- `metadata.imp_indices`, `metadata.positions`, `metadata.attn_mask` are
  `None` after completion.
- The fused KV cache differs from full prefill, but the cosine similarity
  per layer between the fused K (and V) and full-prefill K (V), averaged
  over tokens, is ≥ 0.95.
- The next-token logit (argmax) matches full prefill's next-token logit on
  at least 80% of a 20-prompt smoke set on Mistral-7B in bf16.

These thresholds are sanity checks, not formal guarantees. The point is to
catch gross regressions. Phase 4 produces the real quality numbers.

## Phase 4 — RAG quality comparison

Test/script: `scripts/run_rag_comparison.py` (driver) and
`tests/test_phase4_rag_quality.py` (regression assertion).

Setup:
- Dataset: 2WikiMultihopQA (use `datasets` library, subset of 50 examples
  for the smoke test, 200 for the full report).
- For each example, build the prompt
  ```
  SYS + SEP + ctx_1 + SEP + ctx_2 + ... + SEP + ctx_k + SEP + Q
  ```
  with `k` retrieved context chunks (use the gold supporting paragraphs as
  the chunks for this experiment — retrieval is out of scope).
- Tokenize, truncate contexts as needed to fit the model's max context.
- For each method below, generate up to 32 tokens with `do_sample=False`.

Methods:
1. **Full KV recompute** — stock HF prefill on the concatenated prompt.
2. **Full KV reuse** — concatenate per-chunk pre-computed KV caches with
   RoPE position-shift only, no recomputation. (Implement as a trivial
   variant: same loading path but skip `process_qkv`'s HKVD branch.)
3. **CacheBlend** — the port from Phase 3.

For each method, record:
- F1 score against the gold short answer (LongBench-style F1).
- Prefill latency (CUDA time, averaged over the run).

Output: a markdown table in `results/phase4.md`. Pass criteria:

1. Full recompute F1 reproduces a sensible number (≥ 0.30 on Mistral-7B
   Instruct, ≥ 0.30 on Llama-3.1-8B Instruct — these are sanity floors,
   not state-of-the-art targets).
2. Full reuse F1 is **lower** than Full recompute (showing the cross-
   attention loss). If it isn't, something is wrong with the experiment.
3. CacheBlend F1 is within **0.03** absolute of Full recompute. If the
   gap is larger than 0.05, the port has a bug or the recompute ratio /
   check layer needs tuning — re-check `process_qkv`.
4. CacheBlend prefill latency is **lower** than Full recompute (eager
   attention is slow, but with HKVD reduction the after-check-layer
   compute is small; this should still win).

Phase 4's regression test asserts criteria (1), (3), and the smoke set
passes. Criterion (4) is logged but not asserted (eager perf is noisy).

---

## Running verification

Each phase prompt lists the exact commands. The convention is:

```bash
cd /workspace/cacheblend_hf
pytest tests/test_phase{N}_*.py -v
```

For Phase 4 also:

```bash
python scripts/run_rag_comparison.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --num-examples 50 \
    --dtype bfloat16
```

---

## Review notes

- Phase 1 §5: relaxed "returns (q, k, v) unchanged" because the stub
  blender now applies RoPE to (q, k) (resolved in PHASE1_PROMPT.md).
  Test should assert RoPE happened, not that k is byte-identical to the
  pre-call value.
- Phase 1 §4: replaced the open-ended "memory does not grow
  unboundedly" with a concrete smoke threshold (1.5× of first peak).
- Phase 2: updated the capture description to use the
  `extract_kv(cache, i)` helper (Cache object in transformers ≥4.36)
  and noted that `compute_layer` skips `model.norm`, so the test must
  apply it to the captured final hidden state before comparing.
- Phase 2 pass criteria: explicitly excluded **fp32 real-model**
  runs because Mistral-7B (~29 GB) / Llama-3.1-8B (~32 GB) in fp32
  don't fit on a 24 GB GPU. The fp32 tolerance is still exercised by
  the CPU tiny model. Discovered while running Phase 2 on vast.ai
  RTX 3090 (instance 37009319).
