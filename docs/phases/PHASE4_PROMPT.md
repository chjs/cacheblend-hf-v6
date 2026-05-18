# Phase 4 — RAG quality comparison

## Context

Read `README.md`, `docs/LMCACHE_IMPLEMENTATION.md`, `docs/CODING_CONVENTIONS.md`,
`docs/VERIFICATION_PROTOCOL.md`. Phases 1–3 must be complete with passing
tests.

Prime directive applies. Make decisions yourself.

## Goal

Run a small RAG-style experiment that compares three prefill methods on
2WikiMultihopQA and reports F1 quality plus prefill latency. The
experiment validates that the CacheBlend port behaves as expected
end-to-end.

The three methods (all use the same model and greedy decode):
1. **Full KV recompute** — stock HF prefill on the concatenated prompt.
2. **Full KV reuse** — concatenate per-chunk pre-computed KV caches with
   RoPE position-shift only, no recomputation.
3. **CacheBlend** — the port from Phase 3 (`check_layers=[1]`,
   `recomp_ratios=[0.15]`).

## Prompt construction

For each example in 2WikiMultihopQA:

```
SYSTEM_PROMPT + SEP + ctx_1 + SEP + ctx_2 + ... + SEP + ctx_k + SEP + QUESTION
```

Where:
- `SYSTEM_PROMPT` is fixed:
  `"You are a helpful assistant. Answer the question concisely based on the given context."`
- `SEP` is `" # # "` (the LMCache default `blend_special_str`).
- `ctx_i` are the example's supporting paragraphs. Use the gold
  supporting paragraphs (`supporting_facts`) directly — retrieval is out
  of scope.
- `QUESTION` is `"Question: <question>\nAnswer:"`.

Truncate per-context to keep total tokens within model max minus a 64-token
generation budget. Discard examples that don't fit at the smallest chunk
size.

## Implementation

`scripts/run_rag_comparison.py`:

```
usage: run_rag_comparison.py
    --model {mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Meta-Llama-3.1-8B-Instruct}
    --num-examples N
    --dtype {float16, bfloat16}
    --output results/phase4.md
```

The script:
1. Loads the model with `attn_implementation="eager"`.
2. Builds the `LMCacheEngine`, `HFBufferLayerwiseGPUConnector`,
   `LMCBlender` per `docs/LMCACHE_IMPLEMENTATION.md`.
3. For each example:
   a. Tokenize the full prompt; split conceptually into chunks by SEP.
   b. **Warmup pass**: full prefill the prompt using stock HF, capture
      per-layer K, V, then use `LMCacheEngine.store_from_prefill` to
      populate the per-chunk KV cache.
   c. **Method 1 — Full recompute**: stock HF `generate` from `input_ids`,
      `max_new_tokens=32`, `do_sample=False`. Record TTFT (prefill ms)
      and the generated text.
   d. **Method 2 — Full KV reuse**: load per-chunk KVs into the
      connector buffer with RoPE shift, skip blender recompute (pass
      hidden states through with no `process_qkv` HKVD), then use the
      resulting fused KV as the model's `past_key_values` and generate
      32 tokens. Record TTFT and text.
   e. **Method 3 — CacheBlend**: `blender.blend(tokens)`, then generate
      32 tokens from the fused KV. Record TTFT and text.
4. For each example and method, compute F1 vs the gold answer using
   LongBench's F1 normalization (lowercase, strip punctuation, split
   tokens, compute token-level F1).
5. Aggregate and write `results/phase4.md`:

```
# Phase 4 — RAG comparison

Model: <model>, dtype: <dtype>, num_examples: <N>

| Method            | F1 (mean) | Prefill ms (mean) | Prefill ms (p50) |
|-------------------|-----------|-------------------|------------------|
| Full recompute    | ...       | ...               | ...              |
| Full KV reuse     | ...       | ...               | ...              |
| CacheBlend        | ...       | ...               | ...              |
```

## Method 2 implementation note

"Full KV reuse" is the **Prompt Cache / Gim et al. 2023** baseline
referenced in the CacheBlend paper: concatenate per-chunk pre-computed
KVs with a RoPE position-shift so each chunk's K is rotated from its
cached position to its position in this request, but do **no**
recomputation of K or V. It can be implemented as a variant of the
CacheBlend path that uses the same retrieval + GPU buffer loading +
RoPE shift, but **skips** the HKVD branch entirely. The cleanest port:
a second blender subclass `LMCFullReuseBlender` whose `process_qkv`
runs the RoPE step on the freshly computed Q only, then returns
`(q, old_k, old_v, residual, attn_output, attn_metadata)` — i.e. the
attention path consumes the cached (rotated) K and V untouched and the
newly computed K, V from the live forward are discarded. Same loading
code path; only the recompute differs.

Concretely:

```python
class LMCFullReuseBlender(LMCBlender):
    def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        # Only Q is rotated freshly; cached K was rotated to new positions
        # already by the gpu_connector via FusedRope.
        q, _ = layer.self_attn.rotary_emb(self.metadata.positions, q, k)
        return q, old_k, old_v, residual, attn_output, attn_metadata
```

Note: this returns full-length q (no HKVD restriction), so the attention
backend receives `n_q == n_k` and uses the standard causal mask — no
`update_from_top_indices` call.

## Test

`tests/test_phase4_rag_quality.py`:

- Runs the comparison script on `num_examples=10` with Mistral-7B in
  bf16. Smoke test.
- Asserts:
  1. F1 of Full recompute ≥ 0.20 (sanity floor — depends on the model,
     but well below state-of-the-art is fine; we're testing the harness).
  2. F1 of CacheBlend is within **0.05** absolute of Full recompute.
  3. F1 of Full reuse is **at most** F1 of CacheBlend (Full reuse should
     not beat CacheBlend; if it does on small N due to noise, allow a
     0.03 margin).
  4. Prefill latency of CacheBlend is finite and the run completes.

## Verification

```bash
# Smoke test
pytest tests/test_phase4_rag_quality.py -v

# Full run on both models, both dtypes (latency numbers in results/)
python scripts/run_rag_comparison.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --num-examples 200 \
    --dtype bfloat16 \
    --output results/phase4_mistral_bf16.md

python scripts/run_rag_comparison.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-examples 200 \
    --dtype bfloat16 \
    --output results/phase4_llama_bf16.md
```

## 작업 보고

작업이 끝나면 `reports/phase4.md` 를 작성하라. 양식은
`docs/CODING_CONVENTIONS.md` §"Phase reports (Korean)" 참고. 보고서의
"검증 결과" 섹션에는 (1) smoke 테스트의 assertion 별 pass/fail, (2)
세 방법 (Full recompute / Full reuse / CacheBlend) 의 F1 평균과
prefill latency (mean, p50) 를 표로, (3) `results/phase4_*.md`
경로를 명시하라. CacheBlend 가 Full recompute 대비 F1 갭 0.03 초과면
"4. 작업 중 결정한 사항" 에 원인 가설을 적어라. 보고서 작성 후 stdout
에는 한 줄 안내만 출력하라.

---

## Review notes

- Made the Full-KV-reuse definition concrete by citing its origin (the
  Prompt Cache / Gim et al. 2023 baseline that the CacheBlend paper
  positions itself against) and showing a complete
  `LMCFullReuseBlender.process_qkv` body. The original prompt's
  "implementation note" left the Q-vs-K rotation question
  under-specified — Q must be rotated freshly with `rotary_emb`, K
  must be the cached K already rotated by `FusedRope` in the GPU
  connector. Also clarified that this path takes the no-HKVD branch
  so the attention backend sees `n_q == n_k` and the metadata is not
  `update_from_top_indices`'d.
