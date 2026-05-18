# Project Plan

The project ports LMCache's CacheBlend to HuggingFace Transformers in four
phases. Phases are strictly sequential — each one depends on the previous
one passing its verification.

The guiding rule applies to every phase:

> Follow LMCache. Do not redesign. Adapt only where the HF API forces it.
> Keep class names, function names, signatures, and field names identical
> to LMCache.

---

## Phase 1 — Layerwise forward in HF

**Goal**: build `LMCBaseModel.compute_layer(input_ids)` as a generator that
runs a HF Llama model one transformer layer at a time, yielding after each
layer. No CacheBlend yet — just the layerwise prefill skeleton, structured
exactly like `lmcache/v1/compute/models/base.py`.

**Produces**:
- `lmc/compute/models/base.py` — `LMCBaseModel`
- `lmc/compute/models/llama.py` — `LMCLlamaModel`
- `lmc/compute/models/utils.py` — `infer_model_from_hf` (parallel to
  `infer_model_from_vllm`)
- `lmc/compute/attention/eager.py` — eager attention backend with the
  `forward_contiguous(q, k, v, output, attn_metadata)` interface from
  LMCache. No HKVD logic yet; just full attention.
- `lmc/compute/attention/metadata.py` — `LMCAttnMetadata`,
  `LMCEagerAttnMetadata` (parallel to `LMCFlashAttnMetadata`).
- A minimal stub `Blender` that passes (q, k, v) through unchanged so that
  `compute_layer`'s call site is the same as LMCache.

**Done when** Phase 1 unit tests pass (see `VERIFICATION_PROTOCOL.md` §1).

---

## Phase 2 — Equivalence with stock HF forward

**Goal**: prove that running `compute_layer` to completion produces the same
prefill output (final hidden states **and** final per-layer K, V) as calling
the stock `LlamaForCausalLM.forward` on the same input.

**Produces**:
- `tests/test_phase2_equivalence.py`
- Documented numerical tolerances per dtype.

**Done when** Phase 2 tests pass on both Mistral-7B and Llama-3.1-8B in
fp32, fp16, and bf16.

---

## Phase 3 — Port CacheBlend

**Goal**: port `LMCBlender`, `LMCBlendMetadata`, `LMCBlendCommonMetadata`,
`SegmentTokenDatabase`, `FusedRope`, layerwise retrieval, and the GPU
buffer connector — all behaviorally faithful to LMCache.

**Produces**:
- `lmc/compute/blend/blender.py` — `LMCBlender` (replaces Phase 1 stub)
- `lmc/compute/blend/metadata.py` — `LMCBlendCommonMetadata`,
  `LMCBlendMetadata` (Phase 1 already declares the fields)
- `lmc/compute/blend/utils.py` — `LMCBlenderBuilder`
- `lmc/compute/positional_encoding.py` — `FusedRope`,
  `validate_rope_params`, `get_fused_rope`. `BasicReverseRope` /
  `validate_reverse_correctness` are optional (no callers in the hot
  path; include for parity if you want).
- `lmc/token_database.py` — `SegmentTokenDatabase`, base
  `TokenDatabase`, and the minimum types they reference:
  `CacheEngineKey` (with `split_layers`), `LMCacheMetadata`, the
  `ProcessTokensResult` alias, `NONE_HASH = 0`.
- `lmc/cache_engine.py` — minimal `LMCacheEngine` with `retrieve_layer`
  and a `store_from_prefill` (or `store_layer`) helper that populates
  per-chunk KVs after a warmup full prefill.
- `lmc/storage.py` — in-memory `LocalCPUBackend` keyed by per-layer
  subkey; `MemoryObj` with `tensor` shape `(2, chunk_len, num_kv_heads * head_dim)`
  and `metadata.cached_positions`; a `MemoryFormat.KV_2TD` sentinel.
- `lmc/gpu_connector.py` — `HFBufferLayerwiseGPUConnector` (parallel
  to `VLLMBufferLayerwiseGPUConnector`) that holds a per-layer KV
  buffer dict and applies `FusedRope` to cached K during load. Same
  `num_layers + 2` yield contract.
- `lmc/config.py` — `LMCacheEngineConfig` dataclass with the
  env-mapped fields used by blending (see `LMCACHE_IMPLEMENTATION.md`
  §2.9 for the exact list).
- `lmc/integration/hf/utils.py` — `HFModelTracker`, `ENGINE_NAME`.
- Wire `LMCBaseModel.compute_layer` to call `blender.process_qkv`
  between QKV and attention (already done in Phase 1; in Phase 3 the
  call site is unchanged — the blender just gets swapped out).

**Done when** Phase 3 unit tests pass (see `VERIFICATION_PROTOCOL.md` §3).

---

## Phase 4 — RAG quality comparison

**Goal**: end-to-end experiment that exercises the CacheBlend path under a
realistic RAG-style scenario, compares against Full KV recompute and Full
KV reuse baselines, and reports F1 quality plus latency.

**Produces**:
- `scripts/run_rag_comparison.py`
- `tests/test_phase4_rag_quality.py`
- A short results table in `results/phase4.md` written by the script.

**Done when** Phase 4 verification passes (see `VERIFICATION_PROTOCOL.md` §4).

---

## Review notes

- Phase 1 deliverables (separate file): the stub blender now declares
  the full `LMCBlendMetadata` dataclass and applies RoPE inside
  `process_qkv`. See PHASE1_PROMPT.md "Review notes".
- Phase 3 deliverables: expanded the list with the small auxiliary
  types the original draft elided — `CacheEngineKey.split_layers`,
  `LMCacheMetadata`, `MemoryObj.metadata.cached_positions`,
  `MemoryFormat.KV_2TD`. Without these the layerwise retrieve path
  doesn't compile.
