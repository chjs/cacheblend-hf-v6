# Phase 3 — Port CacheBlend

## Context

Read all reference docs first: `README.md`, `docs/LMCACHE_IMPLEMENTATION.md`,
`docs/CODING_CONVENTIONS.md`, `docs/VERIFICATION_PROTOCOL.md`. Phases 1 and 2 must
be complete with passing tests.

Prime directive: **follow LMCache verbatim**. Class names, method names,
field names, control flow, all match. Read each LMCache source file in
full before writing the port. Make small decisions yourself.

## What to port

Open each of these LMCache files and port them. The list below is the
order to work in.

1. `lmcache/v1/compute/blend/metadata.py`
   → `lmc/compute/blend/metadata.py`
   Both `LMCBlendCommonMetadata` and `LMCBlendMetadata` dataclasses,
   `clean()` method. Identical to LMCache.

2. `lmcache/v1/compute/positional_encoding.py`
   → `lmc/compute/positional_encoding.py`
   - `BasicReverseRope` (optional — used only inside
     `validate_reverse_correctness`; safe to omit if you skip that
     validator)
   - `FusedRope` (with `fused_encode` and `__call__`)
   - `validate_rope_params`
   - `validate_reverse_correctness` (optional)
   - `get_fused_rope`

   LMCache calls `vllm_get_rope` to build the underlying rope. Replace
   that with HF's rotary so we transparently get Llama-3 RoPE scaling
   (which `validate_rope_params` would otherwise reject — see
   `docs/LMCACHE_IMPLEMENTATION.md` §2.6). Recipe:

   1. Build `cos_sin_cache` once from the HF model:
      ```python
      max_pos = hf_model.config.max_position_embeddings
      positions_all = torch.arange(max_pos, device=device).unsqueeze(0)
      dummy = torch.empty(1, dtype=dtype, device=device)
      cos_all, sin_all = hf_model.model.rotary_emb(dummy, positions_all)
      # cos_all, sin_all: (1, max_pos, head_dim)
      ```
   2. `FusedRope.fused_encode(old_positions, new_positions, k)` does:
      ```python
      # k: (num_tokens, num_kv_heads * head_size) — same shape contract
      num_tokens = k.shape[0]
      k = k.view(num_tokens, -1, self.head_size)        # (T, H_kv, D)
      cos_old = cos_all[0, old_positions]                # (T, D)
      sin_old = sin_all[0, old_positions]
      cos_new = cos_all[0, new_positions]
      sin_new = sin_all[0, new_positions]
      # Reverse RoPE at old then apply RoPE at new.
      # For neox-style: apply_rope(x, cos, sin) = x*cos + rotate_half(x)*sin
      # Inverse: apply_rope(x, cos, -sin)
      k = apply_rope(k, cos_old.unsqueeze(1), -sin_old.unsqueeze(1))
      k = apply_rope(k, cos_new.unsqueeze(1),  sin_new.unsqueeze(1))
      k = k.view(num_tokens, -1)
      return k
      ```
      where `apply_rope(x, cos, sin) = x * cos + rotate_half(x) * sin`
      with `rotate_half` from `modeling_llama.py:138-142`.
   3. The kernel `lmc_ops.rotary_embedding_k_fused` mutates K in place
      AND returns it. Our pure-PyTorch port returns a new tensor (the
      caller in `batched_to_gpu` already assigns the return value to
      `compute_gpu_buffer_obj.tensor[0]`, so out-of-place is fine).

   The neox-style rotation matches HF's `apply_rotary_pos_emb` exactly
   (`modeling_llama.py:146-168`), which is what both the live HF rotary
   and our `FusedRope` should be doing.

3. `lmcache/v1/compute/blend/blender.py`
   → `lmc/compute/blend/blender.py`
   Replace the Phase 1 stub `LMCBlender` with the full implementation.
   Port `__init__`, `process_qkv`, `blend_layer`, `blend` verbatim
   following `docs/LMCACHE_IMPLEMENTATION.md` §2.3 and §2.4.

   Constructor signature (LMCache version):
   ```python
   def __init__(self, cache_engine, gpu_connector, vllm_model, config):
       ...
   ```
   In our port, rename `vllm_model` → `hf_model` only in the parameter
   name; everywhere inside the class refer to it as `self.vllm_model =
   hf_model` so all the internals (e.g.
   `self.layerwise_model.vllm_model.model.layers[layer_id]`) stay
   identical to LMCache. Build `self.layerwise_model` via
   `infer_model_from_hf(hf_model, self, enable_sparse)`.

4. `lmcache/v1/compute/blend/utils.py`
   → `lmc/compute/blend/utils.py`
   Port `LMCBlenderBuilder` (`get_or_create`, `get`). Uses `HFModelTracker`
   instead of `VLLMModelTracker`.

5. `lmcache/v1/token_database.py`
   → `lmc/token_database.py`
   Port the abstract `TokenDatabase` (the parts needed:
   `_canonicalize_hash_inputs`, `_hash_tokens`, `_make_key_by_hash`,
   `process_tokens` signature), the `ProcessTokensResult` type alias,
   the `NONE_HASH = 0` constant, and the full `SegmentTokenDatabase`:
   `__init__`, `_fast_split_by_subtensor`, `process_tokens`.

   Watch:
   - `SegmentTokenDatabase.__init__` takes
     `(config: LMCacheEngineConfig, metadata: LMCacheMetadata)`. Add a
     minimal `LMCacheMetadata` dataclass alongside (LMCache's lives at
     `lmcache/v1/metadata.py`) carrying at least `model_name: str` (used
     for `AutoTokenizer.from_pretrained`) and any other fields
     `_make_key_by_hash` needs (`world_size`, `worker_id`, `kv_dtype`).
     For our single-process port: `world_size=1`, `worker_id=0`,
     `kv_dtype=torch.float16` (or matching the model's dtype).
   - `CacheEngineKey` is a small dataclass in LMCache
     (`lmcache/utils.py`); port a minimal version with the same fields:
     `model_name`, `world_size`, `worker_id`, `chunk_hash`, `kv_dtype`,
     `request_configs`. It needs `split_layers(num_layers)` returning a
     list of per-layer subkeys (used by `retrieve_layer`). For the HF
     port, a per-layer subkey can just be `(key, layer_id)` since we
     own both store and retrieve.
   - The hash function: use Python's built-in `hash` on the
     canonicalized tuple, matching `hash_func` in LMCache's
     `TokenDatabase.__init__` when the config selects `"builtin"`. For
     a single-process test run this is fine; no `PYTHONHASHSEED`
     coordination needed.

6. `lmc/config.py`
   Port `LMCacheEngineConfig` with just the fields we use:
   - `chunk_size: int = 256`
   - `enable_blending: bool = False`
   - `blend_recompute_ratios: Optional[List[float]] = None`
   - `blend_thresholds: Optional[List[float]] = None`
   - `blend_check_layers: List[int] = field(default_factory=lambda: [1])`
   - `blend_min_tokens: int = 256`
   - `blend_special_str: str = " # # "`
   - `use_layerwise: bool = True`
   - `pre_caching_hash_algorithm: str = "builtin"`
   - `extra_config: Optional[dict] = None`

   Plus a `from_env()` classmethod that reads `LMCACHE_*` env vars (same
   names as LMCache).

7. `lmc/storage.py`
   Minimal `LocalCPUBackend` (in-memory dict). Keyed by `CacheEngineKey`
   per-layer subkey, values are `MemoryObj` instances that wrap a
   single layer's K, V tensors on CPU plus a small `metadata` carrying
   `cached_positions` (the positions the chunk was originally cached
   at) and `fmt = MemoryFormat.KV_2TD` (a tiny enum or sentinel mirroring
   LMCache's; only one format is used).

   `MemoryObj.tensor` shape is `(2, chunk_len, num_kv_heads * head_dim)`
   where `tensor[0]` is K and `tensor[1]` is V — this matches what
   `batched_to_gpu` reads at `gpu_connectors.py:880-888`.

   Provide methods: `contains(key) -> bool`, `put(key, memory_obj)`,
   `get(key) -> Optional[MemoryObj]`.

8. `lmc/gpu_connector.py`
   `HFBufferLayerwiseGPUConnector`. Equivalent to LMCache's
   `VLLMBufferLayerwiseGPUConnector` but without vLLM's paged-memory
   integration. Required interface:

   - `__init__(num_layers, hidden_dim_size_q, hidden_dim_size_kv, num_heads, num_kv_heads, head_dim, dtype, device, fused_rotary_emb)`.
   - `get_kv(layer_id) -> (K, V)`. K and V are GPU-resident tensors
     covering the full request token range `[starts[0], ends[-1])` with
     shape `(num_all_tokens, num_kv_heads * head_dim)` each. Mutating
     them in `process_qkv` (via `old_k[imp_indices] = k`) must be
     reflected on subsequent reads — i.e. `get_kv` must return views
     into the same underlying storage, not copies. Mirrors
     `gpu_connectors.py:729-739`.
   - `batched_to_gpu(starts, ends, **kwargs)` generator: yields
     **`num_layers + 2`** times (mirror LMCache's count exactly so
     `retrieve_layer`'s drive loop is unchanged). Operationally
     simpler than LMCache's ping-pong:

     1. Allocate a per-layer buffer dict `self.buffer_mapping:
        dict[int, (K, V)]` of size `num_all_tokens`, dtype, device.
     2. Compute `gap_mask` and `current_gap_positions = where(gap_mask)[0]`
        as in `gpu_connectors.py:792-800`.
     3. Compute `new_positions_full = torch.arange(starts[0], ends[-1])`.
     4. Yield once (the "first prime"), receiving nothing.
     5. For `layer_id in range(num_layers)`: receive
        `memory_objs_layer = yield` (a list of MemoryObj for this
        layer's chunks). Copy each chunk's K and V into the
        layer-`layer_id` buffer slot at `[start - buf_offset : end - buf_offset]`.
        Apply `fused_rotary_emb(old_positions_full, new_positions_full, K)`
        in place; then zero K and V at `current_gap_positions`. At
        `layer_id == 0` only, populate `old_positions_full` from each
        chunk's `metadata.cached_positions` (this is invariant across
        layers).
     6. Yield once for the synchronization point (LMCache's
        `layer_id == num_layers` branch), then yield once for the
        final flush (LMCache's `yield` after the loop). Both can be
        bare `yield`s in the port.

   - Strip vLLM paged-memory steps. We only need the buffer in GPU
     memory; we never write back to a paged store.
   - **Order matters**: RoPE first, then gap zeroing. LMCache zeros
     gaps **after** RoPE (`gpu_connectors.py:864-866`) so any garbage
     that the kernel rotated at gap positions is then cleared.

9. `lmc/cache_engine.py`
   Minimal `LMCacheEngine` providing:
   - `store(tokens, hidden_states_or_kv, ...)` — used after a full prefill
     warmup to populate per-chunk KVs. The store path runs the
     `SegmentTokenDatabase` over the tokens, splits the prefill K, V by
     chunk span, applies the *current* positions to each chunk slice's
     cached metadata (`cached_positions`), and inserts into
     `LocalCPUBackend`.
   - `retrieve_layer(tokens, mask=None, **kwargs)` — the layerwise
     retrieval generator from LMCache (§2.8 of the spec). Walks the
     SegmentTokenDatabase chunks, looks up each in CPU backend, builds the
     `(starts, ends, mem_objs_per_layer)` schedule, then drives the
     `gpu_connector.batched_to_gpu` generator in lockstep.

10. `lmc/integration/hf/utils.py`
    `ENGINE_NAME = "hf_cacheblend"` (LMCache uses
    `"vllm-instance"` at `lmcache/integration/vllm/utils.py:27`; the
    string is arbitrary, but keep it distinct so the two trackers
    can't collide in a process that imports both). `HFModelTracker`
    (already from Phase 1).

11. Wire `LMCBlenderBuilder.get_or_create` to construct the full
    pipeline: builds connector, cache engine, then blender. Adjust
    `LMCBaseModel.__init__` so that when given a real blender (not stub),
    its `compute_layer` runs against the connector's GPU buffer (via
    `blender.gpu_connector.get_kv`) as in LMCache.

## Tests

Write `tests/test_phase3_blender.py` covering every criterion in
`docs/VERIFICATION_PROTOCOL.md` §3 (six sub-sections). Each criterion is a
separate test function.

For sub-sections 3.5 and 3.6 (end-to-end), use a workflow like:

```python
# Warmup: full prefill on prompt, populate per-chunk KV store
engine = LMCacheEngine(config, ...)
engine.store_from_prefill(tokens, full_prefill_kv)

# CacheBlend prefill on the same (or a permuted) prompt
blender = LMCBlenderBuilder.get_or_create(ENGINE_NAME, engine, connector, config)
blender.blend(tokens)

# Compare fused KV (the connector's GPU buffer) to the full-prefill KV
```

## Verification

```bash
pytest tests/test_phase3_blender.py -v
```

Pass criteria per `docs/VERIFICATION_PROTOCOL.md` §3 (six sub-sections).

Do not proceed to Phase 4 without user review.

## 작업 보고

작업이 끝나면 `reports/phase3.md` 를 작성하라. 양식은
`docs/CODING_CONVENTIONS.md` §"Phase reports (Korean)" 참고. 보고서의
"검증 결과" 섹션에는 6 개 sub-section 별 pass/fail 와 tolerance 측정값
(특히 §3.5, §3.6 의 cosine similarity / next-token logit 일치율) 을
포함하라. "LMCache와의 일치도" 섹션에는 `FusedRope`, `process_qkv`,
`batched_to_gpu` 가 LMCache 원본과 어떻게 매핑되는지 line 단위로
정리하라. 보고서 작성 후 stdout 에는 한 줄 안내만 출력하라.

---

## Review notes

- Added a concrete pure-PyTorch recipe for `FusedRope.fused_encode`
  (the LMCache version is a CUDA kernel) with explicit reference to
  HF's `apply_rotary_pos_emb` / `rotate_half`. Built atop HF's rotary
  so Llama-3-family RoPE scaling works transparently — LMCache's
  `validate_rope_params` would otherwise reject those models.
- Spelled out the auxiliary types Phase 3 needs but the original
  prompt didn't list: `LMCacheMetadata`, the minimum
  `CacheEngineKey` (including `split_layers`), `MemoryObj` shape and
  `metadata.cached_positions`, and `MemoryFormat.KV_2TD`. Without
  these the port can't drive `retrieve_layer`/`batched_to_gpu`.
- Detailed `HFBufferLayerwiseGPUConnector.batched_to_gpu` operationally
  (still `num_layers + 2` yields) and made the RoPE-then-gap-zero
  order explicit. Also called out the `get_kv` aliasing requirement
  (must return views into the buffer, not copies) — `process_qkv`
  writes through this alias.
- Marked `BasicReverseRope` / `validate_reverse_correctness` as
  optional since they only feed `validate_reverse_correctness` and
  add no runtime value.
- Differentiated the HF `ENGINE_NAME` from LMCache's `"vllm-instance"`
  so trackers don't collide.
