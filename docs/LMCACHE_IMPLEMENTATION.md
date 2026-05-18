# LMCache CacheBlend — Implementation Specification

This document describes **what LMCache's CacheBlend does**. It is the source
of truth for the port. The guiding rule for the whole project is:

> **Follow LMCache. Do not redesign. Do not deviate. Do not "improve".**
> Adapt only when an HF API leaves no choice, and even then keep the same
> class names, function names, signatures, and field names as LMCache.

LMCache reference checkout (branch `fix/cacheblend-vllm-v0.17.1-compat`):

```
lmcache/v1/
├── compute/
│   ├── blend/
│   │   ├── blender.py        ← LMCBlender                          ★core
│   │   ├── metadata.py       ← LMCBlendCommonMetadata, LMCBlendMetadata
│   │   └── utils.py          ← LMCBlenderBuilder
│   ├── models/
│   │   ├── base.py           ← LMCBaseModel.compute_layer          ★core
│   │   ├── llama.py          ← LMCLlamaModel
│   │   └── utils.py          ← infer_model_from_vllm, VLLMModelTracker
│   ├── attention/
│   │   ├── flash_attn.py     ← LMCFlashAttnBackend
│   │   ├── abstract.py       ← AttentionInterface
│   │   └── metadata.py       ← LMCAttnMetadata, LMCFlashAttnMetadata
│   └── positional_encoding.py← FusedRope, BasicReverseRope, get_fused_rope
├── token_database.py         ← SegmentTokenDatabase                ★core
├── cache_engine.py           ← retrieve_layer
└── gpu_connector/gpu_connectors.py ← VLLMBufferLayerwiseGPUConnector
```

When this document is ambiguous, **read LMCache source**.

---

## 1. Request flow

Input prompt layout:

```
<sys_prompt> <SEP> <chunk_1> <SEP> <chunk_2> <SEP> ... <SEP> <chunk_N> <SEP> <question>
```

`<SEP>` is the separator string from `LMCACHE_BLEND_SPECIAL_STR`
(default `" # # "`).

What LMCache does for each request:

1. Tokenize the prompt.
2. `SegmentTokenDatabase.process_tokens` splits the token sequence at every
   occurrence of the separator's token IDs. Yields tuples
   `(start_idx, end_idx, hash_key)` for each chunk (excluding separators).
3. For each chunk, the storage backend is queried by hash key.
   - All chunks miss → request falls back to full prefill; the resulting
     per-chunk KVs are stored under their hash keys for future use.
   - Some/all hit → continue to (4).
4. `LMCBlender.blend(tokens, mask)` drives a `blend_layer` generator that
   advances `retrieve_layer` (KV loading) and `compute_layer` (layerwise
   prefill) in lockstep, one layer at a time.
5. For each layer i:
   - `retrieve_layer` loads layer i's K and V slices for the matched chunks
     into a GPU buffer covering `[starts[0], ends[-1]]`. Cached K is
     **rotated** from its original (cached) positions to the new positions
     in *this* request via `FusedRope.fused_encode(old_pos, new_pos, K)`.
     Gap positions (separator tokens, the question, anything without cached
     KV) are zeroed.
   - `compute_layer` runs one transformer layer. Between QKV projection and
     attention, `LMCBlender.process_qkv` is called.
6. The K, V buffers after the last layer are the **fused KV cache**. Decode
   starts from there.

Hidden states output by CacheBlend's prefill are not used. Only the KV cache
matters; decode re-prefills the last token of the question against this KV.

---

## 2. Components

### 2.1 `LMCBlendCommonMetadata` — `blend/metadata.py`

```python
@dataclass
class LMCBlendCommonMetadata:
    check_layers: List[int]                       # default [1]
    recomp_ratios: Optional[List[float]] = None   # default [0.15]
    thresholds: Optional[List[float]] = None
```

### 2.2 `LMCBlendMetadata` — `blend/metadata.py`

```python
@dataclass
class LMCBlendMetadata:
    imp_indices: Optional[torch.Tensor] = None   # HKVD indices, sorted asc
    attn_mask: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None     # token positions in request

    def clean(self):
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None
```

Lives on the blender. Reset at the end of each request via `clean()`.

### 2.3 `LMCBlender` — `blend/blender.py`

Fields set in `__init__`:
- `cache_engine`
- `gpu_connector`
- `layerwise_model` (LMCBaseModel subclass instance, built by
  `infer_model_from_vllm`)
- `num_layers` = `len(vllm_model.model.layers)`
- `common_metadata = LMCBlendCommonMetadata(check_layers, recomp_ratios, thresholds)`
- `metadata = LMCBlendMetadata(None, None, None)`

#### `process_qkv` — port verbatim

Exact body from `lmcache/v1/compute/blend/blender.py:59-120`:

```python
def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
    old_k, old_v = self.gpu_connector.get_kv(layer_id)

    if attn_output is None:
        attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    # perform positional encoding
    if self.metadata.positions is None:
        self.metadata.positions = torch.arange(
            q.shape[0], device=q.device, dtype=torch.int64
        )
    layer = self.layerwise_model.vllm_model.model.layers[layer_id]
    attn_layer = layer.self_attn
    q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

    if layer_id in self.common_metadata.check_layers:
        diff_k = torch.sum(
            (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
        )
        total_len = diff_k.shape[0]
        assert self.common_metadata.recomp_ratios is not None
        topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
        topk_num = max(topk_num, 1)

        top_indices = torch.topk(diff_k, k=topk_num).indices
        top_indices, _ = torch.sort(top_indices)

        k, v = k[top_indices], v[top_indices]
        q = q[top_indices]
        residual = residual[top_indices]

        self.metadata.imp_indices = top_indices
        self.metadata.positions = self.metadata.positions[top_indices]
        attn_output = attn_output[:topk_num]

        attn_metadata.update_from_top_indices(top_indices)

    if self.metadata.imp_indices is not None:
        old_k[self.metadata.imp_indices] = k
        old_v[self.metadata.imp_indices] = v
        return q, old_k, old_v, residual, attn_output, attn_metadata
    else:
        return q, k, v, residual, attn_output, attn_metadata
```

Key invariants:
- `diff_k` is computed in **float32**, on **K only**. K at this point has
  shape `(seq_len, num_kv_heads * head_dim)` (still 2-D, pre-reshape),
  so `dim=[1]` collapses the hidden axis to give per-token deviations.
- `topk_num = max(int(total_len * recomp_ratios[0]), 1)`.
- `top_indices` is sorted ascending (via the explicit `torch.sort` on
  `topk(...).indices`).
- `attn_output = attn_output[:topk_num]` happens **after** writing
  `imp_indices` and `positions` (matches source order; functionally
  equivalent to other orderings, but the port mandate is verbatim).
- After the check layer, q / residual / attn_output have HKVD-restricted
  length; K and V keep full length with HKVD rows overwritten.
- `imp_indices`, once set, persists for the rest of the layers in this
  request (no per-layer re-selection, no gradual narrowing). The
  CacheBlend paper §4.2 describes a "gradual filtering" variant with
  shrinking `r_i`, but **LMCache does not implement that** — only the
  single-check-layer form. The port mirrors the source, not the paper.
- The cached `old_k`, `old_v` are mutated in place — assignments via
  `old_k[imp_indices] = k` overwrite the GPU buffer that the connector
  returns. Subsequent layers' attention sees this updated buffer.
- `rotary_emb(positions, q, k)` is vLLM's RoPE API; in the HF port this
  is a wrapper attached to each `LlamaAttention` that calls HF's
  `apply_rotary_pos_emb(q, k, cos, sin)` and returns `(q_rot, k_rot)`.
  HF's true RoPE module lives on `model.model.rotary_emb`, not on the
  attention layer (see `CODING_CONVENTIONS.md`).

#### `blend_layer`, `blend` — port verbatim

```python
def blend_layer(self, tokens, mask=None, **kwargs):
    layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
    layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

    next(layerwise_retriever)
    yield

    for i in range(self.num_layers):
        next(layerwise_retriever)
        next(layerwise_model_executor)
        yield

    next(layerwise_retriever)
    self.metadata.clean()
    yield

def blend(self, tokens, mask=None, **kwargs):
    if isinstance(tokens, list):
        tokens = torch.tensor(tokens).cuda()
    g = self.blend_layer(tokens, mask, **kwargs)
    for _ in range(self.num_layers + 2):
        next(g)
```

### 2.4 `LMCBaseModel` and `LMCLlamaModel` — `compute/models/`

Source: `lmcache/v1/compute/models/base.py`.

`compute_layer(input_ids)` is the **layerwise prefill generator**. One
transformer layer per yield. It runs:

1. `input_ids = input_ids.cuda()`
2. `hidden_states = self.vllm_model.model.embed_input_ids(input_ids)`
3. `residual = None`, `attn_output = None`
4. `attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(input_ids=input_ids)`
5. Loop over layers (LMCache uses
   `self.vllm_model.model.layers[start_layer:end_layer]` for vLLM
   pipeline-parallel slicing; in the HF port use the full range
   `self.vllm_model.model.layers` since we don't support PP). For each
   `(idx, layer)`:
   a. Fused input layernorm with residual (LMCache call sites):
      ```python
      if residual is None:
          residual = hidden_states
          hidden_states = layer.input_layernorm(hidden_states)
      else:
          hidden_states, residual = layer.input_layernorm(hidden_states, residual)
      ```
   b. Fused QKV projection then split by `[q_size, kv_size, kv_size]`:
      ```python
      qkv, _ = layer.self_attn.qkv_proj(hidden_states)
      q, k, v = qkv.split(
          [layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size],
          dim=-1,
      )
      ```
   c. `q, k, v = self._process_qkv(q, k, v, layer)` — model-specific hook.
      For Llama this is a no-op (`LMCLlamaModel._process_qkv` returns
      `(q, k, v)` unchanged). At this point Q/K/V are still 2-D with
      shapes `(seq_len, num_heads*head_dim)` and
      `(seq_len, num_kv_heads*head_dim)`.
   d. `q, k, v, residual, attn_output, attn_metadata =
       self.blender.process_qkv(q, k, v, residual, idx, attn_output, attn_metadata)`
   e. Reshape to head layout:
      ```python
      num_heads    = self.vllm_attn_layers[idx].num_heads
      num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
      head_size    = self.vllm_attn_layers[idx].head_size

      q           = q.view(-1, num_heads,    head_size)
      k           = k.view(-1, num_kv_heads, head_size)
      v           = v.view(-1, num_kv_heads, head_size)
      attn_output = attn_output.view(-1, num_heads, head_size)
      ```
   f. Attention:
      `attn_output = self.lmc_attn_layers[idx].forward_contiguous(q, k, v, attn_output, attn_metadata)`.
   g. Reshape back to 2-D and project:
      ```python
      attn_output = attn_output.view(-1, num_heads    * head_size)
      k           = k.view(-1, num_kv_heads * head_size)
      v           = v.view(-1, num_kv_heads * head_size)
      hidden_states, _ = layer.self_attn.o_proj(attn_output)
      ```
   h. Fused post-attention layernorm with residual:
      `hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)`.
   i. `hidden_states = layer.mlp(hidden_states)`.
   j. `yield` (yields exactly `num_layers` times across the loop).

`LMCLlamaModel._process_qkv` is the identity for Llama (no q_norm/k_norm).

The reference `compute_layer` carries `@torch.compile`
(`lmcache/v1/compute/models/base.py:66`). The HF port **drops** this
decorator deliberately — see `CODING_CONVENTIONS.md` §"Code style".

`infer_model_from_vllm(vllm_model, blender, enable_sparse)`
(`lmcache/v1/compute/models/utils.py:14`):
- `LlamaForCausalLM`, `Qwen2ForCausalLM`, `MistralForCausalLM` → `LMCLlamaModel`
- `Qwen3ForCausalLM` → `LMCQwen3Model` (out of scope here)

`VLLMModelTracker.register_model(instance_id, vllm_model)` /
`.get_model(instance_id)` is a global dict keyed by `ENGINE_NAME`
(`lmcache/integration/vllm/utils.py:27` → `ENGINE_NAME = "vllm-instance"`).

The HF port introduces a parallel `infer_attn_backend_from_hf` that
mirrors LMCache's `infer_attn_backend_from_vllm`
(`lmcache/v1/compute/attention/utils.py:9`); it always returns
`LMCEagerAttnBackend`.

### 2.5 Attention — `compute/attention/`

Source: `lmcache/v1/compute/attention/{abstract,flash_attn,metadata}.py`.

`LMCAttnMetadata` is abstract (`@dataclass` with one abstract method
`update_from_top_indices`). `LMCFlashAttnMetadata` carries fields
`query_start_loc`, `seq_lens`, `cu_seqlens_k`, `max_query_len`,
`max_seq_len` (all declared as `torch.Tensor` in the dataclass, though
the constructor in `flash_attn.py:118-129` stores ints for
`max_query_len` and `max_seq_len`).

`update_from_top_indices(top_indices)` (`metadata.py:31-36`) only changes
`max_query_len` to `top_k_num` and `query_start_loc` to
`[0, top_k_num]`. `cu_seqlens_k` and `max_seq_len` are NOT touched —
they stay at the full request length, which is what `flash_attn_varlen`
needs for variable-Q-against-full-K attention on the check layer.

`forward_contiguous(q, k, v, output, attn_metadata, **kwargs)` computes
attention and writes the result into `output` (pre-allocated by
`process_qkv`). It returns `output`. In LMCache's flash backend this
calls `flash_attn_varlen_func(...)` with `causal=True`.

For the HF port we replace the flash backend with `LMCEagerAttnBackend`
and add `LMCEagerAttnMetadata` (mirrors LMCFlashAttnMetadata's fields
but adds `q_positions` / `k_positions` for explicit position-based mask
construction — see `CODING_CONVENTIONS.md` §"HF-required adaptations"
item 7).

### 2.6 RoPE — `compute/positional_encoding.py`

Source: `lmcache/v1/compute/positional_encoding.py`.

`FusedRope(rope, is_neox_style).fused_encode(old_positions, new_positions, k)`
(lines 55-82) rotates each row of K from `old_positions[i]` to
`new_positions[i]`. In LMCache the rotation is performed by the CUDA
kernel `lmc_ops.rotary_embedding_k_fused`, which mutates `k` in place
and **also returns it**. Inside `fused_encode`, K is reshaped from
`(num_tokens, num_kv_heads*head_size)` → `(num_tokens, num_kv_heads, head_size)`
for the kernel call and then reshaped back to 2-D before returning.
Used by the GPU connector (§2.8 step 2) to align cached K to the new
positions in this request.

`BasicReverseRope` (lines 22-52) exposes `__call__(positions, q, k) -> (q, k)`
that reverses a previously applied RoPE by shuffling Q/K halves, calling
`rope(positions, ...)`, and shuffling back. Used only for validation
inside `validate_reverse_correctness` (lines 112-144); not on any hot
path. The HF port can omit it (or include for parity).

`get_fused_rope(head_size, rotary_dim, max_position, base, is_neox_style,
rope_scaling, dtype, partial_rotary_factor)` (lines 148-202) builds and
validates a `FusedRope`. LMCache's `validate_rope_params` (lines 85-109)
rejects `rotary_dim != head_size`, any non-`None` `rope_scaling`, or
`partial_rotary_factor != 1.0`. **This means LMCache itself disables
CacheBlend for Llama-3.1**, which uses `rope_scaling = {"rope_type":
"llama3", ...}`.

For the HF port we sidestep this by building `FusedRope` from HF's
`LlamaRotaryEmbedding` (or directly from `inv_freq` derived per the
model's `rope_parameters`), so models with HF-supported rope scalings
work transparently. The port may still expose `validate_rope_params` as
a thin no-op so the API parallels LMCache, but its single
`rotary_dim == head_size` check is the only one that should reject in
the HF code path. See `PHASE3_PROMPT.md` for the build recipe.

### 2.7 `SegmentTokenDatabase` — `token_database.py`

Source: `lmcache/v1/token_database.py:423-551`. Builds chunk keys by
separator. Init (`SegmentTokenDatabase.__init__`):

```python
self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]
self.sep_tokens = torch.tensor(self.sep_tokens, device="cpu")
self.sep_len = len(self.sep_tokens)
```

Note: `[1:]` strips the BOS that some tokenizers add for the separator
string. (LMCache acknowledges this is a heuristic — see the TODO at
line 434.)

`_fast_split_by_subtensor(tokens)` (lines 441-462) slides an `unfold`
window of size `sep_len`, finds all positions where the window matches
`sep_tokens`, and yields the slices **between** matches:
```python
start = 0
for idx in matches:
    yield tokens[start:idx]      # chunk before this separator
    start = idx + self.sep_len    # skip past the separator tokens
yield tokens[start:]              # final chunk
```
So separator tokens are excluded from every yielded chunk.

`process_tokens(tokens, ...)` (lines 464-551) walks those chunks. State
between iterations is held in `start_idx`. Trace:
- Initial: `start_idx = 0`.
- Iteration `idx`: `chunk_len = len(token_chunk)`; `end_idx = start_idx + chunk_len`.
- If `idx > 0`: `start_idx += sep_len; end_idx += sep_len` (skips the
  separator that precedes this chunk in the full token sequence).
- Yield `(start_idx, end_idx, key)` where
  `key = _make_key_by_hash(_hash_tokens(chunk), request_configs)`.
- `start_idx = end_idx` for the next iteration.

So `(start_idx, end_idx)` are positions in the **full input** (and
separator tokens occupy `[end_idx_i, start_idx_{i+1}) = [end_idx_i, end_idx_i + sep_len)`).

Hash function: `_hash_tokens(tokens, prefix_hash=None, extra_keys=None)`
(lines 242-266) canonicalizes inputs via `_canonicalize_hash_inputs` to a
tuple `(prefix_hash_or_NONE_HASH, tokens_tuple, extra_keys_tuple)` and
applies `self.hash_func`. SegmentTokenDatabase calls `_hash_tokens` with
`prefix_hash=None` for every chunk, so **each chunk hashes
independently** (no prefix chaining; `prefix_hash` defaults to
`NONE_HASH = 0`).

Default `pre_caching_hash_algorithm` is `"builtin"`
(`lmcache/v1/config.py:100-104`), which uses Python's built-in `hash`.
Cross-process consistency requires `PYTHONHASHSEED` to be set; for a
single-process port this is not a concern.

### 2.8 `retrieve_layer` (cache_engine) and the GPU connector

Sources: `lmcache/v1/cache_engine.py:902-1055` (the `retrieve_layer`
method on `LMCacheEngine`) and
`lmcache/v1/gpu_connector/gpu_connectors.py:615-907` (the
`VLLMBufferLayerwiseGPUConnector` class).

`cache_engine.retrieve_layer(tokens, mask, **kwargs)` is a generator
yielding `num_layers + 2` times. Concrete control flow:

1. Walk `self.token_database.process_tokens(tokens=...)` collecting
   `(start, end, key)` for every chunk whose **layer-0** KV is present in
   storage (`storage_manager.contains(keys_multi_layer[0], ...)`).
   Loading stops at the first miss (`break`).
2. Start a `gpu_connector.batched_to_gpu(starts, ends, **kwargs)`
   generator and prime it with one `next(...)` (advances it to its first
   yield, which is a receive point for layer-0 mem-objs).
3. Loop `layer_id in range(num_layers)`:
   - Schedule async load of layer `layer_id` from storage.
   - **Yield**: at `layer_id == 0` yield `torch.sum(ret_mask)` (the
     count of retrieved tokens, surfaced for SGLang integration); at
     other iterations yield `None`. Caller (the blender) advances here.
   - Await the storage future, then `mem_obj_consumer.send(mem_objs_layer)`
     which loads layer `layer_id` into the GPU buffer.
4. After the loop, `yield None`, then `next(mem_obj_consumer)` to flush
   the final RoPE step, then `yield ret_mask` (the boolean mask of which
   token positions were filled). Total yields: `num_layers + 2`.

Caller invariant (used by `LMCBlender.blend_layer`): when the loop body
runs for iteration `i` after `next(layerwise_retriever)`, layer `i`'s K
(rotated to the new positions) and V are present in the GPU buffer and
visible via `gpu_connector.get_kv(i)`.

`VLLMBufferLayerwiseGPUConnector.batched_to_gpu(starts, ends, **kwargs)`
(lines 752-907) is also a generator. Per its docstring it pipelines:
"(1) loads layer i from CPU → GPU buffer, (2) recovers the positional
encoding of layer i-1's K in the GPU buffer, (3) moves layer i-2 from
GPU buffer to paged GPU memory." For our HF port we only need (1) and
(2); (3) is vLLM-paged-memory specific and is skipped.

Inside `batched_to_gpu`, two GPU buffers are allocated (compute + load)
and **ping-ponged** every iteration. Per-iteration logic (`for layer_id
in range(num_layers + 2)`):
- `layer_id > 1` → step (3): writeback layer `layer_id - 2` to paged
  memory and `del self.buffer_mapping[layer_id - 2]`. **Skip this in the
  HF port** (we never write back; the GPU buffer is the final store).
- `0 < layer_id <= num_layers` → step (2):
  1. `torch.cuda.synchronize()` both streams.
  2. Swap compute/load buffers (so `compute_gpu_buffer_obj` now holds
     the data just loaded for layer `layer_id - 1`).
  3. RoPE on K:
     `compute_gpu_buffer_obj.tensor[0] = fused_rotary_emb(old_positions_full, new_positions_full, compute_gpu_buffer_obj.tensor[0])`
  4. **After RoPE**, zero gap positions on **both K and V**:
     `compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0`
  5. Store: `self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj`.
- `layer_id < num_layers` → step (1): `memory_objs_layer = yield`; for
  each chunk copy `mem_obj.tensor[0/1]` into `load_gpu_buffer_obj` at
  `[start - buf_offset : end - buf_offset]`. **Only at `layer_id == 0`**
  populate `old_positions_full[start - buf_offset : end - buf_offset]`
  from `mem_obj.metadata.cached_positions` (the positions at which this
  chunk's K was originally cached).
- `layer_id == num_layers` → `yield` (final flush, no receive).
- After loop: `yield` once more (the `+2` accounting).

`new_positions_full = torch.arange(starts[0], ends[-1], ...)` —
contiguous range from the first chunk's start to the last chunk's end.
`current_gap_positions = where(gap_mask)[0]` where `gap_mask` is True
for token positions in `[starts[0], ends[-1])` that no chunk covers
(separator tokens, the question suffix, etc.).

`get_kv(layer_id)` (lines 729-739) returns `(K, V)` views into
`buffer_mapping[layer_id].tensor[0], .tensor[1]`. Crucially, these are
the **same tensors** that `process_qkv` mutates via
`old_k[imp_indices] = k`. Any HF-port replacement must preserve this
aliasing.

The harness's port (`HFBufferLayerwiseGPUConnector`) keeps a
per-request, GPU-resident `buffer_mapping: dict[layer_id, (K, V)]` and
drops the ping-pong/writeback machinery — `batched_to_gpu` simply loads
each layer's chunks into its buffer slot, applies `FusedRope` to K, and
zeros gap rows. Same yield count and same `get_kv` semantics.

### 2.9 Configuration knobs

From `LMCACHE_*` env vars (the example
`examples/blend_kv_v1/blend.py:20-58` sets these explicitly):

| Env var | Field name in `LMCacheEngineConfig` | Default | Example |
|---|---|---|---|
| `LMCACHE_ENABLE_BLENDING` | `enable_blending` | `False` | `True` |
| `LMCACHE_BLEND_SPECIAL_STR` | `blend_special_str` | `" # # "` | `" # # "` |
| `LMCACHE_BLEND_CHECK_LAYERS` | `blend_check_layers` | `None` | `[1]` (env: `"1"`) |
| `LMCACHE_BLEND_RECOMPUTE_RATIOS` | `blend_recompute_ratios` | `None` | `[0.15]` (env: `"0.15"`) |
| `LMCACHE_BLEND_THRESHOLDS` | `blend_thresholds` | `None` | (unused) |
| `LMCACHE_BLEND_MIN_TOKENS` | `blend_min_tokens` | `256` | — |
| `LMCACHE_USE_LAYERWISE` | `use_layerwise` | `False` | `True` (required for blending) |
| `LMCACHE_CHUNK_SIZE` | `chunk_size` | `256` | (only used by `ChunkedTokenDatabase`) |
| `LMCACHE_PRE_CACHING_HASH_ALGORITHM` | `pre_caching_hash_algorithm` | `"builtin"` | — |
| `LMCACHE_EXTRA_CONFIG` | `extra_config` (dict) | `None` | `{"enable_sparse": True}` |

Field names and defaults come from `lmcache/v1/config.py:62-127`. The
port mirrors these via a `LMCacheEngineConfig`-equivalent dataclass with
a `from_env()` classmethod.

---

## 3. Behavioral guarantees the port must preserve

1. Given the same input tokens and same cached chunk KVs, the **fused KV
   cache** produced by the port must be elementwise equal (within float32
   tolerance) to the one LMCache would produce, modulo attention backend
   differences (LMCache=flash_attn, port=eager) which only show up in the
   attention output for HKVD rows.

2. The same set of HKVD `top_indices` must be chosen given the same K_new
   and K_cached values on the check layer.

3. Per-chunk hash keys produced by `SegmentTokenDatabase` are stable across
   runs (the hash function is deterministic given the same Python hash
   seed).

4. After `blend()`, `metadata.imp_indices`, `metadata.positions`,
   `metadata.attn_mask` are all `None` (request state reset).

5. RoPE rotation of cached K from old to new positions is exact for
   `rotary_dim == head_size`, neox-style, no scaling, fp32-compared
   tolerance.

---

## Review notes

Audit pass against branch `fix/cacheblend-vllm-v0.17.1-compat` of
`chjs/LMCache`. Key revisions:

- §2.3 `process_qkv`: reordered `attn_output = attn_output[:topk_num]`
  to **after** the `imp_indices`/`positions` writes, matching
  `blender.py:103-113`. Added the missing `assert recomp_ratios is not
  None` (line 94). Added a paragraph noting LMCache does **not**
  implement the paper's "gradual filtering" — `imp_indices` is set once
  on the check layer and reused. Added an HF-port note clarifying that
  `attn_layer.rotary_emb(positions, q, k)` is vLLM-style and that HF's
  true rotary lives on `model.model.rotary_emb`.

- §2.4 `compute_layer`: replaced the abbreviated step list with the
  exact code blocks from `base.py:67-142`, including the post-attention
  reshape of K and V back to 2-D (previously missing), the `cuda()`
  call on `input_ids`, and the layer slice `[start_layer:end_layer]`
  (which the HF port replaces with the full range). Flagged that
  LMCache's `compute_layer` carries `@torch.compile`, which the HF
  port intentionally drops.

- §2.5 Attention: clarified that `LMCAttnMetadata` is an abstract
  dataclass with one abstract method, and that
  `update_from_top_indices` leaves `cu_seqlens_k` / `max_seq_len`
  untouched. Documented the introduction of `q_positions` /
  `k_positions` in the HF eager metadata.

- §2.6 RoPE: documented that `FusedRope.fused_encode` mutates K in
  place via the CUDA kernel and reshapes around the call. Flagged the
  important consequence of `validate_rope_params`: **LMCache disables
  CacheBlend for Llama-3.1** (which has `rope_scaling != None`), and
  the HF port intentionally widens this by building `FusedRope` from
  HF's own rotary so Llama-3.1 / Llama-3.2 / Llama-3.3 work.

- §2.7 SegmentTokenDatabase: walked through the exact `start_idx`
  advancement (separators absorbed via `+= sep_len` only for `idx > 0`)
  with line citations, replaced the imprecise "advances by sep_len
  between chunks" wording. Clarified that hash function defaults to
  `"builtin"` (Python's built-in `hash`), which requires
  `PYTHONHASHSEED` for cross-process consistency (irrelevant for a
  single-process HF port).

- §2.8 retrieve_layer + connector: replaced the high-level summary
  with a step-by-step trace of the actual generator coroutine
  (`cache_engine.py:902-1055` and `gpu_connectors.py:752-907`),
  including: yield count derivation, the ping-pong buffer pattern, the
  fact that gap zeroing runs **after** RoPE on **both K and V** (line
  866), and that `cached_positions` is read only at `layer_id == 0`.

- §2.9 Configuration: added the env-var → field-name → default mapping
  with line citations to `config.py:62-127`. Added `LMCACHE_EXTRA_CONFIG`
  (used by sparse mode in the example).
