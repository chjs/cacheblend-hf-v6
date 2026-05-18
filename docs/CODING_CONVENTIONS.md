# Coding Conventions

## Prime directive

**Follow LMCache.** Class names, function names, method signatures,
parameter names, and even field names inside dataclasses must match the
LMCache source as closely as possible. The only acceptable deviations are
where an HF API leaves no choice (e.g., HF doesn't have `qkv_proj` fused;
HF's RoPE returns `(cos, sin)`). In those cases, write a thin wrapper that
restores the LMCache-shaped interface so that `compute_layer`,
`process_qkv`, etc. read identically to the reference.

When in doubt, read the file in `/workspace/LMCache_reference/` and copy
its structure. Do not invent your own organization.

## Working style for Claude Code

- **Move fast.** Do not ask the user clarification questions during a phase
  unless something is genuinely impossible to decide from the spec. If a
  small choice is ambiguous, pick the choice closer to LMCache, document it
  in a short comment, and continue. The user will review at the end of the
  phase and roll back if needed.
- **Read first.** Before writing a file, read the corresponding LMCache
  source file in full. Re-read it whenever there is doubt.
- **Verify before declaring done.** Run the phase's verification commands
  yourself (the prompts list them). Only report completion when those
  commands pass.
- **No bonus features.** Do not add features the spec doesn't list. Do not
  add CLI flags, logging frameworks, configuration loaders, or wrappers
  that the LMCache reference doesn't have. Match LMCache's surface area.
- **No silent simplifications.** If you skip something that is in LMCache,
  call it out explicitly in the final report.

## Phase reports (Korean)

At the end of each phase (Phase 1 onward), produce a file
`reports/phaseN.md` in **Korean** describing what was done. The report
must follow this template exactly:

```
# Phase N 작업 보고서

## 1. 수행한 작업
- 생성/수정한 파일 목록 (경로 포함)
- 각 파일의 핵심 변경 사항 요약

## 2. 검증 결과
- 통과한 테스트와 측정값 (tolerance, latency 등 구체 수치)
- 실패한 테스트가 있다면 원인 분석과 시도한 해결책
- 검증 명령어 출력 요약

## 3. LMCache와의 일치도
- LMCache 원본과 동일하게 유지한 부분 (파일/함수 단위)
- HF 어댑테이션 때문에 달라진 부분과 그 근거
  (LMCache 또는 HF 소스의 파일/라인 인용)

## 4. 작업 중 결정한 사항
- 모호한 부분에서 어떤 선택을 했는지
- 그 선택의 근거

## 5. 다음 Phase 준비도
- 진행 가능 여부
- 사용자 리뷰가 필요한 부분
- 알려진 잔여 리스크
```

The report is in Korean. The **code, comments, and harness `.md`
files stay in English**. Only the files under `reports/` are Korean.
Be specific and technical; this is a record for the user to review,
not a marketing document. Technical terms (e.g., `process_qkv`, `RoPE`,
`Cache`) can stay in English inside the Korean prose.

The final stdout summary at the end of each phase should be a short
pointer rather than the full report:

```
작업 완료. 자세한 내용은 reports/phaseN.md 참고.
```

The full report goes in the file; stdout stays minimal so the
conversation context isn't flooded.

Phase 0 and Phase 0.5 also follow this convention:
`reports/phase0_audit.md` and `reports/phase0_5_setup.md`.

## File layout

Project root is the repo root (where `pyproject.toml` lives). The
top-level layout is documented in the repo `README.md`; the package
layout under `lmc/` is:

```
lmc/                              # mirrors lmcache/v1/
│   ├── __init__.py
│   ├── config.py
│   ├── token_database.py
│   ├── cache_engine.py
│   ├── storage.py
│   ├── gpu_connector.py
│   ├── compute/
│   │   ├── __init__.py
│   │   ├── positional_encoding.py
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py       # AttentionInterface
│   │   │   ├── eager.py          # LMCEagerAttnBackend
│   │   │   └── metadata.py       # LMCAttnMetadata, LMCEagerAttnMetadata
│   │   ├── blend/
│   │   │   ├── __init__.py
│   │   │   ├── blender.py
│   │   │   ├── metadata.py
│   │   │   └── utils.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── llama.py
│   │       └── utils.py
│   └── integration/
│       └── hf/
│           ├── __init__.py
│           └── utils.py          # HFModelTracker, ENGINE_NAME
```

Sibling top-level directories at the repo root (already exist):
`tests/`, `scripts/`, `results/`, `docs/`, `reports/`.

## Naming rules

- Module path under `lmc/` mirrors `lmcache/v1/` exactly.
- Classes keep their LMCache name. Examples:
  - `LMCBlender`, `LMCBlendMetadata`, `LMCBlendCommonMetadata`,
    `LMCBlenderBuilder`
  - `LMCBaseModel`, `LMCLlamaModel`
  - `LMCAttnMetadata`, `AttentionInterface`
  - `FusedRope`, `BasicReverseRope`
  - `SegmentTokenDatabase`, `TokenDatabase`
  - `LMCacheEngine`, `LMCacheEngineConfig`
- Methods keep their LMCache names: `process_qkv`, `blend_layer`, `blend`,
  `compute_layer`, `_process_qkv`, `update_from_top_indices`,
  `init_attn_metadata`, `forward_contiguous`, `get_kv`, `retrieve_layer`,
  `batched_to_gpu`, `process_tokens`, `_fast_split_by_subtensor`,
  `_hash_tokens`, `_make_key_by_hash`, `fused_encode`, `reverse_encode`.
- Field names inside dataclasses: `imp_indices`, `attn_mask`, `positions`,
  `check_layers`, `recomp_ratios`, `thresholds`, `query_start_loc`,
  `seq_lens`, `cu_seqlens_k`, `max_query_len`, `max_seq_len`, etc.
- Module-level constants: `ENGINE_NAME` (string), `NONE_HASH = 0`.

## HF-required adaptations (allowed)

These are the only places you are allowed to diverge structurally from
LMCache. In each case, wrap the HF API so that the *call site* still reads
exactly like LMCache:

1. **Fused QKV projection.** vLLM has `layer.self_attn.qkv_proj`. HF has
   `q_proj`, `k_proj`, `v_proj` (see
   `transformers/models/llama/modeling_llama.py:238-249` in
   transformers 5.7.0). Add a small helper on `LMCBaseModel` that runs
   all three and concatenates, returning `(qkv, None)` so the
   `qkv, _ = layer.self_attn.qkv_proj(hidden_states)` line in
   `compute_layer` can stay character-for-character identical to
   LMCache. Implement this by monkey-patching a `qkv_proj` attribute
   onto each HF `LlamaAttention` instance at adapter setup time. Use
   `dim=-1` for the concatenation (the projections produce 2-D
   `(seq_len, q_or_kv_size)` outputs since `compute_layer` operates on
   un-batched / flattened tokens).

2. **Fused residual LayerNorm.** vLLM's `input_layernorm(x, residual)`
   returns `(x_norm, new_residual)` doing the residual add internally.
   HF's `LlamaRMSNorm` is unary (`modeling_llama.py:62-67`). Provide a
   wrapper around HF's RMSNorm so the call sites
   ```python
   if residual is None:
       residual = hidden_states
       hidden_states = layer.input_layernorm(hidden_states)
   else:
       hidden_states, residual = layer.input_layernorm(hidden_states, residual)
   ```
   are kept identical to LMCache. Same for `post_attention_layernorm`.

3. **Rotary embedding API.** vLLM's `rotary_emb(positions, q, k)` returns
   `(q_rot, k_rot)`. HF returns `(cos, sin)` from
   `LlamaRotaryEmbedding.forward(x, position_ids)` and applies them via
   `apply_rotary_pos_emb(q, k, cos, sin)` (`modeling_llama.py:124-168`).
   The true rotary module **lives on `LlamaModel` itself** (i.e.
   `hf_model.model.rotary_emb`), NOT on the attention layer.

   Wrap HF's RoPE behind a callable that exposes
   `rotary_emb(positions, q, k) -> (q_rot, k_rot)` operating on 2-D
   Q/K of shape `(seq_len, num_heads*head_dim)` /
   `(seq_len, num_kv_heads*head_dim)` (the shapes used inside
   `compute_layer` before the head-layout reshape):

   - Reshape Q, K to `(1, num_heads_or_kv, seq_len, head_dim)`.
   - Compute `cos, sin = model.model.rotary_emb(q_4d,
     positions.unsqueeze(0))` (returns `(1, seq_len, head_dim)`).
   - Call `apply_rotary_pos_emb(q_4d, k_4d, cos, sin, unsqueeze_dim=1)`.
   - Reshape back to 2-D.

   Attach this callable to each layer's attention module as
   `attn.rotary_emb` so `process_qkv`'s `attn_layer.rotary_emb(...)`
   call site is unchanged. RoPE for Llama / Mistral is neox-style;
   HF's `apply_rotary_pos_emb` already implements neox-style rotation.

4. **`embed_input_ids`.** vLLM's `model.model.embed_input_ids(ids)`. HF's
   equivalent is `model.model.embed_tokens(ids)`
   (`modeling_llama.py:361`). Bind one to the other so the
   `compute_layer` call site stays the same:
   `hf_model.model.embed_input_ids = hf_model.model.embed_tokens`.

5. **`o_proj` return.** vLLM's `o_proj(x)` returns `(out, None)`. HF's
   plain `nn.Linear` returns just `out` (`modeling_llama.py:247`).
   Wrap (or assign a tiny lambda module) so it returns a tuple.

6. **`q_size`, `kv_size`.** vLLM attaches these to `self_attn`. Compute
   them from HF config (`q_size = num_attention_heads * head_dim`,
   `kv_size = num_key_value_heads * head_dim`) at adapter setup and
   attach to each HF `self_attn`.

7. **Attention backend.** Replace `LMCFlashAttnBackend` with
   `LMCEagerAttnBackend`. The metadata class is renamed to
   `LMCEagerAttnMetadata` but keeps the same fields where they make
   sense. For eager attention with a possibly-restricted query (HKVD)
   over a full-length K, V, build a per-token causal mask: query row
   `i` (at position `imp_indices[i]` in the full sequence) attends to
   keys at positions `[0..imp_indices[i]]`. Before the check layer,
   when query and key have the same length, this reduces to a standard
   causal triangular mask. Implementation hint: in `forward_contiguous`,
   read `attn_metadata.q_positions` (set by `update_from_top_indices`)
   and `attn_metadata.k_positions` (defaults to
   `arange(seq_len, device=...)` when None); mask
   `scores[..., i, j] = -inf where k_pos[j] > q_pos[i]`.

8. **Embedding-dim aware reshapes.** vLLM has
   `vllm_attn_layers[idx].num_heads`, `.num_kv_heads`, `.head_size`.
   Read them from HF config (`num_attention_heads`, `num_key_value_heads`,
   `head_dim`) at adapter setup; attach a small holder object on
   `LMCBaseModel` as `self.vllm_attn_layers[i]` exposing those fields.

9. **`past_key_values` extraction (Phase 2 only).** Transformers ≥4.36
   (and definitely 5.x as installed) returns a `Cache` object
   (`DynamicCache` by default — see `transformers/cache_utils.py:1173`)
   rather than the legacy tuple-of-tuples. Extract per-layer
   `(K, V)` for comparison via `cache.layers[i].keys` /
   `cache.layers[i].values`, **not** `cache[i][0]`. Both have shape
   `(batch, num_kv_heads, seq_len, head_dim)`.

## Code style

- Python 3.10+. Use modern type hints (`list[int]`, `dict[str, X]`,
  `Optional[X]`).
- Dataclasses for metadata records.
- Generators (`yield`) for `compute_layer`, `blend_layer`, `retrieve_layer`,
  `batched_to_gpu`. Match LMCache's generator yield count exactly:
  `compute_layer` yields `num_layers` times (once per layer body);
  `blend_layer` yields `num_layers + 2` times; `retrieve_layer` yields
  `num_layers + 2` times; `batched_to_gpu` yields `num_layers + 2`
  times.
- **No `@torch.compile`.** LMCache's `compute_layer` carries
  `@torch.compile` at `models/base.py:66`; the HF port intentionally
  drops it. Reasons: (a) graph capture interacts badly with monkey-
  patched `qkv_proj` / `rotary_emb` adapters and with the dict-mutating
  `gpu_connector.get_kv` path; (b) the port's goal is correctness and
  readable side-by-side comparison, not throughput.
- No logging library; `print` is fine for the few diagnostic outputs the
  reference produces. Tests should not depend on stdout content.
- Don't add type-checked Protocols / ABCs beyond what LMCache already has
  (`AttentionInterface`, `LMCAttnMetadata`, `TokenDatabase`).

## Dependencies

`pyproject.toml` (or `requirements.txt`) lists:

- `torch>=2.2`
- `transformers>=4.44`  (developed against 5.7.0; see note below)
- `accelerate`
- `pytest`
- `numpy`
- `datasets` (Phase 4 only, for 2WikiMQA)

No vLLM, no flash_attn, no lmcache, no triton.

> **Transformers version note.** Item 9 in "HF-required adaptations" is
> calibrated to transformers 5.7.0 (the installed version on the dev
> box), where `past_key_values` is a `Cache` object with
> `.layers[i].keys/.values`. If the runtime environment has
> transformers 4.36–4.45, you may instead see a `DynamicCache` with
> `.key_cache[i]` / `.value_cache[i]`, or (very old) a tuple of
> `(K, V)` per layer. Phase 2 tests should branch on type:
> `hasattr(cache, "layers") → cache.layers[i].keys`, else
> `hasattr(cache, "key_cache") → cache.key_cache[i]`, else
> tuple-of-tuples.

## What "the same as LMCache" means concretely

If the reviewer opens `lmcache/v1/compute/blend/blender.py` and the
ported `lmc/compute/blend/blender.py` side by side, the diff should be:

- Identical class name, identical method names and signatures.
- Identical control flow inside `process_qkv`, `blend_layer`, `blend`.
- Almost identical variable names.
- The only diffs are import paths, the docstring author tag, and the
  HF-shape wrappers documented above.

Same goes for `metadata.py`, `models/base.py`, `models/llama.py`,
`positional_encoding.py`, `token_database.py`.

---

## Review notes

- HF adaptation items rewritten with file:line citations to
  `modeling_llama.py` (transformers 5.7.0). The biggest substantive
  change is item 3 (RoPE): the live rotary module is on `LlamaModel`,
  not on `LlamaAttention` — the adapter must read from
  `hf_model.model.rotary_emb` and synthesize a `(positions, q, k) ->
  (q_rot, k_rot)` callable per layer.
- Added a new item 9 covering `past_key_values` extraction: in
  transformers 5.7.0 it is a `Cache` object indexed via
  `cache.layers[i].keys` / `.values`. The Phase 2 prompt previously
  assumed the legacy tuple-of-tuples and would fail.
- Resolved the `@torch.compile` directive: LMCache uses it
  (`base.py:66`), the HF port drops it deliberately. Rationale spelled
  out so a reviewer doesn't try to "restore parity" by adding it.
- Added a transformers version compatibility note in Dependencies — the
  Cache object shape changed across 4.36 / 4.44 / 5.x; tests should
  branch on attribute presence.
