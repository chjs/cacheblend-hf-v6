# Phase 1 — Layerwise forward in HF

## Context

You are porting LMCache's CacheBlend implementation to HuggingFace
Transformers. The full project context is in `README.md`,
`docs/LMCACHE_IMPLEMENTATION.md`, `docs/PROJECT_PLAN.md`, `docs/CODING_CONVENTIONS.md`,
and `docs/VERIFICATION_PROTOCOL.md`. Read all five before you start coding.

The prime directive (from `docs/CODING_CONVENTIONS.md`):

> **Follow LMCache. Do not redesign.** Class names, function names,
> signatures, and field names must match LMCache. Adapt only when an HF API
> leaves no choice, and even then keep the call sites identical.

## Working style

- Make decisions on small ambiguities yourself; do not stop to ask the user.
  When in doubt, pick the choice closer to LMCache and continue. The user
  will review after the phase and roll back if needed.
- Read LMCache source before writing each file.
- Run the verification commands listed at the end of this prompt yourself
  before declaring this phase done.

## Setup

If not already done:

```bash
# Reference checkout
git clone --branch fix/cacheblend-vllm-v0.17.1-compat \
    https://github.com/chjs/LMCache.git /workspace/LMCache_reference

# Project workspace
mkdir -p /workspace/cacheblend_hf
cd /workspace/cacheblend_hf
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.2" "transformers>=4.44" accelerate pytest numpy
```

Create `pyproject.toml` (or `setup.cfg`) so that `pip install -e .` works
and `lmc` is importable.

## Task

Implement the layerwise prefill skeleton without any CacheBlend logic.
After this phase, `LMCBaseModel.compute_layer` should run a HF Llama / Mistral
model one layer at a time, yielding after each layer, with a stub blender
that just passes (q, k, v) through.

Read these LMCache files first:
- `lmcache/v1/compute/models/base.py`
- `lmcache/v1/compute/models/llama.py`
- `lmcache/v1/compute/models/utils.py`
- `lmcache/v1/compute/attention/abstract.py`
- `lmcache/v1/compute/attention/flash_attn.py` (for the interface shape; do
  NOT port flash_attn — implement eager equivalent)
- `lmcache/v1/compute/attention/metadata.py`

Create:

### `lmc/compute/attention/abstract.py`

Port `AttentionInterface` (the abstract base in
`lmcache/v1/compute/attention/abstract.py`). Same method names and shapes:
`forward_contiguous(query, key, value, output, attn_metadata, **kwargs)`,
`init_attn_metadata(input_ids, **kwargs)`.

### `lmc/compute/attention/metadata.py`

Port `LMCAttnMetadata` (abstract dataclass). Add `LMCEagerAttnMetadata`
modeled on `LMCFlashAttnMetadata`:

```python
@dataclass
class LMCEagerAttnMetadata(LMCAttnMetadata):
    query_start_loc: torch.Tensor   # [0, n_q]
    seq_lens: torch.Tensor          # [n_k]
    cu_seqlens_k: torch.Tensor      # [0, n_k]
    max_query_len: int
    max_seq_len: int
    # For eager backend, also track the per-token positions used to build
    # a causal mask correctly when query is HKVD-restricted.
    q_positions: Optional[torch.Tensor] = None  # full positions of query rows
    k_positions: Optional[torch.Tensor] = None  # full positions of key rows
```

`update_from_top_indices(top_indices)` mirrors `LMCFlashAttnMetadata`'s:
sets `query_start_loc = [0, len(top_indices)]`, `max_query_len = len(top_indices)`,
and additionally records `q_positions = top_indices` so the eager backend
can build the correct mask.

### `lmc/compute/attention/eager.py`

`LMCEagerAttnBackend(AttentionInterface)`. Same interface as
`LMCFlashAttnBackend`. `forward_contiguous`:

- Inputs: q `(n_q, num_heads, head_dim)`, k `(n_k, num_kv_heads, head_dim)`,
  v same shape as k, `output` pre-allocated `(n_q, num_heads, head_dim)`.
- Expand k, v across GQA groups to `(n_k, num_heads, head_dim)` if
  `num_kv_heads < num_heads`.
- Compute `scores = q @ k.T * scale`, shape `(num_heads, n_q, n_k)`.
- Build causal mask: for each query row `i` at position
  `attn_metadata.q_positions[i]` (or `i` if not set), allow attention to
  key rows whose position is `≤ q_positions[i]`. Mask others to `-inf`.
  When `q_positions is None` and `n_q == n_k`, fall back to a standard
  causal triangular mask.
- Softmax in fp32 then cast back; multiply by V; write into `output`.
- Return `output`.

`init_attn_metadata(input_ids)` builds the metadata for full-length prefill
(seq_len = len(input_ids)), `q_positions=None`, `k_positions=None`.

### `lmc/compute/models/base.py`

`LMCBaseModel(nn.Module, ABC)`. Port `__init__` from
`lmcache/v1/compute/models/base.py` with these adaptations:

- The constructor signature is `(self, hf_model, blender, enable_sparse=False)`
  (rename `vllm_model` → `hf_model`, but **expose it as
  `self.vllm_model = hf_model`** so that `process_qkv`'s line
  `self.layerwise_model.vllm_model.model.layers[layer_id]` works unchanged.
  This is the one rename. Document it in a comment.)
- `self.num_layers = len(hf_model.model.layers)`
- After construction, set `blender.layerwise_model = self` (the stub
  blender's `__init__` left this as `None`).
- For each layer, build `LMCEagerAttnBackend` and append to
  `self.lmc_attn_layers`. Also store a small holder object exposing
  `num_heads`, `num_kv_heads`, `head_size` and append to
  `self.vllm_attn_layers` (kept under the original LMCache name so
  `compute_layer`'s `self.vllm_attn_layers[idx].num_heads` line is
  unchanged). Read values from HF config: `num_heads = config.num_attention_heads`,
  `num_kv_heads = config.num_key_value_heads`,
  `head_size = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads`.
- The cleanest factoring is to add a small `infer_attn_backend_from_hf`
  helper in `lmc/compute/attention/utils.py` (mirrors LMCache's
  `infer_attn_backend_from_vllm` at `attention/utils.py:9`) that always
  returns `LMCEagerAttnBackend(num_heads, num_kv_heads, head_size,
  scaling)`. Then `LMCBaseModel.__init__` reads
  `infer_attn_backend_from_hf(layer.self_attn, enable_sparse)` per layer.
- `self.fused_rotary_emb`: leave `None` for Phase 1. Phase 3 fills it
  in via `get_fused_rope(...)`.
- Attach `q_size`, `kv_size` to each layer's `self_attn` (read from HF
  config: `q_size = num_heads * head_dim`,
  `kv_size = num_kv_heads * head_dim`).
- Patch each layer's `self_attn` with:
  - `qkv_proj`: callable that runs `q_proj`, `k_proj`, `v_proj` on
    `hidden_states` and returns `(torch.cat([q, k, v], dim=-1), None)`.
    Use `dim=-1` (the projections act on flat 2-D tensors here).
  - `rotary_emb`: callable `(positions, q, k) -> (q_rot, k_rot)` built
    against **`hf_model.model.rotary_emb`** (HF's true rotary lives on
    the model, not the attention layer — see `docs/CODING_CONVENTIONS.md` item 3).
    Inside the callable, reshape Q, K from 2-D to
    `(1, num_heads_or_kv, seq_len, head_dim)`, call
    `hf_model.model.rotary_emb(q_4d, positions.unsqueeze(0))` to get
    `(cos, sin)`, then `apply_rotary_pos_emb(q_4d, k_4d, cos, sin)`,
    then reshape back to 2-D. **This must work for Phase 1** because
    the stub blender already invokes it.
  - Wrap `o_proj` so it returns `(out, None)` (HF's plain `nn.Linear`
    returns a single tensor).
- Patch each layer's `input_layernorm` / `post_attention_layernorm` so the
  signature matches vLLM's fused residual variant:
  ```python
  def __call__(self, hidden_states, residual=None):
      if residual is None:
          return rmsnorm(hidden_states)          # one return value
      out = rmsnorm(hidden_states + residual)
      return out, hidden_states + residual       # (x_norm, new_residual)
  ```
  This matches the LMCache call sites:
  ```python
  if residual is None:
      residual = hidden_states
      hidden_states = layer.input_layernorm(hidden_states)
  else:
      hidden_states, residual = layer.input_layernorm(hidden_states, residual)
  ```
- Bind `hf_model.model.embed_input_ids = hf_model.model.embed_tokens`
  (alias) so `compute_layer`'s
  `self.vllm_model.model.embed_input_ids(input_ids)` line is unchanged.

`compute_layer(input_ids)` — port the LMCache version
(`compute/models/base.py:67-142`) line-for-line, with two HF
adaptations:

1. **Drop `@torch.compile`.** See `docs/CODING_CONVENTIONS.md` §"Code style"
   for rationale.
2. **Replace `model.layers[start_layer:end_layer]` with
   `model.layers`.** vLLM's PP slice attributes don't exist on HF; we
   use the full layer list.

Everything else — embed call, `residual = None`, `attn_output = None`,
`init_attn_metadata`, the residual-layernorm calls, fused
`qkv_proj`/split, `_process_qkv`, `blender.process_qkv`, the 2-D→3-D
view, `forward_contiguous`, the 3-D→2-D view back, `o_proj`,
`post_attention_layernorm`, `mlp`, `yield` — should be **byte-identical**
to the LMCache source. Refer to `docs/LMCACHE_IMPLEMENTATION.md` §2.4 for the
exact code blocks.

Abstract method `_process_qkv(self, q, k, v, layer)` — same as LMCache.

### `lmc/compute/models/llama.py`

```python
class LMCLlamaModel(LMCBaseModel):
    def _process_qkv(self, q, k, v, layer):
        return q, k, v
```

### `lmc/compute/models/utils.py`

Port `infer_model_from_vllm` as `infer_model_from_hf(hf_model, blender,
enable_sparse=False)`. For Phase 1, support `LlamaForCausalLM` and
`MistralForCausalLM` (HF class names) → return `LMCLlamaModel`. Raise for
others.

Port `VLLMModelTracker` as `HFModelTracker` with the same methods
(`register_model`, `get_model`) under `lmc/integration/hf/utils.py`.

### Stub blender

In `lmc/compute/blend/blender.py`, create a minimal `LMCBlender` whose
`process_qkv` body **structurally mirrors the real LMCache one minus
the HKVD branch and the `old_k`/`old_v` write**. The reason for
including RoPE here in Phase 1 (rather than deferring to Phase 2): the
real `process_qkv` always applies RoPE before the optional HKVD logic
(`blender.py:79-86`), so Phase 2's stock-vs-layerwise equivalence test
needs RoPE applied at the same call site. Putting RoPE in the stub now
means Phase 2 inherits it for free and `compute_layer`'s call site
stays byte-identical between the stub and the real blender in Phase 3.

The stub needs a tiny `LMCBlendMetadata` to hold `positions` across
layers within one request (so `torch.arange` is built once, not per
layer). Use a real dataclass; Phase 3 will replace it with the full
version.

```python
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class LMCBlendMetadata:
    positions: Optional[torch.Tensor] = None
    # imp_indices and attn_mask are filled in Phase 3; absent fields here
    # would force Phase 3 to rewrite the dataclass — easier to declare
    # them now as None.
    imp_indices: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None

    def clean(self):
        self.positions = None
        self.imp_indices = None
        self.attn_mask = None


class LMCBlender:
    """Phase 1 stub. Phase 3 replaces this with the full LMCBlender."""

    def __init__(self, hf_model, num_layers):
        self.layerwise_model = None    # set by LMCBaseModel.__init__
        self.num_layers = num_layers
        self.metadata = LMCBlendMetadata()
        self.common_metadata = None    # not used in Phase 1

    def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        return q, k, v, residual, attn_output, attn_metadata
```

`LMCBaseModel.__init__` sets `blender.layerwise_model = self` after
construction.

Reset semantics: the stub's `clean()` mirrors the real version but no
caller invokes it in Phase 1 (no `blend_layer`). Phase 2's test driver
must call `blender.metadata.clean()` between runs if it re-uses the
same blender on a different prompt.

## Tests

Write `tests/test_phase1_layerwise.py` covering the Phase 1 criteria in
`docs/VERIFICATION_PROTOCOL.md` §1.

## Verification

Run from the repo root:

```bash
pytest tests/test_phase1_layerwise.py -v
```

Phase 1 is done when all tests pass on both Mistral-7B and Llama-3.1-8B.
Do not proceed to Phase 2 without asking the user to review the diff
first.

## 작업 보고

작업이 끝나면 `reports/phase1.md` 를 작성하라. 양식은
`docs/CODING_CONVENTIONS.md` §"Phase reports (Korean)" 참고. 보고서
작성 후 stdout 에는 한 줄 안내만 출력하라.

---

## Review notes

- Stub blender now includes RoPE in `process_qkv` (previously punted to
  Phase 2). Reason: LMCache `process_qkv` always runs RoPE before the
  HKVD branch (`blender.py:79-86`), so Phase 2's stock-vs-layerwise
  equivalence test requires it at the same call site. Resolving the
  ambiguity here (Phase 1) instead of Phase 2 also means the
  `compute_layer` call site is byte-identical between the stub and the
  Phase 3 full blender.
- Stub blender now keeps a real `LMCBlendMetadata` dataclass with the
  same three fields as LMCache (`positions`, `imp_indices`, `attn_mask`)
  + a `clean()` method. The fields are declared now so Phase 3 doesn't
  rewrite the dataclass shape.
- Added explicit pointer to `infer_attn_backend_from_hf` as the parallel
  to LMCache's `infer_attn_backend_from_vllm` (`attention/utils.py:9`).
- Added explicit guidance on `rotary_emb` wrapper construction: read
  from `hf_model.model.rotary_emb` (HF's rotary lives on `LlamaModel`,
  not on the attention layer; see `modeling_llama.py:366`), reshape Q/K
  2-D ↔ 4-D around the call.
- Replaced "**Do not** use `@torch.compile`" with a pointer to the
  rationale in `docs/CODING_CONVENTIONS.md` so reviewers see the same reason
  in both places.
- Flagged the vLLM `start_layer:end_layer` slice in
  `compute/models/base.py:84-85` and instructed the port to use the
  full `model.layers` (HF has no PP slice).
