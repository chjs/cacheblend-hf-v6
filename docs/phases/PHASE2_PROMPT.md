# Phase 2 — Equivalence with stock HF forward

## Context

Read `README.md`, `docs/LMCACHE_IMPLEMENTATION.md`, `docs/CODING_CONVENTIONS.md`, and
`docs/VERIFICATION_PROTOCOL.md` before starting. Phase 1 must be complete and
its tests passing.

Prime directive: follow LMCache structure. Make decisions yourself; do not
ask the user. The user reviews after the phase.

## Goal

Prove that the layerwise prefill from Phase 1, run end-to-end on a HF
Llama or Mistral model with the stub blender, produces the same prefill
result as the model's stock `forward(..., use_cache=True)`.

"Same" means within the tolerances in `docs/VERIFICATION_PROTOCOL.md` §2:

| dtype | rtol  | atol  |
|-------|-------|-------|
| fp32  | 1e-5  | 1e-5  |
| fp16  | 1e-3  | 1e-3  |
| bf16  | 1e-2  | 1e-2  |

For both **final hidden states** and **all per-layer K, V**.

## Approach

`LMCBaseModel.compute_layer` already does the layer-by-layer prefill,
and as of Phase 1 the stub blender's `process_qkv` applies RoPE to
(q, k) before returning. To compare K and V per layer, you need to
capture each layer's K and V as the loop runs.

Use a recording proxy around the blender (do NOT modify the stub
blender's `process_qkv` itself, so Phase 3 can drop in the full
`LMCBlender` unchanged):

```python
class RecordingBlender:
    def __init__(self, inner):
        self._inner = inner
        self.captured_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Forward metadata access so compute_layer can still read .metadata etc.
        self.layerwise_model = inner.layerwise_model
        self.metadata = inner.metadata
        self.common_metadata = inner.common_metadata
        self.num_layers = inner.num_layers

    def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
        q, k, v, residual, attn_output, attn_metadata = self._inner.process_qkv(
            q, k, v, residual, layer_id, attn_output, attn_metadata
        )
        # Clone to defend against later in-place mutation by attention or
        # subsequent layers (in Phase 3 this becomes critical because
        # process_qkv writes back through old_k[imp_indices]; in Phase 2
        # with the stub blender it's still cheap insurance).
        self.captured_kv.append((k.detach().clone(), v.detach().clone()))
        return q, k, v, residual, attn_output, attn_metadata
```

Drive `compute_layer` to completion via a small loop that exhausts the
generator (no real `blend_layer` — there is no cache engine in Phase 2):

```python
g = lmc_model.compute_layer(input_ids)
for _ in range(lmc_model.num_layers):
    next(g)
```

The recorded K, V at index `i` have shape `(seq_len, num_kv_heads, head_size)`
(post-reshape inside `compute_layer`), with RoPE already applied to K.

## Test file

`tests/test_phase2_equivalence.py`. Structure:

```python
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lmc.compute.models.utils import infer_model_from_hf
from lmc.compute.blend.blender import LMCBlender

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}

@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_equivalence(model_name, dtype):
    ...
```

Test body:
1. Load model with `torch_dtype=dtype`, `attn_implementation="eager"`,
   on `cuda:0`.
2. Build a 128-token random prompt (fixed seed).
3. Run stock forward in prefill mode. To get per-layer hidden states
   in transformers 5.x, pass `output_hidden_states=True` and read from
   the returned object's `hidden_states` tuple (populated by the
   `@capture_outputs` decorator declared in `LlamaPreTrainedModel._can_record_outputs`).
   The final hidden state is `outputs.hidden_states[-1]`, but **note
   that the equivalence test should compare against the hidden state
   BEFORE the final `model.norm` step**, because `compute_layer`
   yields after the MLP of the last layer and does not run `model.norm`.
   Two acceptable strategies:
   (a) compare `compute_layer`'s post-loop hidden state to
       `outputs.hidden_states[-1]` (the last entry includes
       `model.norm`) by applying `model.norm` to the captured hidden
       state ourselves before comparing; OR
   (b) hook just before `model.norm` to capture the pre-norm hidden
       and compare directly. Strategy (a) is simpler.

   ```python
   out_stock = model(
       input_ids=ids[None, :],        # add batch dim
       use_cache=True,
       output_hidden_states=True,
       return_dict=True,
   )
   stock_kv = out_stock.past_key_values    # transformers.Cache object
   stock_hidden_final = out_stock.hidden_states[-1]
   # If using strategy (a): also need model.model.norm applied to our
   # captured hidden state.
   ```

   Extract per-layer K, V from the Cache object:

   ```python
   def extract_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
       if hasattr(cache, "layers"):              # transformers 5.x
           return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
       if hasattr(cache, "key_cache"):           # transformers 4.36–4.45
           return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
       # legacy tuple-of-tuples
       return cache[layer_idx][0], cache[layer_idx][1]
   ```

   Both K and V come back shaped `(batch=1, num_kv_heads, seq_len, head_dim)`;
   reshape to `(seq_len, num_kv_heads, head_dim)` before comparing to
   the layerwise capture.

4. Build the blender + `LMCLlamaModel`, attach the recording proxy,
   run the generator to completion. Collect
   `layerwise_kv[i] = (K, V)` per layer (shape
   `(seq_len, num_kv_heads, head_size)`) and the final hidden state.
   Apply `hf_model.model.norm` to the final hidden state to compare
   against `outputs.hidden_states[-1]`.
5. For each layer i, `allclose(stock_kv_i, layerwise_kv[i][0])` and
   same for V. For the last hidden state, `allclose(stock_hidden_final, normed_last)`.
6. Log per-layer max abs error.

Notes:
- HF's stock K has **RoPE already applied**; `compute_layer` (via the
  Phase 1 stub blender) also applies RoPE inside `process_qkv`, so the
  comparison is apples-to-apples.
- Setting `attn_implementation="eager"` and not passing
  `attention_mask` causes HF to build its own causal mask
  (`LlamaModel.forward` calls `create_causal_mask`). The layerwise
  path's eager backend (Phase 1) builds an equivalent causal mask from
  `attn_metadata`. They should produce identical attention scores in
  fp32.
- For Mistral, `sliding_window` may be set in the config; for short
  prompts (128 tokens) this has no effect, but if you raise the test
  prompt length above the model's sliding window the comparison breaks.
  Keep the test prompt ≤ 4096 tokens.

## Verification

```bash
pytest tests/test_phase2_equivalence.py -v
```

Pass criteria per `docs/VERIFICATION_PROTOCOL.md` §2.

Do not proceed to Phase 3 without user review.

## 작업 보고

작업이 끝나면 `reports/phase2.md` 를 작성하라. 양식은
`docs/CODING_CONVENTIONS.md` §"Phase reports (Korean)" 참고. 보고서의
"검증 결과" 섹션에 (model, dtype) 별 pass/fail, layer 별 max abs error
(K, V, 마지막 hidden state), 그리고 layerwise vs stock 메모리 사용량을
반드시 포함하라. 보고서 작성 후 stdout 에는 한 줄 안내만 출력하라.

---

## Review notes

- Removed the "if the stub blender does not include RoPE, fix it now"
  branch; the RoPE-in-stub decision is now resolved in Phase 1.
- Replaced `stock_kv[i][0]` / `stock_kv[i][1]` indexing with a
  feature-detecting `extract_kv` helper. In transformers 5.7.0
  `past_key_values` is a `Cache` with `.layers[i].keys/.values`
  (`cache_utils.py:1173-1213`); older versions use `.key_cache[i]` or
  a tuple of tuples. The previous prompt asserted the legacy shape.
- Added an explicit note that `compute_layer` does NOT apply
  `model.norm` and that the test driver must apply it to the captured
  final hidden state before comparing to `outputs.hidden_states[-1]`.
- Replaced the "capture hook on the stub blender" suggestion with a
  recording-proxy pattern so the stub blender itself stays
  byte-identical and Phase 3 can drop in the real blender without
  Phase 2 re-edits. Added a clone to defend against in-place mutation.
- Flagged the Mistral `sliding_window` constraint on prompt length.
