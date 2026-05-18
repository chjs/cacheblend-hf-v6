"""
Phase 4 runtime utilities: HF model patch/unpatch (so we can switch
between blend prefill and stock generate), connector-buffer →
DynamicCache conversion, and a manual decode loop that walks the model
forward one token at a time when starting from a pre-filled cache.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import DynamicCache


# Module-level constants imported by the main script.
PATCH_MARKER = "_lmc_patched"


# ---------------------------------------------------------------------------
# Patch / unpatch.
#
# `LMCBaseModel.__init__` patches `o_proj`, `input_layernorm` and
# `post_attention_layernorm` with nn.Module wrappers (`_OProjTupled`,
# `_ResidualRMSNorm`). Each wrapper stores the original on `self.inner`,
# so the inverse is straightforward.
# ---------------------------------------------------------------------------
def unpatch_hf_model(hf_model: nn.Module) -> None:
    """Restore an `hf_model` previously patched by `LMCBaseModel.__init__`.

    Safe to call on a model that wasn't patched (no-op).
    """
    for layer in hf_model.model.layers:
        attn = layer.self_attn
        # Restore o_proj from _OProjTupled.inner (a Linear).
        op = getattr(attn, "o_proj", None)
        if op is not None and hasattr(op, "inner"):
            attn.o_proj = op.inner
        # input / post_attention_layernorm: restore from _ResidualRMSNorm.inner
        iln = layer.input_layernorm
        if hasattr(iln, "inner"):
            layer.input_layernorm = iln.inner
        pln = layer.post_attention_layernorm
        if hasattr(pln, "inner"):
            layer.post_attention_layernorm = pln.inner
        # Drop the additive attributes (qkv_proj, rotary_emb, q_size, kv_size).
        # `qkv_proj` and `rotary_emb` are plain functions on self_attn —
        # nn.Module.__delattr__ doesn't complain about non-Module fields.
        for attr in ("qkv_proj", "rotary_emb", "q_size", "kv_size"):
            if hasattr(attn, attr):
                try:
                    delattr(attn, attr)
                except AttributeError:
                    pass

    if hasattr(hf_model.model, "embed_input_ids"):
        try:
            delattr(hf_model.model, "embed_input_ids")
        except AttributeError:
            pass
    if hasattr(hf_model, PATCH_MARKER):
        try:
            delattr(hf_model, PATCH_MARKER)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Connector → DynamicCache.
# ---------------------------------------------------------------------------
def connector_to_dyncache(
    connector,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    target_dtype: torch.dtype,
    device: torch.device,
) -> DynamicCache:
    """
    Convert the connector's per-layer 2-D `(seq_len, H_kv * D)` K/V
    buffers into an HF DynamicCache shaped
    `(1, H_kv, seq_len, D)` per layer.
    """
    cache = DynamicCache()
    for layer_id in range(num_layers):
        k_2d, v_2d = connector.get_kv(layer_id)
        seq_len = k_2d.shape[0]
        k_4d = (
            k_2d.view(seq_len, num_kv_heads, head_dim)
            .transpose(0, 1).unsqueeze(0).contiguous().to(target_dtype)
        )
        v_4d = (
            v_2d.view(seq_len, num_kv_heads, head_dim)
            .transpose(0, 1).unsqueeze(0).contiguous().to(target_dtype)
        )
        # DynamicCache.update on an empty cache simply stores the tensors.
        cache.update(k_4d, v_4d, layer_id)
    return cache


# ---------------------------------------------------------------------------
# Decode from a pre-filled cache.
#
# The cache has positions [0, L-1] (full prompt). HF's generate() refuses
# to forward an empty input, so we crop the cache to length L-1 and
# manually forward the *last* prompt token to get logits at position L-1,
# then iteratively decode greedily.
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_from_full_cache(
    model: nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,        # 1-D long tensor on device
    cache: DynamicCache,             # length == prompt_ids.numel()
    max_new_tokens: int,
) -> Tuple[str, List[int]]:
    """Returns `(generated_text, generated_token_ids)`. Greedy."""
    device = prompt_ids.device
    L = prompt_ids.numel()
    assert cache.get_seq_length() == L, (
        f"cache length {cache.get_seq_length()} != prompt length {L}"
    )

    # Crop the cache to L-1 so the forward call can re-process the last
    # prompt token (and produce its logits).
    cache.crop(L - 1)

    eos_id = getattr(tokenizer, "eos_token_id", None)

    cur_input = prompt_ids[-1:].view(1, 1)  # (1, 1)
    cur_pos = L - 1
    cur_cache = cache

    generated: List[int] = []
    for _ in range(max_new_tokens):
        position_ids = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
        out = model(
            input_ids=cur_input,
            past_key_values=cur_cache,
            position_ids=position_ids,
            use_cache=True,
        )
        # `out.logits` shape: (1, 1, vocab_size).
        next_token = int(out.logits[0, -1].argmax().item())
        if eos_id is not None and next_token == eos_id:
            break
        generated.append(next_token)
        cur_cache = out.past_key_values
        cur_input = torch.tensor([[next_token]], device=device, dtype=torch.long)
        cur_pos += 1

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated


# ---------------------------------------------------------------------------
# Full recompute timing helper (Method 1).
# ---------------------------------------------------------------------------
@torch.no_grad()
def full_recompute_run(
    model: nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[str, float, List[int]]:
    """
    Method 1: full prefill on `prompt_ids`, then greedy decode.

    Returns `(generated_text, prefill_ms, generated_token_ids)`.

    `prefill_ms` is wall-clock time of the prefill forward call (NOT
    including decode). cuda.synchronize is called before / after.
    """
    device = prompt_ids.device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    out = model(
        input_ids=prompt_ids.view(1, -1),
        use_cache=True,
        return_dict=True,
    )
    if cuda:
        torch.cuda.synchronize()
    prefill_ms = (time.time() - t0) * 1000.0
    cache: DynamicCache = out.past_key_values
    # Continue greedy decode using the same code path so all three
    # methods share the decoder loop.
    text, gen_ids = decode_from_full_cache(
        model, tokenizer, prompt_ids, cache, max_new_tokens,
    )
    return text, prefill_ms, gen_ids
