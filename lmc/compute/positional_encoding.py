# HF-adapted port of lmcache/v1/compute/positional_encoding.py.
#
# Key deviation from LMCache: we build cos/sin tables from the HF
# model's own `LlamaRotaryEmbedding` (or `MistralRotaryEmbedding`)
# rather than from `vllm_get_rope`. This is the audit-decided
# widening: HF's rotary natively handles Llama-3 rope_scaling, so
# `FusedRope.fused_encode` works for Mistral-7B *and* Llama-3.1-8B
# even though LMCache's own `validate_rope_params` would reject
# rope_scaling != None.
#
# Implementation of fused_encode (pure PyTorch, no CUDA kernel):
#   apply_rope(x, cos, sin)          = x*cos + rotate_half(x)*sin
#   apply_rope(x, cos, -sin)         = inverse rotation
#   fused_encode(old_pos, new_pos, k) =
#       apply_rope( apply_rope(k, cos_old, -sin_old), cos_new, sin_new )
# Mathematically equivalent to LMCache's CUDA kernel in fp32; we cast
# to fp32 for the rotation and back to k's dtype on return.
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    rotate_half,
)


class BasicReverseRope:
    """
    Mirror of lmcache/v1/compute/positional_encoding.py:22-52.

    LMCache uses this only inside `validate_reverse_correctness` (i.e.
    nowhere on the hot path). The HF port keeps the class for parity
    but the implementation is much simpler: with HF's
    `apply_rotary_pos_emb` exposed, the reverse is just `(cos, -sin)`.
    """

    def __init__(self, hf_rope_fn: Callable, head_dim: int, is_neox_style: bool = True):
        # hf_rope_fn(x_4d, position_ids_2d) -> (cos, sin)
        self._rope_fn = hf_rope_fn
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style

    def reverse_encode(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ):
        """Undo `rope(positions, q, k)`."""
        # q, k arrive in `(seq_len, num_heads_or_kv * head_dim)`.
        seq_len = q.shape[0]
        q4 = q.view(seq_len, -1, self.head_dim).unsqueeze(0).transpose(1, 2)
        k4 = k.view(seq_len, -1, self.head_dim).unsqueeze(0).transpose(1, 2)
        cos, sin = self._rope_fn(q4, positions.unsqueeze(0))
        q_unrot, k_unrot = apply_rotary_pos_emb(q4, k4, cos, -sin)
        q_out = q_unrot.transpose(1, 2).reshape(seq_len, -1)
        k_out = k_unrot.transpose(1, 2).reshape(seq_len, -1)
        return q_out, k_out

    def __call__(self, positions, q, k):
        return self.reverse_encode(positions, q, k)


class FusedRope:
    """
    Mirror of lmcache/v1/compute/positional_encoding.py:55-82, but
    backed by HF's rotary tables (so Llama-3 rope_scaling is honoured)
    and implemented in pure PyTorch (no `lmc_ops.rotary_embedding_k_fused`
    CUDA kernel — correctness > speed, per docs/CODING_CONVENTIONS.md).

    `fused_encode(old_positions, new_positions, k)` rotates each row of
    K *from* the cached positions *to* the new positions in this
    request. K's expected shape is 2-D `(num_tokens, num_kv_heads *
    head_dim)`, mirroring LMCache's call from
    `gpu_connectors.py:864-862`.

    Returns the rotated K with the same shape and dtype as the input
    (LMCache's CUDA kernel writes in-place AND returns; we return
    a new tensor and let the connector assign it back).
    """

    def __init__(
        self,
        hf_rope_fn: Callable,
        head_dim: int,
        is_neox_style: bool = True,
    ):
        # hf_rope_fn(x_4d, position_ids_2d) -> (cos, sin)
        # Source: hf_model.model.rotary_emb (forward signature
        # established in `transformers/models/llama/modeling_llama.py`).
        self._rope_fn = hf_rope_fn
        self.head_dim = head_dim   # named `head_size` in LMCache; same value
        self.head_size = head_dim
        self.is_neox_style = is_neox_style

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """`x * cos + rotate_half(x) * sin` with broadcastable cos/sin."""
        return x * cos + rotate_half(x) * sin

    def fused_encode(
        self,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        K with shape `(num_tokens, num_kv_heads * head_dim)`.
        Both `old_positions` and `new_positions` are 1-D int64 tensors
        of length `num_tokens` on the same device as K.
        """
        num_tokens = k.shape[0]
        device = k.device
        orig_dtype = k.dtype

        # Reshape K -> (1, num_kv_heads, num_tokens, head_dim) for the
        # apply_rope broadcast (cos/sin will be (1, 1, num_tokens, D)).
        k_4d = k.view(num_tokens, -1, self.head_dim).unsqueeze(0).transpose(1, 2)
        # Promote to fp32 for the rotation to keep numerical error well
        # below the dtype tolerance (1e-3 fp16 / 1e-2 bf16).
        k_4d = k_4d.to(torch.float32)

        # HF's rotary forward signature: rotary(x, position_ids).
        # It ignores most of x except dtype/device; passing the 4-D k
        # is fine. We pass position_ids as (1, num_tokens).
        cos_old, sin_old = self._rope_fn(k_4d, old_positions.unsqueeze(0).to(device))
        cos_new, sin_new = self._rope_fn(k_4d, new_positions.unsqueeze(0).to(device))

        # cos/sin arrive shaped (1, num_tokens, head_dim). Match HF's
        # apply_rotary_pos_emb broadcasting convention (unsqueeze_dim=1).
        cos_old = cos_old.unsqueeze(1).to(torch.float32)
        sin_old = sin_old.unsqueeze(1).to(torch.float32)
        cos_new = cos_new.unsqueeze(1).to(torch.float32)
        sin_new = sin_new.unsqueeze(1).to(torch.float32)

        # Inverse-rotate from cached positions, then forward-rotate to
        # new positions. Mathematically equivalent to a single rotation
        # by `delta = new_pos - old_pos`, but the two-step form keeps
        # parity with HF's `apply_rotary_pos_emb` (so the same code
        # path runs on Llama-3 with rope_scaling applied through
        # `_rope_fn`).
        k_unrot = self._apply_rope(k_4d, cos_old, -sin_old)
        k_rot   = self._apply_rope(k_unrot, cos_new,  sin_new)

        k_rot = k_rot.to(orig_dtype)
        # (1, H_kv, T, D) -> (T, H_kv, D) -> (T, H_kv*D)
        out = k_rot.transpose(1, 2).reshape(num_tokens, -1)
        return out

    def __call__(self, old_positions, new_positions, k):
        return self.fused_encode(old_positions, new_positions, k)


def validate_rope_params(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> bool:
    """
    HF-widened version of LMCache's validator. LMCache rejects
    rope_scaling != None; we accept it because the HF rotary handles
    Llama-3 scaling internally. The remaining checks (rotary_dim ==
    head_size and partial_rotary_factor == 1.0) are still enforced.
    """
    if rotary_dim != head_size:
        return False
    if partial_rotary_factor != 1.0:
        return False
    return True


def validate_reverse_correctness(
    fused_rope: "FusedRope",
    head_dim: int,
    num_kv_heads: int = 4,
    num_tokens: int = 10,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
) -> bool:
    """
    Sanity check: fused_encode(old, new, k) should equal directly-rotated
    K at `new` positions, given that K was rotated at `old` positions
    coming in. Mirrors lmcache/v1/compute/positional_encoding.py:112-144
    (but uses HF's `apply_rotary_pos_emb` directly rather than
    BasicReverseRope+vllm_get_rope).
    """
    device = device or torch.device("cpu")
    hidden = num_kv_heads * head_dim
    torch.manual_seed(0)
    raw_k = torch.rand(num_tokens, hidden, device=device, dtype=dtype)

    old_positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    new_positions = torch.arange(100, 100 + num_tokens, device=device, dtype=torch.int64)

    # Reference: rotate raw_k at `new` positions directly.
    raw_k_4d = raw_k.view(num_tokens, num_kv_heads, head_dim).unsqueeze(0).transpose(1, 2).to(torch.float32)
    cos_new, sin_new = fused_rope._rope_fn(raw_k_4d, new_positions.unsqueeze(0))
    cos_new = cos_new.unsqueeze(1).to(torch.float32)
    sin_new = sin_new.unsqueeze(1).to(torch.float32)
    k_new_ref = FusedRope._apply_rope(raw_k_4d, cos_new, sin_new)
    k_new_ref_2d = k_new_ref.transpose(1, 2).reshape(num_tokens, hidden).to(dtype)

    # FusedRope path: take K already rotated at `old`, fuse-encode to `new`.
    cos_old, sin_old = fused_rope._rope_fn(raw_k_4d, old_positions.unsqueeze(0))
    cos_old = cos_old.unsqueeze(1).to(torch.float32)
    sin_old = sin_old.unsqueeze(1).to(torch.float32)
    k_at_old = FusedRope._apply_rope(raw_k_4d, cos_old, sin_old)
    k_at_old_2d = k_at_old.transpose(1, 2).reshape(num_tokens, hidden).to(dtype)

    k_at_new_fused = fused_rope.fused_encode(old_positions, new_positions, k_at_old_2d)
    return torch.allclose(k_at_new_fused, k_new_ref_2d, atol=atol, rtol=atol)


def get_fused_rope(
    hf_model,
    head_dim: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    partial_rotary_factor: float = 1.0,
) -> Optional[FusedRope]:
    """
    Build a FusedRope wired to `hf_model.model.rotary_emb`. Mirrors
    `get_fused_rope` at lmcache/v1/compute/positional_encoding.py:148-202,
    minus the vllm_get_rope path: we always source cos/sin from the HF
    model's own rotary.
    """
    if not validate_rope_params(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position=hf_model.config.max_position_embeddings,
        base=getattr(hf_model.config, "rope_theta", 10000.0),
        is_neox_style=is_neox_style,
        rope_scaling=rope_scaling,
        partial_rotary_factor=partial_rotary_factor,
    ):
        return None

    hf_rope = hf_model.model.rotary_emb

    def hf_rope_fn(x_4d: torch.Tensor, position_ids_2d: torch.Tensor):
        # HF's rotary takes a 4-D tensor (any shape) and 2-D positions;
        # returns (cos, sin) of shape (batch, seq_len, head_dim).
        return hf_rope(x_4d, position_ids_2d)

    return FusedRope(hf_rope_fn=hf_rope_fn, head_dim=head_dim, is_neox_style=is_neox_style)
