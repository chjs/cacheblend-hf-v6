# Phase 1 stub blender. Phase 3 replaces this with the full LMCBlender
# (lmcache/v1/compute/blend/blender.py:18-168).
#
# The stub keeps the call-site identical to the real one: same
# constructor shape (positional-only here, since there's no cache
# engine yet), same `process_qkv` signature, same return tuple. The
# only thing missing is the HKVD branch and the `old_k[imp_indices]=k`
# write-back — both Phase 3.
#
# RoPE is applied inside process_qkv per the Phase 0 audit decision
# (see docs/phases/PHASE1_PROMPT.md "Review notes"), so Phase 2's
# stock-vs-layerwise equivalence test sees post-RoPE K just like real
# CacheBlend would.
from typing import Optional

import torch

from lmc.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata


class LMCBlender:
    """Phase 1 stub. Phase 3 replaces with the full LMCBlender."""

    def __init__(self, hf_model, num_layers: int):
        # The real constructor (Phase 3) takes
        # (cache_engine, gpu_connector, vllm_model, config). The stub
        # only needs the model handle (kept under vllm_model name to
        # mirror the source) and a num_layers cache for parity.
        self.vllm_model = hf_model
        self.num_layers = num_layers
        # layerwise_model is set by LMCBaseModel.__init__ after the
        # base model is constructed, matching the real wiring.
        self.layerwise_model = None
        # metadata holds `positions` across layers within one request.
        self.metadata = LMCBlendMetadata()
        # common_metadata kept None until Phase 3 fills it.
        self.common_metadata = LMCBlendCommonMetadata()

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata,
    ):
        # Mirrors lmcache/v1/compute/blend/blender.py:59-86 (the
        # pre-HKVD prefix). Phase 3 adds the HKVD branch and the
        # cache write-back below this.
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

        # No HKVD selection, no old_k write-back in Phase 1.
        return q, k, v, residual, attn_output, attn_metadata
