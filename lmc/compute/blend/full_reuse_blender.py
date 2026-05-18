# Prompt Cache / Gim et al. 2023 baseline ("Full KV reuse").
# See docs/phases/PHASE4_PROMPT.md "Method 2 implementation note".
#
# Same retrieval + GPU buffer loading + RoPE shift path as LMCBlender,
# but the HKVD recomputation branch is skipped: Q is freshly rotated,
# K and V are reused unchanged from the cache. The attention path
# consumes the cached (rotated) K and V.
from __future__ import annotations

from typing import Optional

import torch

from lmc.compute.attention.metadata import LMCAttnMetadata
from lmc.compute.blend.blender import LMCBlender


class LMCFullReuseBlender(LMCBlender):
    """
    Drop-in replacement for `LMCBlender` that disables HKVD selection
    so every layer uses the cached (rotated) K, V without any
    recomputation.

    Subclass keeps the full `LMCBlender.__init__` (cache_engine,
    gpu_connector, hf_model, config — only `process_qkv` is overridden).
    """

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape, dtype=q.dtype, device=q.device,
            )

        # Positional encoding — same as LMCBlender's prefix:
        # only Q needs fresh RoPE; cached K was already rotated to the
        # new positions by the GPU connector (`FusedRope`) on load.
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64,
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        # Rotate Q at the new positions. The freshly-projected K is
        # *discarded* (we use old_k from cache instead), so its rotation
        # is irrelevant — but `attn_layer.rotary_emb` always returns
        # both, so we accept the rotated k_throwaway here.
        q, _k_throwaway = attn_layer.rotary_emb(self.metadata.positions, q, k)

        # No HKVD branch, no `imp_indices`. Return cached K, V directly.
        return q, old_k, old_v, residual, attn_output, attn_metadata
