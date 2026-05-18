# Phase 1 stub blender — for Phase 1 / Phase 2 test-time use only.
# Phase 3 replaced the original stub with the full `LMCBlender` (with
# cache_engine / gpu_connector / config constructor args). The full
# blender is the production class; this stub stays around so that
# Phase 1 §1 (layerwise prefill skeleton) and Phase 2 §2 (stock-vs-
# layerwise equivalence) can be tested without standing up a cache
# engine, segment DB, and GPU connector — none of which exist at
# Phase 1 / Phase 2 time.
#
# Behaviour mirrors the original Phase 1 stub described in
# `docs/phases/PHASE1_PROMPT.md` "Stub blender" section and
# `reports/phase1.md` §1. Specifically:
#   - constructor takes (hf_model, num_layers) only;
#   - `process_qkv` applies the layer's RoPE (which the Phase 0 audit
#     mandated must be in the stub so Phase 2's comparison sees
#     post-RoPE K just like the real CacheBlend would);
#   - everything else is a pass-through (no HKVD branch, no
#     `old_k[imp_indices] = k` write-back).
from __future__ import annotations

from typing import Optional

import torch

from lmc.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata


class LMCStubBlender:
    """
    Phase 1 stub. Mirrors `lmcache/v1/compute/blend/blender.py:18-86`
    (pre-HKVD prefix) without the cache_engine / gpu_connector
    dependencies the full LMCBlender needs.

    Test files instantiate this directly:

        stub = LMCStubBlender(hf_model, num_layers=len(hf_model.model.layers))
        lmc_model = infer_model_from_hf(hf_model, stub)
        # LMCBaseModel.__init__ runs `stub.layerwise_model = self` —
        # so `process_qkv` can read `self.layerwise_model.vllm_model...`.
    """

    def __init__(self, hf_model, num_layers: int):
        # The real `LMCBlender` constructor takes
        # (cache_engine, gpu_connector, hf_model, config). The stub
        # only needs the model handle (kept under `vllm_model` name to
        # match the source) and `num_layers` for parity with the full
        # blender's attribute.
        self.vllm_model = hf_model
        self.num_layers = num_layers
        # `layerwise_model` is set by `LMCBaseModel.__init__` after the
        # base model is constructed, matching the real wiring.
        self.layerwise_model = None
        # `metadata` holds `positions` across layers within one request.
        self.metadata = LMCBlendMetadata()
        # `common_metadata` is kept as a default-None dataclass so
        # Phase 1 / Phase 2 tests can read its fields without a separate
        # branch from the full blender.
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
        """
        Mirrors the pre-HKVD prefix of
        `lmcache/v1/compute/blend/blender.py:59-86`. No HKVD branch
        and no `old_k[imp_indices] = k` write-back (those are Phase 3).
        """
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

        return q, k, v, residual, attn_output, attn_metadata
