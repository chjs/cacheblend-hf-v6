# Full LMCBlender — replaces the Phase 1 stub.
# Mirrors lmcache/v1/compute/blend/blender.py line-by-line; the audit
# in docs/LMCACHE_IMPLEMENTATION.md §2.3 already verified every claim
# in here against the LMCache source.
from __future__ import annotations

from typing import List, Optional, Union

import torch

from lmc.compute.attention.metadata import LMCAttnMetadata
from lmc.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmc.config import LMCacheEngineConfig


class LMCBlender:
    """
    Cache-blender backend for LMCache (HF port).

    Mirrors `lmcache/v1/compute/blend/blender.py:18-168` with the
    audit-approved adaptations:
      - `vllm_model` param → `hf_model`, exposed as `self.vllm_model`
        (so `process_qkv`'s `self.layerwise_model.vllm_model...` lines
        stay byte-identical to LMCache);
      - `infer_model_from_hf` instead of `infer_model_from_vllm`.

    The Phase 1 stub had only the constructor + the pre-HKVD prefix of
    `process_qkv`. Phase 3 fills in everything else.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        hf_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        # Late import to avoid circular dependency
        # (compute/models/utils.py imports compute/blend/metadata.py
        # via base.py).
        from lmc.compute.models.utils import infer_model_from_hf

        # `hf_model` is exposed as `self.vllm_model` to keep
        # process_qkv's `self.layerwise_model.vllm_model.model.layers[...]`
        # identical to LMCache's source.
        self.vllm_model = hf_model
        self.layerwise_model = infer_model_from_hf(hf_model, self, enable_sparse)

        # TODO: remove this hardcode (matches LMCache's comment).
        self.num_layers = len(hf_model.model.layers)

        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )

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
        """
        Byte-for-byte port of LMCache's `process_qkv`
        (lmcache/v1/compute/blend/blender.py:59-120). The audit
        validated every line of this body against the source; see
        docs/LMCACHE_IMPLEMENTATION.md §2.3.
        """
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

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

            # TODO(Jiayi): remove `[0]` hardcode
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

    # ------------------------------------------------------------------
    # `blend_layer` / `blend` — drive `compute_layer` and `retrieve_layer`
    # in lockstep. Byte-for-byte port of
    # lmcache/v1/compute/blend/blender.py:122-168.
    # ------------------------------------------------------------------
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Layerwise retrieve + blending."""
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

    def blend(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run the full blend pipeline for one request."""
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).to(next(self.vllm_model.parameters()).device)

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
