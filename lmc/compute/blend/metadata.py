# Port of lmcache/v1/compute/blend/metadata.py.
# Phase 1 already declares the full LMCBlendMetadata shape (positions,
# imp_indices, attn_mask, clean()) so Phase 3 can drop in the real
# blender without rewriting the dataclass.
# LMCBlendCommonMetadata is also declared now (default-None fields) so
# Phase 3's blender constructor can populate it without us re-shaping.
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class LMCBlendCommonMetadata:
    """CommonMetadata (fixed hyperparams) for blending operations."""
    # Phase 3 populates these; Phase 1 leaves them as defaults.
    check_layers: Optional[List[int]] = None
    recomp_ratios: Optional[List[float]] = None
    thresholds: Optional[List[float]] = None


@dataclass
class LMCBlendMetadata:
    """Per-request blending metadata. Reset between requests via clean()."""
    imp_indices: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None

    def clean(self):
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None
