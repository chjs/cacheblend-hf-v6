# Mirrors lmcache/v1/compute/attention/metadata.py. LMCAttnMetadata is
# verbatim. LMCEagerAttnMetadata replaces LMCFlashAttnMetadata for the
# HF eager backend (CODING_CONVENTIONS.md "HF-required adaptations" §7).
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
import abc

import torch


@dataclass
class LMCAttnMetadata(metaclass=abc.ABCMeta):
    @abstractmethod
    def update_from_top_indices(self, top_indices: torch.Tensor):
        raise NotImplementedError("This method should be implemented in subclasses.")


@dataclass
class LMCEagerAttnMetadata(LMCAttnMetadata):
    # Field names mirror LMCFlashAttnMetadata so process_qkv's call
    # `attn_metadata.update_from_top_indices(top_indices)` and the
    # backend's reads (query_start_loc, max_query_len) stay identical.
    query_start_loc: torch.Tensor   # int32, [0, n_q]
    seq_lens: torch.Tensor          # [n_k]
    cu_seqlens_k: torch.Tensor      # int32, [0, n_k]
    max_query_len: int
    max_seq_len: int
    # Extras for the eager backend's explicit causal mask. When
    # update_from_top_indices is called on the check layer, q_positions
    # carries the HKVD-restricted query rows' positions in the full
    # sequence so the mask is built correctly.
    q_positions: Optional[torch.Tensor] = None
    k_positions: Optional[torch.Tensor] = None

    def update_from_top_indices(self, top_indices: torch.Tensor):
        top_k_num = len(top_indices)
        self.max_query_len = top_k_num
        device = self.query_start_loc.device
        dtype = self.query_start_loc.dtype
        self.query_start_loc = torch.tensor([0, top_k_num], dtype=dtype, device=device)
        # cu_seqlens_k and max_seq_len intentionally untouched — they stay
        # at the full request length so the backend sees variable-Q
        # against full-K. (Matches LMCFlashAttnMetadata.update_from_top_indices
        # at lmcache/v1/compute/attention/metadata.py:31-36.)
        self.q_positions = top_indices
