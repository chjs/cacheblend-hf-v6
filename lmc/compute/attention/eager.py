# HF eager replacement for lmcache/v1/compute/attention/flash_attn.py.
# Same `forward_contiguous` interface as LMCFlashAttnBackend so the call
# site in `LMCBaseModel.compute_layer` is unchanged.
import torch
import torch.nn.functional as F

from lmc.compute.attention.abstract import AttentionInterface
from lmc.compute.attention.metadata import LMCAttnMetadata, LMCEagerAttnMetadata


class LMCEagerAttnBackend(AttentionInterface):
    """
    Eager (pure-PyTorch) attention backend. Replaces LMCFlashAttnBackend
    for the HF port. Builds an explicit causal mask that respects
    `q_positions` so HKVD-restricted queries attend to the right keys
    at later layers.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        scaling: float,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scaling = scaling
        self.num_kv_groups = num_heads // num_kv_heads

    def forward_contiguous(
        self,
        query: torch.Tensor,   # (n_q, num_heads,    head_size)
        key:   torch.Tensor,   # (n_k, num_kv_heads, head_size)
        value: torch.Tensor,   # (n_k, num_kv_heads, head_size)
        output: torch.Tensor,  # (n_q, num_heads,    head_size), pre-allocated
        attn_metadata: LMCAttnMetadata,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, LMCEagerAttnMetadata)

        n_q = query.shape[0]
        n_k = key.shape[0]
        H, D = self.num_heads, self.head_size

        # Expand K, V across GQA groups so heads align with Q.
        if self.num_kv_groups != 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        # Move heads to dim 0 for batched-matmul friendliness.
        # q_h: (H, n_q, D); k_h: (H, n_k, D); v_h: (H, n_k, D)
        q_h = query.transpose(0, 1)
        k_h = key.transpose(0, 1)
        v_h = value.transpose(0, 1)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scaling  # (H, n_q, n_k)

        # Causal mask. Each query row's effective position is in
        # attn_metadata.q_positions; default is arange(n_q). The key
        # positions default to arange(n_k).
        device = query.device
        q_pos = attn_metadata.q_positions
        if q_pos is None:
            q_pos = torch.arange(n_q, device=device, dtype=torch.int64)
        k_pos = attn_metadata.k_positions
        if k_pos is None:
            k_pos = torch.arange(n_k, device=device, dtype=torch.int64)

        # allowed[i, j] = k_pos[j] <= q_pos[i]
        allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (n_q, n_k)
        # Cast and broadcast over heads.
        # We add a large negative number instead of -inf to avoid NaN
        # when an entire row would be masked (shouldn't happen here, but
        # cheap insurance for the early-token rows).
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=scores.dtype, device=device),
            torch.full((), float("-inf"), dtype=scores.dtype, device=device),
        )
        scores = scores + mask.unsqueeze(0)

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(attn, v_h)  # (H, n_q, D)
        out = out.transpose(0, 1).contiguous()  # (n_q, H, D)

        # In-place write to honor the LMCFlashAttn `out=output` contract;
        # the caller (compute_layer) reuses this storage.
        output.copy_(out)
        return output

    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> LMCEagerAttnMetadata:
        seq_len = input_ids.shape[0]
        device = input_ids.device
        return LMCEagerAttnMetadata(
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_query_len=seq_len,
            max_seq_len=seq_len,
            q_positions=None,
            k_positions=None,
        )
