# HF parallel of lmcache/v1/gpu_connector/gpu_connectors.py:
# VLLMBufferLayerwiseGPUConnector. Strips the vLLM paged-memory
# write-back path entirely — for the HF port the GPU buffer *is* the
# final KV store; there is no paged backend behind it.
#
# Coroutine shape (`batched_to_gpu`) preserves the LMCache contract:
# yields num_layers + 2 times. Each yield except the last and the
# (num_layers+1)-th receives the per-chunk MemoryObj list for that
# layer; FusedRope is applied to K from cached positions to the
# new positions; gap positions are zeroed on both K and V.
#
# `get_kv(layer_id)` returns *views* into the per-layer GPU buffer.
# `process_qkv`'s in-place `old_k[imp_indices] = k` mutates this same
# storage, so subsequent layers' attention sees the updated K.
from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import torch

from lmc.compute.positional_encoding import FusedRope
from lmc.storage import MemoryFormat, MemoryObj


class HFBufferLayerwiseGPUConnector:
    """
    Per-layer GPU KV buffer. Loads chunk KVs via `batched_to_gpu`,
    applies `FusedRope` to K, zeros gap positions. Exposes `get_kv`
    so `LMCBlender.process_qkv` can mutate K/V in place across layers.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        fused_rotary_emb: Optional[FusedRope] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim_size = num_kv_heads * head_dim
        self.dtype = dtype
        self.device = device
        self.fused_rotary_emb = fused_rotary_emb

        # Per-layer (K, V) GPU buffers; populated on each request.
        self._buffer: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._buf_start: int = 0
        self._buf_end: int = 0
        self._current_gap_positions: Optional[torch.Tensor] = None

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return views into the per-layer (K, V) buffer. Mutations via
        `old_k[imp_indices] = k` inside `process_qkv` must be visible
        to subsequent layers' reads, so we return the live tensors,
        not copies. Mirrors gpu_connectors.py:729-739.
        """
        if layer_id not in self._buffer:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")
        return self._buffer[layer_id]

    def reset(self) -> None:
        """Drop all per-request state. Call between requests."""
        self._buffer.clear()
        self._buf_start = 0
        self._buf_end = 0
        self._current_gap_positions = None

    def get_buffer_token_count(self) -> int:
        return max(0, self._buf_end - self._buf_start)

    def batched_to_gpu(
        self,
        starts: List[int],
        ends: List[int],
        **kwargs,
    ) -> Generator[None, Optional[List[MemoryObj]], None]:
        """
        Mirror of gpu_connectors.py:752-907. Differences:
          - No vLLM paged-memory writeback step (removed entirely).
          - No ping-pong double buffer; the HF port allocates one
            (K, V) pair per layer up front and writes directly into it.
          - Otherwise the yield contract is preserved: total
            `num_layers + 2` yields, with yields 0..num_layers-1
            receiving the chunk MemoryObj list for that layer.

        For each layer, this method:
          1. Copies each chunk's K, V from the MemoryObj into the
             per-layer GPU buffer at `[start - buf_offset : end - buf_offset]`.
          2. At layer 0 only: populates `old_positions_full` from each
             chunk's `metadata.cached_positions`.
          3. Applies `fused_rotary_emb(old_positions_full,
             new_positions_full, K)` over the whole buffer.
          4. Zeros K *and* V at gap positions (positions in
             [starts[0], ends[-1]) not covered by any chunk).
        """
        if not starts:
            raise ValueError("batched_to_gpu requires at least one chunk")

        buf_offset = starts[0]
        num_all_tokens = ends[-1] - starts[0]
        self._buf_start = starts[0]
        self._buf_end = ends[-1]

        device, dtype = self.device, self.dtype

        # Pre-allocate per-layer (K, V) buffers. View-aliased writes
        # from process_qkv land here.
        for layer_id in range(self.num_layers):
            k = torch.zeros((num_all_tokens, self.hidden_dim_size), dtype=dtype, device=device)
            v = torch.zeros((num_all_tokens, self.hidden_dim_size), dtype=dtype, device=device)
            self._buffer[layer_id] = (k, v)

        # Gap mask: True for positions [starts[0], ends[-1]) that no
        # chunk covers. Mirrors gpu_connectors.py:792-800.
        gap_mask = torch.ones(num_all_tokens, dtype=torch.bool, device=device)
        for s, e in zip(starts, ends, strict=False):
            gap_mask[s - buf_offset:e - buf_offset] = False
        self._current_gap_positions = torch.where(gap_mask)[0]

        # Contiguous range from first chunk's start to last chunk's end.
        # Mirrors gpu_connectors.py:803-806.
        new_positions_full = torch.arange(
            starts[0], ends[-1], dtype=torch.int64, device=device,
        )
        old_positions_full = torch.zeros(
            (num_all_tokens,), dtype=torch.int64, device=device,
        )

        # ------------------------------------------------------------
        # Yield loop. Total: num_layers + 2 yields, matching LMCache.
        # Yield i (0 <= i < num_layers): receive layer-i chunk mem objs.
        # Yield num_layers: synchronization sentinel (no value).
        # Yield num_layers+1: final flush sentinel (no value).
        # ------------------------------------------------------------
        for layer_id in range(self.num_layers):
            # `yield` returns whatever the caller .send()s; the cache
            # engine `retrieve_layer` sends a list of per-chunk MemoryObj
            # for this layer.
            mem_objs_layer = yield
            if mem_objs_layer is None:
                # Caller signalled "no chunks for this layer" — leave
                # the pre-zeroed buffer alone. Still apply gap zero
                # (which is the whole buffer) so behaviour stays
                # consistent.
                continue

            k_buf, v_buf = self._buffer[layer_id]
            for start, end, mem_obj in zip(starts, ends, mem_objs_layer, strict=False):
                assert mem_obj.metadata.fmt == MemoryFormat.KV_2TD, (
                    f"unexpected MemoryFormat {mem_obj.metadata.fmt!r}"
                )
                slot_lo = start - buf_offset
                slot_hi = end - buf_offset

                # mem_obj.tensor: (2, chunk_len, hidden_dim_kv) on CPU.
                k_buf[slot_lo:slot_hi].copy_(
                    mem_obj.tensor[0].to(device, non_blocking=True)
                )
                v_buf[slot_lo:slot_hi].copy_(
                    mem_obj.tensor[1].to(device, non_blocking=True)
                )

                if layer_id == 0:
                    cached = mem_obj.metadata.cached_positions
                    assert cached is not None, (
                        "MemoryObj for chunk missing cached_positions; "
                        "store_from_prefill must populate this."
                    )
                    old_positions_full[slot_lo:slot_hi] = cached.to(device)

            # Apply FusedRope to K (rotation from cached_positions to
            # new positions). Mirrors gpu_connectors.py:855-862.
            if self.fused_rotary_emb is not None:
                k_rot = self.fused_rotary_emb(
                    old_positions_full, new_positions_full, k_buf,
                )
                k_buf.copy_(k_rot)

            # Gap zero — must happen *after* RoPE so any garbage the
            # rotation wrote at gap rows is wiped. Mirrors
            # gpu_connectors.py:864-866 (zeros both K and V).
            if self._current_gap_positions.numel():
                k_buf[self._current_gap_positions] = 0.0
                v_buf[self._current_gap_positions] = 0.0

        # `layer_id == num_layers` yield: bare yield, no receive.
        # Mirrors gpu_connectors.py:895-896.
        yield

        # Final flush yield (gpu_connectors.py:907).
        yield
