# Minimal `LMCacheEngine` for the HF port.
#   - `retrieve_layer(tokens, ...)`: coroutine mirroring
#     lmcache/v1/cache_engine.py:902-1055 (subset). Walks the
#     SegmentTokenDatabase chunks, drives the GPU connector's
#     `batched_to_gpu` in lockstep, yields num_layers + 2 times.
#   - `store_from_prefill(tokens, per_layer_kv)`: warmup helper used by
#     Phase 3 tests. Runs the SegmentTokenDatabase over `tokens`,
#     slices each chunk's K, V out of `per_layer_kv` (a list of
#     (K, V) tensors from a stock prefill), and inserts a MemoryObj
#     per (chunk, layer) into the storage backend with
#     `metadata.cached_positions` = `arange(start, end)`. This is the
#     equivalent of LMCache's full-prefill store path, simplified for
#     a single-process in-memory store.
from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Union

import torch

from lmc.gpu_connector import HFBufferLayerwiseGPUConnector
from lmc.storage import (
    LayerCacheEngineKey, LocalCPUBackend, MemoryFormat, MemoryObj,
    MemoryObjMetadata,
)
from lmc.token_database import SegmentTokenDatabase


class LMCacheEngine:
    """
    Minimal cache engine for the HF port.

    Drops everything LMCache's full version needs for multi-rank,
    paged-memory, disk offload, async loading, etc. — leaving only the
    pieces CacheBlend's blender requires: chunk segmentation, in-memory
    storage, and the layerwise `retrieve_layer` coroutine.
    """

    def __init__(
        self,
        token_database: SegmentTokenDatabase,
        storage: LocalCPUBackend,
        gpu_connector: HFBufferLayerwiseGPUConnector,
        num_layers: int,
    ):
        self.token_database = token_database
        self.storage = storage
        self.gpu_connector = gpu_connector
        self.num_layers = num_layers

    # ------------------------------------------------------------------
    # Warmup / store path.
    # ------------------------------------------------------------------
    def store_from_prefill(
        self,
        tokens: Union[torch.Tensor, List[int]],
        per_layer_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Slice a stock-prefill KV cache by chunk and insert per-layer
        MemoryObjs into the backend. Returns the number of chunks
        stored.

        :param tokens: token sequence the prefill was run on.
        :param per_layer_kv: list (len == num_layers) of (K, V) tensors
            on any device; each shaped `(batch=1, num_kv_heads, seq_len,
            head_dim)` (the HF DynamicCache layout). Will be reshaped
            and moved to CPU before insertion.
        """
        if isinstance(tokens, torch.Tensor):
            tokens_cpu = tokens.detach().to(device="cpu", dtype=torch.long)
        else:
            tokens_cpu = torch.tensor(tokens, dtype=torch.long, device="cpu")

        # Normalize each (K, V) tensor to (seq_len, num_kv_heads * head_dim)
        # on CPU so chunk slicing is a single index op.
        flat_kv = []
        for k, v in per_layer_kv:
            assert k.dim() == 4 and v.dim() == 4, "expect (1, H_kv, T, D) tensors"
            seq_len = k.shape[2]
            num_kv = k.shape[1]
            head_dim = k.shape[3]
            k2 = k.transpose(1, 2).reshape(seq_len, num_kv * head_dim).to("cpu").contiguous()
            v2 = v.transpose(1, 2).reshape(seq_len, num_kv * head_dim).to("cpu").contiguous()
            flat_kv.append((k2, v2))

        chunks_stored = 0
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens_cpu,
            request_configs=request_configs,
        ):
            cached_positions = torch.arange(start, end, dtype=torch.int64, device="cpu")
            layer_keys = key.split_layers(self.num_layers)
            for layer_id, layer_key in enumerate(layer_keys):
                k_layer, v_layer = flat_kv[layer_id]
                k_chunk = k_layer[start:end].clone()
                v_chunk = v_layer[start:end].clone()
                tensor = torch.stack([k_chunk, v_chunk], dim=0)  # (2, chunk_len, H_kv*D)
                mem_obj = MemoryObj(
                    tensor=tensor,
                    metadata=MemoryObjMetadata(
                        fmt=MemoryFormat.KV_2TD,
                        cached_positions=cached_positions,
                    ),
                )
                self.storage.put(layer_key, mem_obj)
            chunks_stored += 1
        return chunks_stored

    # ------------------------------------------------------------------
    # Retrieve / load path.
    # ------------------------------------------------------------------
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        request_configs: Optional[dict] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Mirror of lmcache/v1/cache_engine.py:902-1055.

        Yield count: `num_layers + 2`. Drives the GPU connector's
        `batched_to_gpu` coroutine in lockstep.

        Yields:
          - yield 0: count of retrieved tokens (Tensor) — surfaced for
            SGLang integration in LMCache; we yield it as a tensor for
            parity.
          - yields 1..num_layers-1: None (per-layer progress sentinel).
          - yield num_layers: None (advances connector past its
            `layer_id == num_layers` synchronization yield).
          - yield num_layers+1: a boolean mask `(len(tokens),)` showing
            which positions were filled by the cache.
        """
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        # Collect chunks present in the backend. Stop at the first miss
        # (LMCache:961-984).
        starts: List[int] = []
        ends: List[int] = []
        keys_per_chunk: List[List[LayerCacheEngineKey]] = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs,
        ):
            layer_keys = key.split_layers(self.num_layers)
            if not self.storage.contains(layer_keys[0]):
                break
            starts.append(start)
            ends.append(end)
            keys_per_chunk.append(layer_keys)
            ret_mask[start:end] = True

        if not starts:
            # No chunks at all — preserve the yield count contract so
            # the blender's drive loop still advances correctly.
            for _ in range(self.num_layers):
                yield None
            yield None
            yield ret_mask
            return

        # keys_per_chunk is chunk-major; the connector wants
        # layer-major lists (one list per layer, one MemoryObj per
        # chunk). Transpose.
        keys_layer_major: List[List[LayerCacheEngineKey]] = [
            [keys_per_chunk[c][L] for c in range(len(keys_per_chunk))]
            for L in range(self.num_layers)
        ]

        # Drive the GPU connector coroutine in lockstep with our yields.
        connector_gen = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
        # Prime to the first yield (layer 0 waits for mem_objs).
        next(connector_gen)

        for layer_id in range(self.num_layers):
            # Fetch this layer's per-chunk MemoryObjs.
            mem_objs_layer = []
            for layer_key in keys_layer_major[layer_id]:
                mem_obj = self.storage.get(layer_key)
                assert mem_obj is not None, (
                    f"missing MemoryObj for layer={layer_id}, key={layer_key!r}"
                )
                mem_objs_layer.append(mem_obj)

            if layer_id == 0:
                yield torch.sum(ret_mask)
            else:
                yield None

            # Send the mem_objs to the connector; it loads them into
            # the GPU buffer, RoPEs K, gap-zeros, and advances to the
            # next yield (layer_id+1 receive, or the post-loop sentinel
            # at layer num_layers).
            connector_gen.send(mem_objs_layer)

        yield None
        # Drive the connector through its `layer_id == num_layers`
        # sentinel yield. LMCache calls `next(mem_obj_consumer)` here
        # (cache_engine.py:1034).
        next(connector_gen)

        yield ret_mask
