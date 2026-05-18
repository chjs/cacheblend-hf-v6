# Minimal in-memory chunk store + auxiliary types LMCache references.
# Mirrors (subset of):
#   lmcache/utils.py             — CacheEngineKey, LayerCacheEngineKey
#   lmcache/v1/metadata.py       — LMCacheMetadata
#   lmcache/v1/memory_management.py
#                                 — MemoryFormat (KV_2TD only), MemoryObj,
#                                   MemoryObjMetadata.cached_positions
# The full LMCache versions carry serializer hooks, reference-counted
# allocator slots, multi-format support, MLA, etc. — none of which the
# HF port needs for a single-process in-memory store.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch


# -----------------------------------------------------------------------------
# Mirror of lmcache/v1/memory_management.py:MemoryFormat. Only KV_2TD is used;
# others are kept as enum members for parity.
# -----------------------------------------------------------------------------
class MemoryFormat(Enum):
    UNDEFINED = 0
    KV_2LTD = auto()
    KV_T2D = auto()
    KV_2TD = auto()              # [2, num_tokens, hidden_dim] — the one we use
    BINARY = auto()
    BINARY_BUFFER = auto()
    KV_MLA_FMT = auto()


# -----------------------------------------------------------------------------
# Mirror of lmcache/v1/metadata.py:LMCacheMetadata (subset).
# -----------------------------------------------------------------------------
@dataclass
class LMCacheMetadata:
    """
    Subset of LMCache's LMCacheMetadata. Fields beyond model_name +
    world_size / worker_id / kv_dtype are dropped (the HF port runs a
    single process; multi-rank coordination and KVLayerGroupsManager
    are out of scope).
    """
    model_name: str
    world_size: int = 1
    worker_id: int = 0
    kv_dtype: torch.dtype = torch.float16
    chunk_size: int = 256
    use_mla: bool = False  # always False here


# -----------------------------------------------------------------------------
# Mirror of lmcache/utils.py:CacheEngineKey + LayerCacheEngineKey (subset).
# `slots=True` matches the LMCache dataclass. The serializer / from_string
# helpers are dropped — we never round-trip these keys to disk.
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class CacheEngineKey:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype: torch.dtype
    request_configs: Optional[dict] = None

    def __hash__(self):
        return hash((
            self.model_name, self.world_size, self.worker_id,
            self.chunk_hash, str(self.dtype),
        ))

    def __eq__(self, other):
        if not isinstance(other, CacheEngineKey):
            return False
        return (
            self.model_name == other.model_name
            and self.world_size == other.world_size
            and self.worker_id == other.worker_id
            and self.chunk_hash == other.chunk_hash
            and self.dtype == other.dtype
        )

    def split_layers(self, num_layers: int) -> list["LayerCacheEngineKey"]:
        return [
            LayerCacheEngineKey(
                model_name=self.model_name,
                world_size=self.world_size,
                worker_id=self.worker_id,
                chunk_hash=self.chunk_hash,
                dtype=self.dtype,
                request_configs=self.request_configs,
                layer_id=i,
            )
            for i in range(num_layers)
        ]


@dataclass(slots=True)
class LayerCacheEngineKey(CacheEngineKey):
    layer_id: int = 0

    def __hash__(self):
        return hash((
            self.model_name, self.world_size, self.worker_id,
            self.chunk_hash, str(self.dtype), self.layer_id,
        ))

    def __eq__(self, other):
        if not isinstance(other, LayerCacheEngineKey):
            return False
        # CacheEngineKey.__eq__(self, other) explicitly — Python 3.12.x
        # slotted-dataclass inheritance can mis-resolve `super()` inside
        # `__eq__` and raise
        #   TypeError: super(type, obj): obj must be an instance or subtype of type.
        return CacheEngineKey.__eq__(self, other) and self.layer_id == other.layer_id


# -----------------------------------------------------------------------------
# MemoryObj — pairs `.tensor` (the actual KV blob) with `.metadata`
# (format + cached_positions). The HF port uses tensors on CPU.
# -----------------------------------------------------------------------------
@dataclass
class MemoryObjMetadata:
    fmt: MemoryFormat = MemoryFormat.KV_2TD
    cached_positions: Optional[torch.Tensor] = None


@dataclass
class MemoryObj:
    """
    Single-layer KV blob for one chunk. `tensor.shape == (2, chunk_len,
    hidden_dim_kv)` per LMCache's KV_2TD layout. `tensor[0]` is K
    (post-RoPE at cached positions), `tensor[1]` is V.
    """
    tensor: torch.Tensor
    metadata: MemoryObjMetadata = field(default_factory=MemoryObjMetadata)


# -----------------------------------------------------------------------------
# LocalCPUBackend — just an in-memory dict keyed by LayerCacheEngineKey.
# Mirrors the smallest possible subset of LMCache's storage manager.
# -----------------------------------------------------------------------------
class LocalCPUBackend:
    def __init__(self):
        self._store: dict[LayerCacheEngineKey, MemoryObj] = {}

    def contains(self, key: LayerCacheEngineKey) -> bool:
        return key in self._store

    def put(self, key: LayerCacheEngineKey, mem_obj: MemoryObj) -> None:
        self._store[key] = mem_obj

    def get(self, key: LayerCacheEngineKey) -> Optional[MemoryObj]:
        return self._store.get(key)

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
