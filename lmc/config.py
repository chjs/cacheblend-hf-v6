# Minimal port of lmcache/v1/config.py. Only the blend_* and chunking
# fields the HF port actually uses; other LMCache knobs (P2P, remote
# storage, controller, etc.) are dropped. Mirrors the field-name table
# in docs/LMCACHE_IMPLEMENTATION.md §2.9.
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _env_to_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _env_to_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _env_to_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y", "on")


@dataclass
class LMCacheEngineConfig:
    # Mirrors lmcache/v1/config.py:_CONFIG_DEFINITIONS for the keys the
    # HF port reads at runtime.
    chunk_size: int = 256
    use_layerwise: bool = True

    enable_blending: bool = False
    blend_special_str: str = " # # "
    blend_check_layers: Optional[List[int]] = None
    blend_recompute_ratios: Optional[List[float]] = None
    blend_thresholds: Optional[List[float]] = None
    blend_min_tokens: int = 256

    pre_caching_hash_algorithm: str = "builtin"
    extra_config: Optional[dict] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LMCacheEngineConfig":
        """
        Read `LMCACHE_*` env vars matching the names LMCache itself uses
        (see docs/LMCACHE_IMPLEMENTATION.md §2.9). Unknown vars are
        ignored; unset vars keep the dataclass defaults.
        """
        kwargs: dict = {}
        if v := os.environ.get("LMCACHE_CHUNK_SIZE"):
            kwargs["chunk_size"] = int(v)
        if v := os.environ.get("LMCACHE_USE_LAYERWISE"):
            kwargs["use_layerwise"] = _env_to_bool(v)
        if v := os.environ.get("LMCACHE_ENABLE_BLENDING"):
            kwargs["enable_blending"] = _env_to_bool(v)
        if v := os.environ.get("LMCACHE_BLEND_SPECIAL_STR"):
            kwargs["blend_special_str"] = v
        if v := os.environ.get("LMCACHE_BLEND_CHECK_LAYERS"):
            kwargs["blend_check_layers"] = _env_to_int_list(v)
        if v := os.environ.get("LMCACHE_BLEND_RECOMPUTE_RATIOS"):
            kwargs["blend_recompute_ratios"] = _env_to_float_list(v)
        if v := os.environ.get("LMCACHE_BLEND_THRESHOLDS"):
            kwargs["blend_thresholds"] = _env_to_float_list(v)
        if v := os.environ.get("LMCACHE_BLEND_MIN_TOKENS"):
            kwargs["blend_min_tokens"] = int(v)
        if v := os.environ.get("LMCACHE_PRE_CACHING_HASH_ALGORITHM"):
            kwargs["pre_caching_hash_algorithm"] = v
        return cls(**kwargs)

    def get_extra_config_value(self, key: str, default=None):
        """Mirror of the LMCache helper used in token_database.py."""
        if self.extra_config is None:
            return default
        return self.extra_config.get(key, default)
