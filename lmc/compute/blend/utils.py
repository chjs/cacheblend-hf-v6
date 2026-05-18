# Mirror of lmcache/v1/compute/blend/utils.py:LMCBlenderBuilder.
# The vLLM version stores per-instance blenders keyed by an
# `instance_id`; the HF port keeps the same shape so users who already
# write `LMCBlenderBuilder.get_or_create(ENGINE_NAME, ...)` get a
# drop-in.
from __future__ import annotations

from typing import Dict

from lmc.cache_engine import LMCacheEngine
from lmc.compute.blend.blender import LMCBlender
from lmc.config import LMCacheEngineConfig
from lmc.gpu_connector import HFBufferLayerwiseGPUConnector
from lmc.integration.hf.utils import HFModelTracker


class LMCBlenderBuilder:
    """
    Per-instance blender cache. Mirrors LMCache's class shape.
    """

    _blenders: Dict[str, LMCBlender] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        cache_engine: LMCacheEngine,
        gpu_connector: HFBufferLayerwiseGPUConnector,
        config: LMCacheEngineConfig,
    ) -> LMCBlender:
        if instance_id not in cls._blenders:
            hf_model = HFModelTracker.get_model(instance_id)
            blender = LMCBlender(
                cache_engine=cache_engine,
                gpu_connector=gpu_connector,
                hf_model=hf_model,
                config=config,
            )
            cls._blenders[instance_id] = blender
        return cls._blenders[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> LMCBlender:
        if instance_id not in cls._blenders:
            raise ValueError(f"Blender for {instance_id} not found.")
        return cls._blenders[instance_id]

    @classmethod
    def reset(cls) -> None:
        """Test helper — drop all cached blenders."""
        cls._blenders.clear()
