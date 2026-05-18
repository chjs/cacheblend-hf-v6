# Parallel to lmcache/integration/vllm/utils.py.
# ENGINE_NAME differs from LMCache's "vllm-instance" so the two trackers
# don't collide if both packages are imported into the same process
# (see docs/phases/PHASE3_PROMPT.md item 10).
from typing import Dict

from torch import nn

ENGINE_NAME = "hf_cacheblend"


class HFModelTracker:
    """Mirror of lmcache/v1/compute/models/utils.py:VLLMModelTracker."""
    _hf_models: Dict[str, nn.Module] = {}

    @classmethod
    def register_model(cls, instance_id: str, hf_model: nn.Module) -> None:
        if instance_id not in cls._hf_models:
            cls._hf_models[instance_id] = hf_model
        # else: the LMCache reference warns and silently keeps the
        # original. We follow that semantic.

    @classmethod
    def get_model(cls, instance_id: str) -> nn.Module:
        if instance_id not in cls._hf_models:
            raise ValueError(f"hf model for {instance_id} not found.")
        return cls._hf_models[instance_id]
