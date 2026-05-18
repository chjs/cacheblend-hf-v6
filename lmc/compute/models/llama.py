# Port of lmcache/v1/compute/models/llama.py. The _process_qkv hook is
# an identity for Llama / Mistral (no q_norm / k_norm); Qwen3 is the
# only LMCache-supported model that overrides it, and Qwen3 is out of
# scope for this port.
from lmc.compute.models.base import LMCBaseModel


class LMCLlamaModel(LMCBaseModel):
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors for LLaMa model (no additional processing)."""
        return q, k, v
