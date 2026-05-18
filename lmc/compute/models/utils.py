# Parallel to lmcache/v1/compute/models/utils.py.
# `infer_model_from_hf` mirrors `infer_model_from_vllm` (line 14): it
# dispatches on the HF class name. For Phase 1 only Llama / Mistral are
# supported (they share the LMCLlamaModel adapter).
from lmc.compute.models.llama import LMCLlamaModel


def infer_model_from_hf(hf_model, blender, enable_sparse: bool = False):
    model_name = type(hf_model).__name__
    if model_name in ("LlamaForCausalLM", "MistralForCausalLM"):
        return LMCLlamaModel(hf_model, blender, enable_sparse)
    # Qwen2 / Qwen3 are LMCache-supported but out of scope here.
    raise NotImplementedError(
        f"Model type {model_name} is not supported in the HF port."
    )
