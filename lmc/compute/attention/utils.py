# Parallel to lmcache/v1/compute/attention/utils.py.
# LMCache's `infer_attn_backend_from_vllm` (line 9) inspects vllm_attn.impl
# and returns LMCFlashAttnBackend / LMCFlashInferSparseBackend. The HF port
# only supports eager attention, so this is a single-branch factory whose
# call shape mirrors the reference.
from transformers.models.llama.modeling_llama import LlamaAttention as _LlamaAttention

from lmc.compute.attention.eager import LMCEagerAttnBackend


def infer_attn_backend_from_hf(hf_attn, enable_sparse: bool = False):
    """
    Build an attention backend for one HF attention module. Mirrors
    `infer_attn_backend_from_vllm(vllm_attn, enable_sparse)` at
    `lmcache/v1/compute/attention/utils.py:9-23`.

    enable_sparse is accepted for signature parity but only the eager
    backend exists in this port.
    """
    if enable_sparse:
        raise NotImplementedError(
            "Sparse attention is not part of the HF port; eager backend only."
        )
    config = hf_attn.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)
    # HF's LlamaAttention uses head_dim**-0.5 (modeling_llama.py:234).
    scaling = head_size ** -0.5
    return LMCEagerAttnBackend(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        scaling=scaling,
    )
