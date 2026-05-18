# Port of lmcache/v1/compute/models/base.py. compute_layer is the
# layerwise prefill generator; its body mirrors the source line-for-line
# with two deliberate HF adaptations:
#   1. @torch.compile is dropped (docs/CODING_CONVENTIONS.md §"Code style").
#   2. `model.layers[start_layer:end_layer]` becomes `model.layers`
#      since HF has no vLLM pipeline-parallel slice.
#
# Every HF-API gap is bridged by an adapter installed in __init__ so the
# call sites inside compute_layer read identically to LMCache. See
# docs/CODING_CONVENTIONS.md §"HF-required adaptations" for the full
# list and rationale.
from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from lmc.compute.attention.utils import infer_attn_backend_from_hf


def _make_residual_rmsnorm(rmsnorm: nn.Module):
    """
    Wrap HF's unary RMSNorm so it accepts the vLLM fused-residual
    signature. Backward-compatible: called with one arg behaves like HF.

      __call__(x)            -> rmsnorm(x)                  # HF semantics
      __call__(x, residual)  -> (rmsnorm(x + residual),
                                 x + residual)              # vLLM semantics
    """
    class _ResidualRMSNorm(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, residual=None):
            if residual is None:
                return self.inner(hidden_states)
            new_residual = hidden_states + residual
            return self.inner(new_residual), new_residual

    return _ResidualRMSNorm(rmsnorm)


def _make_qkv_proj(self_attn: nn.Module):
    """
    Synthesize a fused `qkv_proj(hidden_states) -> (qkv_concat, None)`
    callable that runs HF's separate q_proj/k_proj/v_proj and
    concatenates on the last dim. Matches vLLM's `qkv_proj` return
    convention so compute_layer's `qkv, _ = layer.self_attn.qkv_proj(...)`
    stays identical to the LMCache call site.
    """
    q_proj, k_proj, v_proj = self_attn.q_proj, self_attn.k_proj, self_attn.v_proj

    def qkv_proj(hidden_states):
        q = q_proj(hidden_states)
        k = k_proj(hidden_states)
        v = v_proj(hidden_states)
        return torch.cat([q, k, v], dim=-1), None

    return qkv_proj


def _make_rotary_emb_wrapper(hf_model, num_heads: int, num_kv_heads: int, head_size: int):
    """
    Build a callable with the vLLM-style signature
    `rotary_emb(positions, q, k) -> (q_rot, k_rot)` that delegates to
    HF's `LlamaRotaryEmbedding` (which lives on `hf_model.model`, not
    on the attention layer — see modeling_llama.py:366).

    Inputs at process_qkv time are 2-D:
        q: (seq_len, num_heads    * head_size)
        k: (seq_len, num_kv_heads * head_size)
    The wrapper reshapes to (1, num_heads_or_kv, seq_len, head_size) for
    apply_rotary_pos_emb (unsqueeze_dim=1 default), then reshapes back.
    """
    rotary_emb = hf_model.model.rotary_emb

    def wrapper(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.shape[0]
        # (seq_len, H*D) -> (1, seq_len, H, D) -> (1, H, seq_len, D)
        q_4d = q.view(seq_len, num_heads,    head_size).unsqueeze(0).transpose(1, 2)
        k_4d = k.view(seq_len, num_kv_heads, head_size).unsqueeze(0).transpose(1, 2)
        # HF's rotary_emb returns (cos, sin) of shape (1, seq_len, head_dim).
        # It only reads positions.shape; pass the 2-D batch form it expects.
        position_ids = positions.unsqueeze(0)  # (1, seq_len)
        cos, sin = rotary_emb(q_4d, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q_4d, k_4d, cos, sin)
        # (1, H, seq_len, D) -> (seq_len, H, D) -> (seq_len, H*D)
        q_out = q_rot.transpose(1, 2).reshape(seq_len, num_heads    * head_size)
        k_out = k_rot.transpose(1, 2).reshape(seq_len, num_kv_heads * head_size)
        return q_out, k_out

    return wrapper


def _make_o_proj_tupled(self_attn: nn.Module):
    """
    Wrap HF's plain o_proj Linear so it returns `(out, None)`. Must be
    an nn.Module: replacing an existing child Module with a plain
    function is rejected by nn.Module.__setattr__. Wrapping the inner
    Linear as a registered child also keeps its parameters discoverable
    by `.to(device)` / state_dict / etc.
    """
    class _OProjTupled(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x), None

    return _OProjTupled(self_attn.o_proj)


# Sentinel attribute name used to detect (and skip) double-patching.
_PATCH_MARKER = "_lmc_patched"


class LMCBaseModel(nn.Module, ABC):
    """
    Port of lmcache/v1/compute/models/base.py:LMCBaseModel.

    Constructor signature replaces `vllm_model` with `hf_model` but
    exposes it as `self.vllm_model` so `process_qkv`'s line
    `self.layerwise_model.vllm_model.model.layers[layer_id]` and
    compute_layer's `self.vllm_model.model.embed_input_ids(...)` stay
    identical to the LMCache source.
    """

    def __init__(self, hf_model: nn.Module, blender, enable_sparse: bool = False):
        super().__init__()
        # The one rename: vllm_model -> hf_model in the parameter, but
        # the attribute name matches LMCache verbatim.
        self.vllm_model = hf_model

        self.num_layers = len(hf_model.model.layers)

        # Build per-layer holders (mirrors LMCache's
        # self.vllm_attn_layers / self.lmc_attn_layers split in
        # compute/models/base.py:29-37).
        self.vllm_attn_layers = []
        self.lmc_attn_layers = []

        config = hf_model.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_size = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)
        q_size = num_heads * head_size
        kv_size = num_kv_heads * head_size

        for i, layer in enumerate(hf_model.model.layers):
            self.lmc_attn_layers.append(
                infer_attn_backend_from_hf(layer.self_attn, enable_sparse)
            )
            # Mirror vllm_attn_layers[idx].num_heads / num_kv_heads /
            # head_size reads inside compute_layer.
            self.vllm_attn_layers.append(SimpleNamespace(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            ))

        # Wire the blender (LMCache does the same at base.py:41).
        self.blender = blender
        blender.layerwise_model = self

        # Phase 3 fills this via get_fused_rope(...).
        self.fused_rotary_emb = None

        # Apply HF-required adaptation patches. Marker prevents
        # double-patching if LMCBaseModel is built twice on the same
        # hf_model (e.g. across pytest fixture invocations).
        if not getattr(hf_model, _PATCH_MARKER, False):
            for layer in hf_model.model.layers:
                # qkv_proj: fused QKV synthesized from q/k/v_proj.
                layer.self_attn.qkv_proj = _make_qkv_proj(layer.self_attn)
                # rotary_emb: vLLM-style (positions, q, k) -> (q_rot, k_rot).
                layer.self_attn.rotary_emb = _make_rotary_emb_wrapper(
                    hf_model, num_heads, num_kv_heads, head_size,
                )
                # o_proj: wrap to return (out, None).
                layer.self_attn.o_proj = _make_o_proj_tupled(layer.self_attn)
                # q_size / kv_size for the split call inside compute_layer.
                layer.self_attn.q_size = q_size
                layer.self_attn.kv_size = kv_size
                # Fused-residual RMSNorm wrappers.
                layer.input_layernorm = _make_residual_rmsnorm(layer.input_layernorm)
                layer.post_attention_layernorm = _make_residual_rmsnorm(
                    layer.post_attention_layernorm
                )

            # Alias so compute_layer's
            # `self.vllm_model.model.embed_input_ids(input_ids)` works.
            hf_model.model.embed_input_ids = hf_model.model.embed_tokens
            setattr(hf_model, _PATCH_MARKER, True)

    @abstractmethod
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors. Model-specific implementation."""

    def compute_layer(self, input_ids: torch.Tensor):
        """
        Layerwise prefill generator. Mirrors
        lmcache/v1/compute/models/base.py:67-142 verbatim except:
          - no @torch.compile (deliberate, see CODING_CONVENTIONS.md);
          - `model.layers[start:end]` -> `model.layers` (HF has no PP).
        Yields exactly self.num_layers times.
        """
        # input_ids may arrive on CPU; move to the model's device. The
        # source does .cuda(); we use .to(device) to stay portable.
        device = next(self.vllm_model.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)

        hidden_states = self.vllm_model.model.embed_input_ids(input_ids)
        residual = None

        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )

        for idx, layer in enumerate(self.vllm_model.model.layers):
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)

            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            # Model-specific QKV processing
            q, k, v = self._process_qkv(q, k, v, layer)

            q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                q, k, v, residual, idx, attn_output, attn_metadata
            )

            num_heads    = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size    = self.vllm_attn_layers[idx].head_size

            q           = q.view(-1, num_heads,    head_size)
            k           = k.view(-1, num_kv_heads, head_size)
            v           = v.view(-1, num_kv_heads, head_size)
            attn_output = attn_output.view(-1, num_heads, head_size)

            attn_output = self.lmc_attn_layers[idx].forward_contiguous(
                q, k, v, attn_output, attn_metadata
            )

            attn_output = attn_output.view(-1, num_heads    * head_size)
            k           = k.view(-1, num_kv_heads * head_size)
            v           = v.view(-1, num_kv_heads * head_size)

            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            yield
