"""
Phase 2 — equivalence of layerwise prefill (Phase 1's compute_layer +
stub blender) against stock HF `LlamaForCausalLM.forward` for both
target models in fp32 / fp16 / bf16. Tolerances per
docs/VERIFICATION_PROTOCOL.md §2.

Build order is critical (see reports/phase1.md §3 "부수 효과 / 주의사항"):

  1. Load HF model.
  2. Run stock forward with `use_cache=True, output_hidden_states=True`
     and snapshot per-layer K / V from the returned Cache, plus the
     final hidden state (= post-norm).
  3. *Then* build the LMC adapter via `infer_model_from_hf`. This step
     patches the hf_model in place (qkv_proj, rotary_emb, o_proj,
     RMSNorms), so any stock forward after this point is wrong.
  4. Drive `compute_layer` with the RecordingBlender proxy, captured
     per-layer K / V (post-RoPE, pre-attention) and the post-MLP final
     hidden state.
  5. Apply `model.model.norm` to the layerwise final state because
     `compute_layer` does NOT run model.norm; stock does (its return
     is `BaseModelOutputWithPast(last_hidden_state=model.norm(...))`).
  6. Reshape captures into a common layout and `torch.allclose` per
     (model, dtype) tolerance.

Real-model tests gate on CUDA + `LMC_PHASE2_REAL_MODELS=1`; tiny-model
tests run on CPU regardless. Mistral and Llama-3.1 must run in
*separate pytest processes* on a 24 GB GPU — see
`scripts/run_phase2_remote.sh`.
"""
from __future__ import annotations

import gc
import os
from typing import Any

import pytest
import torch

from lmc.compute.blend.blender import LMCBlender
from lmc.compute.models.utils import infer_model_from_hf


# ---------------------------------------------------------------------------
# Recording proxy. Captures Phase-1-stub's post-RoPE (q, k, v) for each
# layer so Phase 2 can compare against stock's per-layer Cache entries.
# Mirrors the Phase 1 RecordingBlender (which only tracked shapes and
# k-changed flag); this one keeps actual tensor clones.
# ---------------------------------------------------------------------------
class RecordingBlender:
    def __init__(self, inner: LMCBlender):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "captured", [])
        self.vllm_model = inner.vllm_model
        self.num_layers = inner.num_layers
        self.metadata = inner.metadata
        self.common_metadata = inner.common_metadata

    def __setattr__(self, name, value):
        # LMCBaseModel.__init__ runs `blender.layerwise_model = self`;
        # forward that into the inner stub so its process_qkv sees it.
        if name == "layerwise_model" and hasattr(self, "_inner"):
            self._inner.layerwise_model = value
        object.__setattr__(self, name, value)

    def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
        q_out, k_out, v_out, residual_out, attn_output_out, attn_metadata_out = (
            self._inner.process_qkv(q, k, v, residual, layer_id, attn_output, attn_metadata)
        )
        # Clone to defend against later in-place mutation (compute_layer
        # reshapes K, V multiple times; views are cheap but reshape on
        # non-contiguous can copy).
        self.captured.append({
            "layer_id": layer_id,
            "k_post": k_out.detach().clone(),   # 2-D: (seq, num_kv_heads * head_dim)
            "v_post": v_out.detach().clone(),   # same shape; never RoPEd
        })
        return q_out, k_out, v_out, residual_out, attn_output_out, attn_metadata_out


# ---------------------------------------------------------------------------
# Cache extraction helper (feature-detects across transformers versions).
# transformers 5.x → DynamicCache.layers[i].keys / .values
# transformers 4.36-4.45 → DynamicCache.key_cache[i] / .value_cache[i]
# legacy (<=4.35) → tuple of (K, V) tuples
# ---------------------------------------------------------------------------
def extract_kv(cache: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    return cache[layer_idx][0], cache[layer_idx][1]


# ---------------------------------------------------------------------------
# Reshape helpers.
# - LMC layerwise K, V are 2-D: (seq_len, num_kv_heads * head_dim).
# - Stock K, V are 4-D:         (batch=1, num_kv_heads, seq_len, head_dim).
# Convert LMC 2-D to a (1, num_kv_heads, seq_len, head_dim) view to
# match stock. The flat dimension layout is identical (HF builds K via
# `.view(*, num_kv_heads, head_dim).transpose(1, 2)`, which has the
# same row-major contiguity as our `.view(seq, num_kv_heads, head_dim)`).
# ---------------------------------------------------------------------------
def lmc_2d_to_stock_4d(t: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    seq_len = t.shape[0]
    return t.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0).contiguous()


# Tolerances (docs/VERIFICATION_PROTOCOL.md §2).
TOLERANCES = {
    torch.float32:  (1e-5, 1e-5),
    torch.float16:  (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


def _stock_then_layerwise(model, input_ids: torch.Tensor, dtype: torch.dtype):
    """
    Run stock forward (snapshot K/V + last hidden), then build the LMC
    adapter and drive compute_layer with a RecordingBlender. Returns
    a dict with stock + layerwise captures, ready for comparison.
    """
    cfg = model.config
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    # ---- Stock forward (must run BEFORE LMC patches the model) ----
    with torch.no_grad():
        stock_out = model(
            input_ids=input_ids.unsqueeze(0),  # add batch dim
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    stock_last_hidden = stock_out.hidden_states[-1].detach().clone()  # post-norm
    stock_cache = stock_out.past_key_values
    stock_kv = []
    for i in range(cfg.num_hidden_layers):
        k, v = extract_kv(stock_cache, i)
        stock_kv.append((k.detach().clone(), v.detach().clone()))

    # ---- Build LMC adapter (patches model in place) ----
    stub = LMCBlender(model, num_layers=cfg.num_hidden_layers)
    recorder = RecordingBlender(stub)
    lmc_model = infer_model_from_hf(model, recorder)

    # ---- Capture the layerwise "final hidden state" ----
    # LMCache's `compute_layer` matches LMCache's own design where the
    # MLP's residual add is deferred to the *next* layer's fused
    # input-layernorm. After the last yield, the locals are:
    #   hidden_states = mlp(post_attention_norm_out_{N-1})       (no residual)
    #   residual      = o_proj_out_{N-1} + running_sum_{N-2}
    # HF's last-layer output is `hidden_states + residual` (then norm).
    # So we hook both: the last MLP's output, and the post-attention
    # layernorm's *residual* slot (the second tuple element of our
    # _ResidualRMSNorm wrapper). Sum them, then apply model.norm.
    captured_final: dict[str, torch.Tensor] = {}
    last_layer = model.model.layers[-1]

    def mlp_hook(_mod, _inp, out):
        captured_final["mlp_out"] = out.detach()

    def post_norm_hook(_mod, _inp, out):
        # _ResidualRMSNorm returns (rmsnorm(x + residual), x + residual).
        # The second element is the "residual" carried to the next layer
        # (or to the implicit post-MLP add for the final layer).
        if isinstance(out, tuple):
            captured_final["residual_after_post"] = out[1].detach()
        else:
            # Should not happen — last layer post_attention_layernorm is
            # always called with a residual inside compute_layer.
            raise RuntimeError("post_attention_layernorm returned non-tuple")

    h_mlp = last_layer.mlp.register_forward_hook(mlp_hook)
    h_pal = last_layer.post_attention_layernorm.register_forward_hook(post_norm_hook)

    try:
        gen = lmc_model.compute_layer(input_ids)
        for _ in range(lmc_model.num_layers):
            next(gen)
    finally:
        h_mlp.remove()
        h_pal.remove()

    final_layerwise_2d = (
        captured_final["mlp_out"] + captured_final["residual_after_post"]
    )  # (seq, hidden_size) — equivalent to HF's last-layer output, pre-norm
    layerwise_post_norm = model.model.norm(final_layerwise_2d).unsqueeze(0)

    return {
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "stock_kv": stock_kv,
        "stock_last_hidden": stock_last_hidden,
        "layerwise_kv": recorder.captured,
        "layerwise_last_hidden": layerwise_post_norm,
        "lmc_model": lmc_model,
        "recorder": recorder,
    }


def _compare_and_report(captures, dtype, label, max_layer_log: int = 5):
    """
    Run allclose checks. Prints per-layer max-abs-error summary.
    Returns a dict of measurements; the test asserts at the end.
    """
    rtol, atol = TOLERANCES[dtype]
    num_kv_heads = captures["num_kv_heads"]
    head_dim = captures["head_dim"]

    per_layer = []
    failed_layers_k = []
    failed_layers_v = []

    for i, ((stk_k, stk_v), lmc) in enumerate(zip(captures["stock_kv"], captures["layerwise_kv"], strict=False)):
        lmc_k_4d = lmc_2d_to_stock_4d(lmc["k_post"], num_kv_heads, head_dim)
        lmc_v_4d = lmc_2d_to_stock_4d(lmc["v_post"], num_kv_heads, head_dim)

        # Reshape stock to same dtype as the live tensors for a fair compare.
        stk_k = stk_k.to(lmc_k_4d.dtype)
        stk_v = stk_v.to(lmc_v_4d.dtype)

        k_diff = (stk_k - lmc_k_4d).abs().max().item()
        v_diff = (stk_v - lmc_v_4d).abs().max().item()
        per_layer.append((i, k_diff, v_diff))

        k_ok = torch.allclose(stk_k, lmc_k_4d, rtol=rtol, atol=atol)
        v_ok = torch.allclose(stk_v, lmc_v_4d, rtol=rtol, atol=atol)
        if not k_ok:
            failed_layers_k.append(i)
        if not v_ok:
            failed_layers_v.append(i)

    # Hidden state.
    stk_h = captures["stock_last_hidden"].to(captures["layerwise_last_hidden"].dtype)
    lw_h = captures["layerwise_last_hidden"]
    h_diff = (stk_h - lw_h).abs().max().item()
    h_ok = torch.allclose(stk_h, lw_h, rtol=rtol, atol=atol)

    # Log a short per-layer summary (worst layers first).
    worst = sorted(per_layer, key=lambda x: -max(x[1], x[2]))[:max_layer_log]
    print(f"\n[Phase2 {label} dtype={dtype}] per-layer worst-case max-abs-err:")
    for i, dk, dv in worst:
        print(f"  layer {i:>3}: K={dk:.3e}  V={dv:.3e}")
    print(f"  final hidden (post-norm) max-abs-err: {h_diff:.3e}  (tol={atol})")

    return {
        "per_layer": per_layer,
        "h_diff": h_diff,
        "k_ok_all": not failed_layers_k,
        "v_ok_all": not failed_layers_v,
        "h_ok": h_ok,
        "failed_layers_k": failed_layers_k,
        "failed_layers_v": failed_layers_v,
    }


# ---------------------------------------------------------------------------
# Tiny-model regression. CPU + 2-layer Llama, fp32 only (fp16/bf16 are
# numerically too noisy at this tiny scale to be meaningful, but the
# kernel-path coverage is what we want here).
# ---------------------------------------------------------------------------
def _build_tiny_llama(dtype: torch.dtype = torch.float32):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(dtype=dtype)
    model.eval()
    return model


class TestTinyModelEquivalence:
    """
    CPU 2-layer Llama equivalence. fp32 only — small models are too
    noisy for fp16/bf16 to be a meaningful regression signal.
    """

    @pytest.fixture(scope="class")
    def tiny_captures(self):
        model = _build_tiny_llama(dtype=torch.float32)
        torch.manual_seed(1234)
        ids = torch.randint(0, model.config.vocab_size, (16,))
        return _stock_then_layerwise(model, ids, torch.float32)

    def test_kv_per_layer_allclose(self, tiny_captures):
        result = _compare_and_report(tiny_captures, torch.float32, "tiny")
        assert result["k_ok_all"], f"K mismatch on layers {result['failed_layers_k']}"
        assert result["v_ok_all"], f"V mismatch on layers {result['failed_layers_v']}"

    def test_final_hidden_allclose(self, tiny_captures):
        result = _compare_and_report(tiny_captures, torch.float32, "tiny")
        assert result["h_ok"], f"final hidden mismatch (max abs err {result['h_diff']:.3e})"


# ---------------------------------------------------------------------------
# Real-model equivalence. CUDA + 24 GB VRAM required + env gate so the
# dev box doesn't try to pull ~30 GB of gated weights.
# ---------------------------------------------------------------------------
REAL_MODELS_ENABLED = os.environ.get("LMC_PHASE2_REAL_MODELS") == "1"
CUDA_AVAILABLE = torch.cuda.is_available()
REAL_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]
# fp32 Mistral-7B is ~29 GB; fp32 Llama-3.1-8B is ~32 GB. Neither fits
# on a 24 GB GPU, so real-model fp32 is intentionally excluded here.
# docs/VERIFICATION_PROTOCOL.md §2 will be updated to reflect this; the
# fp32 numerical tolerance (1e-5) is still exercised by the CPU
# TestTinyModelEquivalence above. The bigger models stay at the
# meaningful inference dtypes (fp16, bf16).
DTYPES_REAL = [torch.float16, torch.bfloat16]


@pytest.mark.skipif(
    not (CUDA_AVAILABLE and REAL_MODELS_ENABLED),
    reason=(
        "Real-model Phase 2 tests need CUDA + LMC_PHASE2_REAL_MODELS=1. "
        "Use scripts/run_phase2_remote.sh on a 24 GB GPU."
    ),
)
class TestRealModelEquivalence:
    """
    Parametrized over (model_name, dtype). Each (model, dtype) load is
    its own fixture invocation; reload between dtypes is acceptable
    since the model weights fit on a 24 GB GPU only one at a time.

    The scripts/run_phase2_remote.sh wrapper splits Mistral and Llama
    across separate pytest processes to avoid the 24 GB OOM that
    Phase 1 hit.
    """

    # Parametrize at fixture level (pytest 9 rejects function-scoped
    # @parametrize on a class-scoped fixture — Phase 1 fix #4b3a751).
    @pytest.fixture(
        scope="function",
        params=[(m, d) for m in REAL_MODELS for d in DTYPES_REAL],
        ids=lambda p: f"{p[0].rsplit('/', 1)[-1]}-{str(p[1]).split('.')[-1]}",
    )
    def real_captures(self, request):
        model_name, dtype = request.param
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager",
        ).to("cuda:0")
        model.eval()

        torch.manual_seed(1234)
        # 128-token prompt per docs/VERIFICATION_PROTOCOL.md §2.
        ids = torch.randint(0, model.config.vocab_size, (128,), device="cuda:0")
        captures = _stock_then_layerwise(model, ids, dtype)
        captures["model_name"] = model_name
        captures["dtype"] = dtype

        yield captures

        # Aggressive cleanup — function-scope keeps each fixture
        # invocation isolated so an OOM in one (model, dtype) cell
        # doesn't poison the next.
        try:
            model.to("cpu")
        except Exception:
            pass
        recorder = captures.get("recorder")
        if recorder is not None:
            recorder.vllm_model = None
            if getattr(recorder, "_inner", None) is not None:
                recorder._inner.vllm_model = None
                recorder._inner.layerwise_model = None
            recorder.layerwise_model = None
        lmc = captures.get("lmc_model")
        if lmc is not None:
            lmc.vllm_model = None
            lmc.blender = None
        captures.clear()
        del captures, recorder, lmc, model
        gc.collect()
        torch.cuda.empty_cache()

    def test_kv_per_layer_allclose(self, real_captures):
        dtype = real_captures["dtype"]
        result = _compare_and_report(real_captures, dtype, real_captures["model_name"])
        assert result["k_ok_all"], (
            f"K mismatch on layers {result['failed_layers_k']}; "
            f"dtype={dtype}, model={real_captures['model_name']}"
        )
        assert result["v_ok_all"], (
            f"V mismatch on layers {result['failed_layers_v']}; "
            f"dtype={dtype}, model={real_captures['model_name']}"
        )

    def test_final_hidden_allclose(self, real_captures):
        dtype = real_captures["dtype"]
        result = _compare_and_report(real_captures, dtype, real_captures["model_name"])
        assert result["h_ok"], (
            f"final hidden mismatch (max abs err {result['h_diff']:.3e}); "
            f"dtype={dtype}, model={real_captures['model_name']}"
        )
