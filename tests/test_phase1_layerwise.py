"""
Phase 1 verification — covers the 5 criteria in
docs/VERIFICATION_PROTOCOL.md §1.

Tests are split into two tiers:
  - tiny_model: a synthetic LlamaConfig with 2 layers, runnable on CPU.
    Catches integration bugs (call-site shape, generator yield count,
    stub blender wiring) without needing CUDA or downloading weights.
  - real_model: Mistral-7B-Instruct-v0.2 + Meta-Llama-3.1-8B-Instruct,
    gated on CUDA availability + a sentinel env var so the smoke run
    on the dev box doesn't try to pull ~30 GB of weights.

To enable the real-model tests on the GPU box:
    LMC_PHASE1_REAL_MODELS=1 pytest tests/test_phase1_layerwise.py -v
"""
from __future__ import annotations

import gc
import os
from typing import Optional

import pytest
import torch

# Phase 1 / Phase 2 use the stub blender, not the full Phase 3 LMCBlender:
# the layerwise-skeleton tests run before there is any cache engine /
# GPU connector to construct a full blender. See
# `lmc/compute/blend/stub_blender.py` (introduced in Phase 5 to undo a
# Phase 3 regression — Phase 3's full LMCBlender swap broke these
# Phase 1 / Phase 2 tests at the import-time keyword `num_layers`).
from lmc.compute.blend.stub_blender import LMCStubBlender as LMCBlender
from lmc.compute.blend.metadata import LMCBlendMetadata
from lmc.compute.models.utils import infer_model_from_hf


# ---------------------------------------------------------------------------
# Recording proxy around the stub blender.
#
# Captures the (q, k, v) seen BEFORE and the (q, k) returned AFTER each
# call to process_qkv. Criterion #5 needs both to verify RoPE happened
# (post.k != pre.k for positions > 0).
# ---------------------------------------------------------------------------
class RecordingBlender:
    def __init__(self, inner: LMCBlender):
        # __setattr__ below forwards mutations to the inner stub for
        # layerwise_model so process_qkv works whichever object the
        # caller writes to. Avoid recursion by using object.__setattr__
        # to install _inner / calls.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "calls", [])
        # Forward attributes so compute_layer sees an LMCBlender-shaped
        # object.
        self.vllm_model = inner.vllm_model
        self.num_layers = inner.num_layers
        self.metadata = inner.metadata
        self.common_metadata = inner.common_metadata

    def __setattr__(self, name, value):
        # LMCBaseModel.__init__ runs `blender.layerwise_model = self`.
        # The inner stub's process_qkv reads
        # `self.layerwise_model.vllm_model...`, so the write must reach
        # the inner stub as well.
        if name == "layerwise_model" and hasattr(self, "_inner"):
            self._inner.layerwise_model = value
        object.__setattr__(self, name, value)

    def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
        pre = {
            "layer_id": layer_id,
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "v_shape": tuple(v.shape),
            "residual_shape": tuple(residual.shape) if residual is not None else None,
            "attn_output_is_none": attn_output is None,
            "k_pre": k.detach().clone(),
        }
        q_out, k_out, v_out, residual_out, attn_output_out, attn_metadata_out = (
            self._inner.process_qkv(q, k, v, residual, layer_id, attn_output, attn_metadata)
        )
        pre["k_post"] = k_out.detach().clone()
        pre["k_changed_by_rope"] = not torch.equal(pre["k_pre"], pre["k_post"])
        self.calls.append(pre)
        return q_out, k_out, v_out, residual_out, attn_output_out, attn_metadata_out


# ---------------------------------------------------------------------------
# Tiny-model helper. Constructs a 2-layer Llama on CPU with random weights
# so the integration tests can run anywhere.
# ---------------------------------------------------------------------------
def _build_tiny_llama(dtype: torch.dtype = torch.float32):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Shared end-to-end driver.
# ---------------------------------------------------------------------------
def _run_compute_layer(model, seq_len: int, dtype: torch.dtype, device: torch.device):
    """
    Construct the LMC adapter + stub blender + recording proxy, then
    drive compute_layer to completion. Returns (recorder, captured_hidden,
    num_hidden_layers, hidden_size, num_tokens).
    """
    stub = LMCBlender(model, num_layers=len(model.model.layers))
    recorder = RecordingBlender(stub)
    lmc_model = infer_model_from_hf(model, recorder)
    # LMCBaseModel.__init__ ran `recorder.layerwise_model = self`; the
    # proxy's __setattr__ forwards that into the inner stub as well.

    captured = {}
    # Forward hook on the last layer's mlp so we can inspect the final
    # hidden state without altering compute_layer or yielding from it.
    last_layer = model.model.layers[-1]
    def hook(_mod, _inp, out):
        captured["final_hidden"] = out.detach()
    handle = last_layer.mlp.register_forward_hook(hook)

    torch.manual_seed(1234)
    input_ids = torch.randint(0, model.config.vocab_size, (seq_len,), device=device)
    gen = lmc_model.compute_layer(input_ids)

    yield_count = 0
    for _ in range(lmc_model.num_layers):
        next(gen)
        yield_count += 1
    # Exhausted: a further next() should raise StopIteration.
    extra_yield = False
    try:
        next(gen)
        extra_yield = True
    except StopIteration:
        pass

    handle.remove()
    return {
        "recorder": recorder,
        "captured_hidden": captured.get("final_hidden"),
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_tokens": seq_len,
        "yield_count": yield_count,
        "extra_yield": extra_yield,
        "lmc_model": lmc_model,
    }


# ---------------------------------------------------------------------------
# Tiny-model tests (run on CPU; no model download).
# ---------------------------------------------------------------------------
class TestTinyModelLayerwise:
    """
    Phase 1 §1–§5 against a synthetic 2-layer Llama on CPU. Same code
    paths as the real-model tests, just smaller.
    """

    @pytest.fixture(scope="class")
    def tiny_run(self):
        model = _build_tiny_llama(dtype=torch.float32)
        device = torch.device("cpu")
        return _run_compute_layer(model, seq_len=16, dtype=torch.float32, device=device)

    def test_criterion_1_runs_without_raising(self, tiny_run):
        assert tiny_run["yield_count"] == tiny_run["num_hidden_layers"]

    def test_criterion_2_yields_exactly_num_hidden_layers(self, tiny_run):
        assert tiny_run["yield_count"] == tiny_run["num_hidden_layers"]
        assert tiny_run["extra_yield"] is False, (
            "compute_layer yielded more than num_hidden_layers times"
        )

    def test_criterion_3_final_hidden_state_shape(self, tiny_run):
        hidden = tiny_run["captured_hidden"]
        assert hidden is not None, "MLP forward hook did not fire"
        # The MLP output keeps the (..., hidden_size) trailing dim.
        # In our compute_layer hidden_states is (num_tokens, hidden_size).
        assert hidden.shape == (tiny_run["num_tokens"], tiny_run["hidden_size"])

    def test_criterion_5_process_qkv_per_layer_with_rope(self, tiny_run):
        rec = tiny_run["recorder"]
        n_layers = tiny_run["num_hidden_layers"]
        n_tokens = tiny_run["num_tokens"]
        assert len(rec.calls) == n_layers, (
            f"Expected {n_layers} process_qkv calls, got {len(rec.calls)}"
        )
        for layer_id, call in enumerate(rec.calls):
            assert call["layer_id"] == layer_id
            assert call["q_shape"][0] == n_tokens
            assert call["k_shape"][0] == n_tokens
            assert call["v_shape"][0] == n_tokens
            # K must have changed via RoPE (positions > 0 in a 16-token
            # prompt always produce non-identity rotation).
            assert call["k_changed_by_rope"], (
                f"layer {layer_id}: K identical before/after stub process_qkv; "
                "RoPE wrapper did not run"
            )

    def test_blend_metadata_dataclass_shape(self):
        """
        Sanity check: Phase 1 already declares the three-field shape so
        Phase 3 can drop in the full blender without rewriting it.
        (Per the docs/phases/PHASE1_PROMPT.md Review notes.)
        """
        md = LMCBlendMetadata()
        assert md.positions is None
        assert md.imp_indices is None
        assert md.attn_mask is None
        md.clean()  # idempotent on a fresh instance

    def test_double_construction_idempotent(self):
        """
        Building LMCBaseModel twice on the same hf_model must not
        double-wrap the qkv_proj / rotary_emb / o_proj / layernorm
        adapters. Otherwise iterated Phase 1 tests inside the same
        session would corrupt the model.
        """
        model = _build_tiny_llama()
        stub1 = LMCBlender(model, num_layers=len(model.model.layers))
        m1 = infer_model_from_hf(model, stub1)
        stub2 = LMCBlender(model, num_layers=len(model.model.layers))
        m2 = infer_model_from_hf(model, stub2)
        # The o_proj wrapper introduced by the first patch should still
        # be a function (not a function-wrapping-a-function with extra
        # tuple). Verify by calling it on a tensor.
        x = torch.zeros(4, model.config.hidden_size)
        out = model.model.layers[0].self_attn.o_proj(x)
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert out[1] is None


# ---------------------------------------------------------------------------
# Real-model tests (Mistral-7B + Llama-3.1-8B). CUDA required.
# ---------------------------------------------------------------------------
REAL_MODELS_ENABLED = os.environ.get("LMC_PHASE1_REAL_MODELS") == "1"
CUDA_AVAILABLE = torch.cuda.is_available()
REAL_MODEL_NAMES = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]


@pytest.mark.skipif(
    not (CUDA_AVAILABLE and REAL_MODELS_ENABLED),
    reason=(
        "Real-model Phase 1 tests need CUDA + LMC_PHASE1_REAL_MODELS=1. "
        "Tiny-model tests cover the same code paths on CPU."
    ),
)
class TestRealModelLayerwise:

    # Class-scoped fixture that loads each model once and shares the run
    # across all five criteria tests. `params=` on the fixture itself
    # parametrizes at class scope (function-scope @parametrize on the
    # class would mismatch scopes here).
    @pytest.fixture(scope="class", params=REAL_MODEL_NAMES)
    def real_run(self, request):
        model_name = request.param
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        ).to("cuda:0")
        model.eval()
        result = _run_compute_layer(
            model, seq_len=64, dtype=torch.float16, device=torch.device("cuda:0"),
        )
        result["model_name"] = model_name
        result["model"] = model
        yield result
        # Clean up GPU memory between parametrized invocations.
        # The result dict keeps several handles that all chain back to
        # the model (`lmc_model.vllm_model`, `recorder.vllm_model`,
        # `recorder._inner.vllm_model`, etc.), so del'ing them in order
        # is fragile. Move the model to CPU first to free VRAM
        # unconditionally, *then* drop refs so Python can release the
        # CPU copy at its leisure.
        try:
            model.to("cpu")
        except Exception:
            pass
        recorder = result.get("recorder")
        if recorder is not None:
            recorder.vllm_model = None
            if getattr(recorder, "_inner", None) is not None:
                recorder._inner.vllm_model = None
                recorder._inner.layerwise_model = None
            recorder.layerwise_model = None
        lmc = result.get("lmc_model")
        if lmc is not None:
            lmc.vllm_model = None
            lmc.blender = None
        result.clear()
        del result, recorder, lmc, model
        gc.collect()
        torch.cuda.empty_cache()

    def test_criterion_1_runs(self, real_run):
        assert real_run["yield_count"] == real_run["num_hidden_layers"]

    def test_criterion_2_yield_count(self, real_run):
        assert real_run["yield_count"] == real_run["num_hidden_layers"]
        assert real_run["extra_yield"] is False

    def test_criterion_3_final_hidden_shape(self, real_run):
        h = real_run["captured_hidden"]
        assert h is not None
        assert h.shape == (real_run["num_tokens"], real_run["hidden_size"])

    def test_criterion_4_memory_smoke(self, real_run):
        """
        Run compute_layer a second time on the same model and assert
        peak memory delta stays within ~1.5× the first peak. Treat as
        smoke check — eager attention has cuBLAS workspace fluctuations
        (per docs/VERIFICATION_PROTOCOL.md §1).
        """
        model = real_run["model"]
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        _ = _run_compute_layer(
            model, seq_len=64, dtype=torch.float16, device=device,
        )
        first_peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        _ = _run_compute_layer(
            model, seq_len=64, dtype=torch.float16, device=device,
        )
        second_peak = torch.cuda.max_memory_allocated(device)
        assert second_peak <= int(first_peak * 1.5), (
            f"Memory grew unexpectedly: first {first_peak} bytes -> "
            f"second {second_peak} bytes"
        )

    def test_criterion_5_process_qkv_per_layer_with_rope(self, real_run):
        rec = real_run["recorder"]
        n_layers = real_run["num_hidden_layers"]
        n_tokens = real_run["num_tokens"]
        assert len(rec.calls) == n_layers
        for layer_id, call in enumerate(rec.calls):
            assert call["layer_id"] == layer_id
            assert call["q_shape"][0] == n_tokens
            assert call["k_shape"][0] == n_tokens
            assert call["k_changed_by_rope"], (
                f"layer {layer_id}: stub blender did not apply RoPE"
            )
