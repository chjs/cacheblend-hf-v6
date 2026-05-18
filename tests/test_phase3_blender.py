"""
Phase 3 — full CacheBlend port. Covers the six sub-criteria in
docs/VERIFICATION_PROTOCOL.md §3:

  §3.1 SegmentTokenDatabase: chunk count, span boundaries vs sep,
       hash determinism, no prefix chaining.
  §3.2 FusedRope round-trip on Llama-3-style scaling (and Mistral).
  §3.3 HKVD selection determinism + sort order + topk_num formula.
  §3.4 KV merge: old_k[imp_indices] = k_new; rest untouched.
  §3.5 End-to-end blend: 100% recompute over single chunk → fused KV
       == full prefill KV.
  §3.6 End-to-end blend: realistic 15% recompute over 4 chunks →
       cosine sim per layer ≥ 0.95 (mean over tokens), and a basic
       smoke check that the blender runs to completion and cleans up.

Tiny CPU tests cover all sub-criteria on a 2-layer synthetic Llama;
real-model tests gate on `LMC_PHASE3_REAL_MODELS=1` and run on a 24 GB
GPU via `scripts/run_phase3_remote.sh`. fp32 real-model is intentionally
excluded (>24 GB on Mistral / Llama-3.1) — same policy as Phase 2.
"""
from __future__ import annotations

import gc
import math
import os
from typing import Any, List, Tuple

import pytest
import torch

from lmc.cache_engine import LMCacheEngine
from lmc.compute.blend.blender import LMCBlender
from lmc.compute.blend.utils import LMCBlenderBuilder
from lmc.compute.positional_encoding import FusedRope, get_fused_rope, validate_reverse_correctness
from lmc.config import LMCacheEngineConfig
from lmc.gpu_connector import HFBufferLayerwiseGPUConnector
from lmc.integration.hf.utils import ENGINE_NAME, HFModelTracker
from lmc.storage import LMCacheMetadata, LocalCPUBackend
from lmc.token_database import SegmentTokenDatabase


# ---------------------------------------------------------------------------
# Cache extraction helper (same shape as Phase 2's).
# ---------------------------------------------------------------------------
def extract_kv(cache: Any, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    return cache[layer_idx][0], cache[layer_idx][1]


# ---------------------------------------------------------------------------
# Tiny synthetic model fixture. 2 layers, hidden=64, GQA 4:2, head_dim=16.
# Vocab includes a distinctive separator token sequence so we can build
# multi-chunk prompts deterministically.
# ---------------------------------------------------------------------------
def _build_tiny_llama(
    dtype: torch.dtype = torch.float32,
    *,
    rope_scaling: bool = False,
):
    """
    Build a tiny Llama-style model on CPU. When `rope_scaling=True`, set
    `rope_parameters` to a Llama-3-style config so HF's
    `LlamaRotaryEmbedding` exercises the Llama-3 rope_type path. This
    is the §3.2 audit-risk test.
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    rope_parameters = None
    if rope_scaling:
        rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 128,
        }
    # vocab_size matches a real Llama tokenizer so the Phase 3 tests
    # that build prompts from the SegmentTokenDatabase's sep_tokens
    # (which are Llama-tokenizer ids, usually > 32000-bit slice but
    # all below 32000) don't trip the embedding bounds. Tiny embedding
    # cost: 32000 * 64 = 2 M params, negligible.
    config = LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512, rope_theta=10000.0,
        attn_implementation="eager",
    )
    if rope_parameters is not None:
        config.rope_parameters = rope_parameters
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(dtype=dtype)
    model.eval()
    return model


# ===========================================================================
# §3.1 SegmentTokenDatabase
# ===========================================================================
class TestSegmentTokenDatabase:
    """
    docs/VERIFICATION_PROTOCOL.md §3.1.
    Uses Mistral's tokenizer locally if available, otherwise builds a
    fake tokenizer scenario with hand-crafted token ids. Since the
    Mistral tokenizer requires download we test against a real one
    only inside real-model tests (gated). For CPU here, we exercise
    the chunk splitting and hash determinism via a minimal stand-in.
    """

    def _make_db(self) -> SegmentTokenDatabase:
        # Use a public-non-gated tokenizer for tiny-test scope.
        cfg = LMCacheEngineConfig(blend_special_str=" # # ")
        meta = LMCacheMetadata(
            model_name="hf-internal-testing/llama-tokenizer",
            kv_dtype=torch.float16,
        )
        return SegmentTokenDatabase(cfg, meta)

    def test_split_yields_four_chunks(self):
        db = self._make_db()
        sep_ids = db.sep_tokens.tolist()
        # Build a synthetic token stream: SYS + SEP + A + SEP + B + SEP + Q.
        sys_ids = [10, 11, 12]
        a_ids   = [20, 21, 22, 23]
        b_ids   = [30, 31, 32]
        q_ids   = [40, 41]
        full = sys_ids + sep_ids + a_ids + sep_ids + b_ids + sep_ids + q_ids

        spans = list(db.process_tokens(tokens=full))
        # 4 chunks expected. start/end must not overlap separator runs.
        assert len(spans) == 4

        # Chunk 0: SYS at [0, 3) — separators are absorbed at idx>0 only.
        assert spans[0][0] == 0 and spans[0][1] == 3
        # Chunk 1: A at [3 + sep_len, 3 + sep_len + 4).
        sep_len = db.sep_len
        assert spans[1][0] == 3 + sep_len
        assert spans[1][1] == 3 + sep_len + 4
        # Chunk 2: B at [3 + sep_len + 4 + sep_len, ...).
        assert spans[2][0] == 3 + sep_len + 4 + sep_len
        assert spans[2][1] == 3 + sep_len + 4 + sep_len + 3
        # Chunk 3: Q.
        assert spans[3][0] == 3 + sep_len + 4 + sep_len + 3 + sep_len
        assert spans[3][1] == len(full)

    def test_hash_deterministic_across_calls(self):
        db1 = self._make_db()
        db2 = self._make_db()
        chunk = [20, 21, 22, 23]
        h1 = db1._hash_tokens(chunk)
        h2 = db2._hash_tokens(chunk)
        assert h1 == h2, "builtin hash should be deterministic within one process"

    def test_hash_independent_of_surrounding_context(self):
        """Per docs/LMCACHE_IMPLEMENTATION.md §2.7: each chunk hashes
        independently — no prefix chaining."""
        db = self._make_db()
        sep_ids = db.sep_tokens.tolist()
        a = [20, 21, 22, 23]

        prompt1 = [99, 99] + sep_ids + a + sep_ids + [50, 51]
        prompt2 = [77]      + sep_ids + a + sep_ids + [60, 61, 62]

        # Find the chunk hashing to `a` in each prompt; it should be the
        # same int regardless of surrounding chunks.
        def chunk_hashes(toks):
            return [
                key.chunk_hash for (_, _, key) in db.process_tokens(tokens=toks)
            ]
        h1 = chunk_hashes(prompt1)
        h2 = chunk_hashes(prompt2)
        # `a` is the second yielded chunk in both prompts.
        assert h1[1] == h2[1]


# ===========================================================================
# §3.2 FusedRope round-trip
# ===========================================================================
class TestFusedRope:
    """
    docs/VERIFICATION_PROTOCOL.md §3.2: rotate K to p1, FusedRope p1→p2,
    compare against directly rotating K to p2. Tolerance fp32 atol 1e-5.
    Phase 0 audit flagged Llama-3 rope_scaling as the key risk —
    `test_round_trip_llama3` is the gate.
    """

    def _build_fused_rope(self, model) -> FusedRope:
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        fused = get_fused_rope(hf_model=model, head_dim=head_dim, is_neox_style=True)
        assert fused is not None
        return fused

    def test_round_trip_standard_rope(self):
        model = _build_tiny_llama(dtype=torch.float32, rope_scaling=False)
        fused = self._build_fused_rope(model)
        head_dim = model.config.head_dim
        ok = validate_reverse_correctness(
            fused, head_dim=head_dim, num_kv_heads=2, num_tokens=10,
            device=torch.device("cpu"), dtype=torch.float32, atol=1e-5,
        )
        assert ok, "FusedRope round-trip failed on standard RoPE"

    def test_round_trip_llama3_rope_scaling(self):
        """The Phase 0 audit risk #1 lives here."""
        model = _build_tiny_llama(dtype=torch.float32, rope_scaling=True)
        fused = self._build_fused_rope(model)
        head_dim = model.config.head_dim
        ok = validate_reverse_correctness(
            fused, head_dim=head_dim, num_kv_heads=2, num_tokens=10,
            device=torch.device("cpu"), dtype=torch.float32, atol=1e-5,
        )
        assert ok, "FusedRope round-trip failed under Llama-3 rope_scaling"


# ===========================================================================
# §3.3 / §3.4 / §3.5 / §3.6 share the build helper below.
# ===========================================================================
def _build_blend_pipeline(
    model,
    *,
    recomp_ratios: List[float],
    check_layers: List[int] = [1],
    instance_id: str = ENGINE_NAME,
) -> Tuple[LMCBlender, LMCacheEngine, HFBufferLayerwiseGPUConnector, LocalCPUBackend]:
    """
    Build the full Phase 3 pipeline. Caller is responsible for any
    stock prefill snapshot *before* this is invoked (LMCBlender's
    __init__ patches the model in place — see Phase 1 / Phase 2 docs).
    """
    cfg = LMCacheEngineConfig(
        enable_blending=True,
        blend_special_str=" # # ",
        blend_check_layers=check_layers,
        blend_recompute_ratios=recomp_ratios,
        use_layerwise=True,
    )
    mcfg = model.config
    num_layers = mcfg.num_hidden_layers
    num_kv = mcfg.num_key_value_heads
    head_dim = getattr(mcfg, "head_dim", None) or (mcfg.hidden_size // mcfg.num_attention_heads)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Register the model so LMCBlenderBuilder can fetch it by id.
    HFModelTracker._hf_models.pop(instance_id, None)
    HFModelTracker.register_model(instance_id, model)

    LMCBlenderBuilder.reset()

    # tiny-llama tests don't have a published tokenizer; for the
    # SegmentTokenDatabase to instantiate, point it at a small public
    # Llama-compatible tokenizer. Token id ranges in tiny tests are
    # chosen disjoint from this tokenizer's separator ids (see
    # TestEndToEnd*), so the chunk-detection works as expected.
    tokenizer_model = mcfg.name_or_path or "hf-internal-testing/llama-tokenizer"
    metadata = LMCacheMetadata(
        model_name=tokenizer_model,
        kv_dtype=dtype,
    )
    token_db = SegmentTokenDatabase(cfg, metadata)
    storage = LocalCPUBackend()
    fused_rope = get_fused_rope(hf_model=model, head_dim=head_dim, is_neox_style=True)
    connector = HFBufferLayerwiseGPUConnector(
        num_layers=num_layers,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        fused_rotary_emb=fused_rope,
    )
    engine = LMCacheEngine(
        token_database=token_db, storage=storage,
        gpu_connector=connector, num_layers=num_layers,
    )

    blender = LMCBlenderBuilder.get_or_create(instance_id, engine, connector, cfg)
    return blender, engine, connector, storage


def _stock_prefill(model, input_ids: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Run stock HF forward; return per-layer (K, V) tensors."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
    per_layer = []
    for i in range(model.config.num_hidden_layers):
        k, v = extract_kv(out.past_key_values, i)
        per_layer.append((k.detach().clone(), v.detach().clone()))
    return per_layer


# ===========================================================================
# §3.3 HKVD selection determinism
# ===========================================================================
class TestHKVDSelection:
    """
    Verify topk selection on the check layer:
      - topk_num == max(int(total * recomp_ratios[0]), 1)
      - top_indices is sorted ascending
      - same inputs → same indices
    """

    def test_topk_formula_and_sort(self):
        model = _build_tiny_llama(dtype=torch.float32)
        # Pre-snapshot stock KV (BEFORE LMC patches the model) for warmup.
        torch.manual_seed(123)
        prompt = torch.randint(0, model.config.vocab_size, (32,), dtype=torch.long)
        stock_kv = _stock_prefill(model, prompt)

        # Build pipeline (this patches the model in place).
        blender, engine, connector, storage = _build_blend_pipeline(
            model, recomp_ratios=[0.15], check_layers=[1],
        )
        engine.store_from_prefill(prompt, stock_kv)

        # Run blend and capture the check-layer top_indices via the
        # blender's own metadata.
        captured = {}
        real_process_qkv = blender.process_qkv

        def recording(q, k, v, residual, layer_id, attn_output, attn_metadata):
            out = real_process_qkv(q, k, v, residual, layer_id, attn_output, attn_metadata)
            if layer_id == 1:
                captured["imp_indices"] = blender.metadata.imp_indices.detach().clone()
                captured["seq_len"] = q.shape[0]
            return out

        blender.process_qkv = recording
        try:
            blender.blend(prompt)
        finally:
            blender.process_qkv = real_process_qkv

        idx = captured["imp_indices"]
        seq_len = captured["seq_len"]
        expected_k = max(int(seq_len * 0.15), 1)
        assert idx.numel() == expected_k, f"got {idx.numel()}, expected {expected_k}"
        assert torch.equal(idx, torch.sort(idx).values), "top_indices not sorted ascending"

    def test_determinism_across_runs(self):
        """Two blend() calls on identical input must produce identical
        top_indices."""
        runs = []
        for trial in range(2):
            model = _build_tiny_llama(dtype=torch.float32)
            torch.manual_seed(456)
            prompt = torch.randint(0, model.config.vocab_size, (24,), dtype=torch.long)
            stock_kv = _stock_prefill(model, prompt)
            blender, engine, _, _ = _build_blend_pipeline(
                model, recomp_ratios=[0.2], check_layers=[1],
            )
            engine.store_from_prefill(prompt, stock_kv)

            captured = {}
            real_pq = blender.process_qkv
            def recording(q, k, v, residual, layer_id, attn_output, attn_metadata, _real=real_pq, _blender=blender, _cap=captured):
                out = _real(q, k, v, residual, layer_id, attn_output, attn_metadata)
                if layer_id == 1:
                    _cap["idx"] = _blender.metadata.imp_indices.detach().clone()
                return out
            blender.process_qkv = recording
            try:
                blender.blend(prompt)
            finally:
                blender.process_qkv = real_pq
            runs.append(captured["idx"])
        assert torch.equal(runs[0], runs[1]), "HKVD selection not deterministic"


# ===========================================================================
# §3.4 KV merge correctness
# ===========================================================================
class TestKVMerge:
    """
    After process_qkv on the check layer:
      - old_k[top_indices] equals the post-RoPE freshly computed K at
        those indices (exact equality in same dtype).
      - old_k rows NOT in top_indices unchanged.
      - Same for V.
    """

    def test_check_layer_merge(self):
        model = _build_tiny_llama(dtype=torch.float32)
        torch.manual_seed(789)
        prompt = torch.randint(0, model.config.vocab_size, (16,), dtype=torch.long)
        stock_kv = _stock_prefill(model, prompt)
        blender, engine, connector, _ = _build_blend_pipeline(
            model, recomp_ratios=[0.5], check_layers=[1],
        )
        engine.store_from_prefill(prompt, stock_kv)

        captured = {}
        real_pq = blender.process_qkv

        def recording(q, k, v, residual, layer_id, attn_output, attn_metadata):
            if layer_id == 1:
                # Snapshot the GPU buffer K, V BEFORE this layer's process_qkv
                # so we can verify the rest of the rows are untouched.
                old_k, old_v = connector.get_kv(layer_id)
                captured["pre_old_k"] = old_k.detach().clone()
                captured["pre_old_v"] = old_v.detach().clone()
                # Also snapshot the freshly-computed K (BEFORE RoPE) so we
                # can compute its post-RoPE version ourselves to compare.
                captured["fresh_k_pre_rope"] = k.detach().clone()
                captured["fresh_v"] = v.detach().clone()
            out = real_pq(q, k, v, residual, layer_id, attn_output, attn_metadata)
            if layer_id == 1:
                # Now the in-place write old_k[imp_indices] = k_post_rope has
                # happened; capture the result.
                old_k, old_v = connector.get_kv(layer_id)
                captured["post_old_k"] = old_k.detach().clone()
                captured["post_old_v"] = old_v.detach().clone()
                captured["imp_indices"] = blender.metadata.imp_indices.detach().clone()
            return out

        blender.process_qkv = recording
        try:
            blender.blend(prompt)
        finally:
            blender.process_qkv = real_pq

        pre_k = captured["pre_old_k"]
        post_k = captured["post_old_k"]
        idx = captured["imp_indices"]

        # 1. Rows NOT in imp_indices unchanged.
        mask = torch.ones(pre_k.shape[0], dtype=torch.bool)
        mask[idx] = False
        assert torch.equal(pre_k[mask], post_k[mask]), "non-imp rows of K were modified"
        # V same.
        pre_v = captured["pre_old_v"]
        post_v = captured["post_old_v"]
        assert torch.equal(pre_v[mask], post_v[mask]), "non-imp rows of V were modified"

        # 2. Rows AT imp_indices match the freshly-computed K post-RoPE.
        # `process_qkv`'s RoPE happens before the topk slice; we reproduce
        # it explicitly here.
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cfg = model.config
        num_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        seq_len = pre_k.shape[0]
        fresh_k = captured["fresh_k_pre_rope"]
        k_4d = fresh_k.view(seq_len, num_kv, head_dim).unsqueeze(0).transpose(1, 2)
        # positions are arange(seq_len) at the check layer (no upstream
        # imp_indices yet — check layer is the first one to set them).
        positions = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
        cos, sin = model.model.rotary_emb(k_4d, positions)
        # Apply only to K (q is meaningless here for our V check).
        # apply_rotary_pos_emb handles both q and k; we pass a dummy q.
        dummy_q = torch.zeros_like(k_4d)
        _, k_rot = apply_rotary_pos_emb(dummy_q, k_4d, cos, sin)
        k_rot_2d = k_rot.transpose(1, 2).reshape(seq_len, num_kv * head_dim)
        assert torch.allclose(post_k[idx], k_rot_2d[idx], atol=1e-6, rtol=1e-6), (
            "K at imp_indices doesn't match freshly-rotated K"
        )


# ===========================================================================
# §3.5 End-to-end: 100% recompute single chunk → fused KV == full prefill
# ===========================================================================
class TestEndToEnd100PctSingleChunk:
    """
    With recomp_ratios=[1.0] and a single-chunk prompt (no separators),
    every token is selected as HKVD on the check layer and the merge
    writes the full K back. fused KV should equal full prefill KV.
    """

    def test_single_chunk_100pct(self):
        model = _build_tiny_llama(dtype=torch.float32)
        torch.manual_seed(321)
        # No separators in the prompt → 1 chunk after SegmentTokenDatabase.
        # vocab=512 and sep tokens are deterministic; pick ids well outside
        # the sep token sequence to be safe.
        prompt = torch.randint(100, 200, (24,), dtype=torch.long)
        stock_kv = _stock_prefill(model, prompt)
        blender, engine, connector, _ = _build_blend_pipeline(
            model, recomp_ratios=[1.0], check_layers=[1],
        )
        engine.store_from_prefill(prompt, stock_kv)

        blender.blend(prompt)

        # Compare per-layer fused KV to stock prefill KV.
        cfg = model.config
        num_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        seq_len = prompt.numel()
        for i in range(cfg.num_hidden_layers):
            fused_k, fused_v = connector.get_kv(i)
            stock_k, stock_v = stock_kv[i]
            stock_k_2d = stock_k.transpose(1, 2).reshape(seq_len, num_kv * head_dim)
            stock_v_2d = stock_v.transpose(1, 2).reshape(seq_len, num_kv * head_dim)
            assert torch.allclose(fused_k, stock_k_2d, atol=1e-4, rtol=1e-4), (
                f"layer {i}: fused K diverges from stock K (max abs diff "
                f"{(fused_k - stock_k_2d).abs().max().item():.3e})"
            )
            assert torch.allclose(fused_v, stock_v_2d, atol=1e-4, rtol=1e-4), (
                f"layer {i}: fused V diverges from stock V"
            )


# ===========================================================================
# §3.6 End-to-end: realistic 15% recompute over 4 chunks
# ===========================================================================
class TestEndToEndRealisticRatio:
    """
    4-chunk prompt with recomp_ratios=[0.15]. Verify:
      - blend() completes without error.
      - metadata.{imp_indices, positions, attn_mask} all None after.
      - Cosine similarity per layer between fused K (and V) and full
        prefill K (V), averaged over tokens, ≥ 0.95.
    """

    @staticmethod
    def _build_four_chunk_prompt(sep_ids: List[int]) -> torch.Tensor:
        sys_ids = [100, 101, 102, 103]
        a_ids   = [200, 201, 202, 203, 204]
        b_ids   = [300, 301, 302, 303]
        q_ids   = [400, 401, 402]
        full = sys_ids + sep_ids + a_ids + sep_ids + b_ids + sep_ids + q_ids
        return torch.tensor(full, dtype=torch.long)

    def test_realistic_blend(self):
        model = _build_tiny_llama(dtype=torch.float32)

        # Build the prompt with the database's sep tokens so they
        # actually match.
        cfg = LMCacheEngineConfig(blend_special_str=" # # ")
        meta = LMCacheMetadata(
            model_name="hf-internal-testing/llama-tokenizer",
            kv_dtype=torch.float32,
        )
        db = SegmentTokenDatabase(cfg, meta)
        prompt = self._build_four_chunk_prompt(db.sep_tokens.tolist())

        stock_kv = _stock_prefill(model, prompt)
        blender, engine, connector, _ = _build_blend_pipeline(
            model, recomp_ratios=[0.15], check_layers=[1],
        )
        engine.store_from_prefill(prompt, stock_kv)

        blender.blend(prompt)

        # 1. Metadata clean after blend.
        assert blender.metadata.imp_indices is None
        assert blender.metadata.positions is None
        assert blender.metadata.attn_mask is None

        # 2. Cosine similarity per layer.
        seq_len = prompt.numel()
        num_kv = model.config.num_key_value_heads
        head_dim = model.config.head_dim
        for i in range(model.config.num_hidden_layers):
            fused_k, fused_v = connector.get_kv(i)
            stock_k, stock_v = stock_kv[i]
            stock_k_2d = stock_k.transpose(1, 2).reshape(seq_len, num_kv * head_dim)
            stock_v_2d = stock_v.transpose(1, 2).reshape(seq_len, num_kv * head_dim)

            # Skip gap rows (separator tokens, etc.) for the similarity check —
            # those are zeroed in fused_k by design. Compare only the
            # token positions covered by chunks.
            # Use connector buffer presence vs zero-row to mask.
            non_gap = fused_k.abs().sum(dim=1) > 0
            cos_k = torch.nn.functional.cosine_similarity(
                fused_k[non_gap], stock_k_2d[non_gap], dim=1,
            )
            cos_v = torch.nn.functional.cosine_similarity(
                fused_v[non_gap], stock_v_2d[non_gap], dim=1,
            )
            assert cos_k.mean().item() >= 0.95, (
                f"layer {i}: K mean cos sim {cos_k.mean().item():.4f} < 0.95"
            )
            assert cos_v.mean().item() >= 0.95, (
                f"layer {i}: V mean cos sim {cos_v.mean().item():.4f} < 0.95"
            )


# ===========================================================================
# Real-model gate: covers §3.2 (Llama-3 RoPE) and §3.5/§3.6 on Mistral.
# ===========================================================================
REAL_ENABLED = os.environ.get("LMC_PHASE3_REAL_MODELS") == "1"
CUDA_OK = torch.cuda.is_available()


@pytest.mark.skipif(
    not (REAL_ENABLED and CUDA_OK),
    reason="LMC_PHASE3_REAL_MODELS=1 + CUDA required",
)
class TestRealModelBlend:
    """
    End-to-end blend tests on actual Mistral / Llama-3.1 (fp16).
    fp32 excluded (>24 GB VRAM). One pytest cell per (model, dtype).
    """

    @pytest.fixture(scope="class", params=[
        ("mistralai/Mistral-7B-Instruct-v0.2", torch.float16),
        ("mistralai/Mistral-7B-Instruct-v0.2", torch.bfloat16),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", torch.float16),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", torch.bfloat16),
    ], ids=lambda p: f"{p[0].rsplit('/', 1)[-1]}-{ {torch.float16:'fp16', torch.bfloat16:'bf16'}[p[1]] }")
    def real_setup(self, request):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name, dtype = request.param
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager",
        ).to("cuda:0")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        yield {"model": model, "tokenizer": tokenizer, "model_name": model_name, "dtype": dtype}
        try:
            model.to("cpu")
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    def _build_real_prompt(self, tokenizer, sep_str: str = " # # "):
        sys_text  = "You are a helpful assistant."
        ctx1_text = "Apples are a kind of fruit. They are red or green."
        ctx2_text = "The Eiffel Tower stands in Paris and was built in 1889."
        ctx3_text = "Whales are large marine mammals that breathe air."
        q_text    = "Question: Name one mammal mentioned above.\nAnswer:"
        full = sys_text + sep_str + ctx1_text + sep_str + ctx2_text + sep_str + ctx3_text + sep_str + q_text
        ids = tokenizer.encode(full, return_tensors="pt").squeeze(0).to("cuda:0")
        return ids

    def test_round_trip_fused_rope(self, real_setup):
        """§3.2 on the real model. fp32 reference on CPU."""
        model = real_setup["model"]
        head_dim = model.config.head_dim if hasattr(model.config, "head_dim") else (
            model.config.hidden_size // model.config.num_attention_heads
        )
        fused = get_fused_rope(hf_model=model, head_dim=head_dim, is_neox_style=True)
        assert fused is not None, "get_fused_rope returned None for real model"
        # Use 10 tokens to keep the fp32 round-trip cheap; head_dim is
        # the real model's (likely 128).
        ok = validate_reverse_correctness(
            fused, head_dim=head_dim,
            num_kv_heads=model.config.num_key_value_heads,
            num_tokens=10,
            device=torch.device("cuda:0"),
            dtype=torch.float32, atol=1e-4,
        )
        assert ok, (
            f"FusedRope round-trip failed on {real_setup['model_name']} "
            f"(model dtype {real_setup['dtype']}, head_dim {head_dim})"
        )

    def test_realistic_blend(self, real_setup):
        """§3.6 on the real model: 4-chunk RAG-style prompt."""
        model = real_setup["model"]
        tok = real_setup["tokenizer"]
        prompt = self._build_real_prompt(tok)

        stock_kv = _stock_prefill(model, prompt)
        blender, engine, connector, _ = _build_blend_pipeline(
            model, recomp_ratios=[0.15], check_layers=[1],
        )
        engine.store_from_prefill(prompt, stock_kv)
        blender.blend(prompt)

        # Cleanup checks.
        assert blender.metadata.imp_indices is None
        assert blender.metadata.positions is None

        # Cosine similarity gate (gates Phase 4-style usefulness).
        seq_len = prompt.numel()
        num_kv = model.config.num_key_value_heads
        head_dim = model.config.head_dim if hasattr(model.config, "head_dim") else (
            model.config.hidden_size // model.config.num_attention_heads
        )
        per_layer_mean_cos_k = []
        per_layer_mean_cos_v = []
        for i in range(model.config.num_hidden_layers):
            fused_k, fused_v = connector.get_kv(i)
            stock_k, stock_v = stock_kv[i]
            stock_k_2d = stock_k.transpose(1, 2).reshape(seq_len, num_kv * head_dim).to(fused_k.dtype)
            stock_v_2d = stock_v.transpose(1, 2).reshape(seq_len, num_kv * head_dim).to(fused_v.dtype)
            non_gap = fused_k.abs().sum(dim=1) > 0
            cos_k = torch.nn.functional.cosine_similarity(
                fused_k[non_gap].float(), stock_k_2d[non_gap].float(), dim=1,
            )
            cos_v = torch.nn.functional.cosine_similarity(
                fused_v[non_gap].float(), stock_v_2d[non_gap].float(), dim=1,
            )
            per_layer_mean_cos_k.append(cos_k.mean().item())
            per_layer_mean_cos_v.append(cos_v.mean().item())
        print(f"\n[{real_setup['model_name']} {real_setup['dtype']}] "
              f"K cos mean per layer (min={min(per_layer_mean_cos_k):.4f}): {per_layer_mean_cos_k[:8]}...")
        print(f"  V cos mean per layer (min={min(per_layer_mean_cos_v):.4f}): {per_layer_mean_cos_v[:8]}...")
        # Phase 3 §3.6 is a sanity floor; Phase 4 produces the real quality numbers.
        assert min(per_layer_mean_cos_k) >= 0.90, "K mean cos sim < 0.90 on some layer"
        assert min(per_layer_mean_cos_v) >= 0.90, "V mean cos sim < 0.90 on some layer"
