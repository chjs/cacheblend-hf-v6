"""
Phase 4 — RAG quality comparison for CacheBlend-HF.

Two datasets are supported via `--dataset {musique,loong}`:
  - MuSiQue Answerable: 6 chunks (supporting + nearest non-supporting
    by L2). Shorter prompts.
  - Loong: ~11 chunks per example, all provided documents used. Longer
    prompts; intended to expose cache-miss recovery more clearly.

Compares prefill methods on the second prompt of each example:
  1. Full KV recompute (stock HF prefill on the second prompt).
  2. Full KV reuse (Prompt Cache baseline: cached chunk KVs + RoPE
     shift, no recomputation).
  3. CacheBlend at a sweep of recompute ratios.

The first prompt is a *warmup-only* prompt ending in a dummy query;
its sole purpose is to populate per-chunk KV under the chunks' hashes.
The second prompt ends in the *real* question and is the only prompt
evaluated for F1. The real question segment must therefore be a cache
MISS — segment-level diagnostics are written for every example.

See docs/phases/PHASE4_PROMPT.md for the full spec.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

# Make `scripts.` package-style imports work when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from scripts._phase4_f1 import f1_with_aliases  # noqa: E402
from scripts._phase4_loong import (               # noqa: E402
    DEFAULT_LOONG_NUM_CHUNKS, assign_length_bucket, iter_loong_cases,
    prompt_token_budget,
)
from scripts._phase4_musique import (             # noqa: E402
    DEFAULT_DUMMY_WARMUP_QUERY, DEFAULT_PREFIX, DEFAULT_QUESTION_TEMPLATE,
    iter_cases,
)
from scripts._phase4_runtime import (             # noqa: E402
    connector_to_dyncache, decode_from_full_cache, full_recompute_run,
    unpatch_hf_model,
)
from scripts._phase4_tokenize import (            # noqa: E402
    InternalSeparatorError, MaterializedCase, materialize,
)


# ---------------------------------------------------------------------------
# Method-name helpers (single source of truth for the order in tables).
# ---------------------------------------------------------------------------
METHOD_FULL_RECOMPUTE = "full_recompute"
METHOD_FULL_KV_REUSE = "full_kv_reuse"


def cacheblend_method_name(ratio: float) -> str:
    """`CacheBlend r=0.15` style for the markdown table; `cacheblend_r0.15`
    for JSON-friendly keys. Use the same convention everywhere."""
    return f"cacheblend_r{ratio:.2f}"


def cacheblend_display_name(ratio: float) -> str:
    return f"CacheBlend r={ratio:.2f}"


# ---------------------------------------------------------------------------
# Ratio parser (unit-testable).
# ---------------------------------------------------------------------------
def parse_recompute_ratios(spec: str) -> List[float]:
    """Parse `--cacheblend-recompute-ratios` from a comma-separated string.

    Preserves order, deduplicates while preserving first occurrence,
    raises ValueError on values outside [0, 1] or non-floats.
    """
    out: List[float] = []
    seen: set = set()
    for raw in spec.split(","):
        s = raw.strip()
        if not s:
            continue
        v = float(s)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"recompute ratio out of [0,1]: {v}")
        # Round to 4 dp for dedupe but preserve the parsed value in output.
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    if not out:
        raise ValueError("no valid recompute ratios parsed from spec")
    return out


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 — RAG comparison (MuSiQue / Loong)")
    p.add_argument("--dataset", choices=["musique", "loong"], default="musique",
                   help="Which dataset to evaluate against.")
    p.add_argument("--model", required=True, choices=[
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ])
    p.add_argument("--input-jsonl", required=True,
                   help="Path to dataset JSONL. For MuSiQue: musique_ans_v1.0_train.jsonl. "
                        "For Loong: loong.jsonl (or whatever conversion path).")
    p.add_argument("--num-examples", type=int, required=True,
                   help="Maximum number of examples to USE (after skips).")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
    p.add_argument("--output", required=True,
                   help="Path to write the aggregate markdown report.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--num-chunks", type=int, default=6,
                   help="Number of chunks for MuSiQue. Loong uses --loong-num-chunks.")
    p.add_argument("--blend-special-str", default="# #")
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument("--embedding-normalize", default="false",
                   help="'true' to L2-normalize embeddings before distance.")
    p.add_argument("--prefix", default=None,
                   help=f"Prompt prefix text. Default: {DEFAULT_PREFIX!r}")
    p.add_argument("--prefix-file", default=None,
                   help="Read prefix text from this file (takes precedence over --prefix).")
    p.add_argument("--question-template", default=DEFAULT_QUESTION_TEMPLATE)
    p.add_argument("--dummy-warmup-query", default=DEFAULT_DUMMY_WARMUP_QUERY,
                   help="Trailing segment of the FIRST (warmup-only) prompt. "
                        "Must differ from the real question.")
    p.add_argument("--on-too-many-supporting", choices=["skip", "error"], default="skip")
    p.add_argument("--on-internal-separator", choices=["skip", "error"], default="skip")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="If set, override the model's max position embedding for the "
                        "prompt-fits-in-context check. For Loong + Mistral-7B-v0.2, "
                        "defaults to 32768.")
    p.add_argument("--safety-margin", type=int, default=128,
                   help="Tokens reserved for generation overhead beyond max_new_tokens.")
    p.add_argument("--write-jsonl-details", default=None,
                   help="Optional path for per-example JSONL details.")
    p.add_argument("--cacheblend-check-layers", type=int, default=1)
    p.add_argument("--cacheblend-recompute-ratios", default="0.15",
                   help="Comma-separated list of CacheBlend recompute ratios. "
                        "Default keeps Phase 4 baseline (`0.15`). Pass "
                        "`0.0,0.05,0.15,0.30,0.50,1.00` for the rerun sweep.")

    # Loong-specific.
    p.add_argument("--loong-num-chunks", type=int, default=DEFAULT_LOONG_NUM_CHUNKS,
                   help="Number of chunks per Loong example.")
    p.add_argument("--loong-first-order", choices=["original", "random"], default="original",
                   help="First-prompt chunk order policy for Loong. "
                        "Second order is always the exact reverse.")
    p.add_argument("--loong-on-extra-chunks", choices=["first", "skip", "error"],
                   default="first",
                   help="If a Loong example has MORE than --loong-num-chunks chunks.")
    p.add_argument("--loong-on-fewer-chunks", choices=["skip", "error", "use_all"],
                   default="skip",
                   help="If a Loong example has FEWER than --loong-num-chunks chunks.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Embedder.
# ---------------------------------------------------------------------------
class _CachedSTEmbedder:
    """Tiny wrapper around sentence-transformers; cached at module level
    so we only load the model once per script run."""

    def __init__(self, name: str, batch_size: int = 32, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(name, device=device or "cpu")
        self._batch_size = batch_size

    def __call__(self, texts):
        return self._model.encode(
            list(texts), batch_size=self._batch_size,
            convert_to_numpy=True, show_progress_bar=False,
        )


def _build_embedder(name: str, batch_size: int) -> Optional[object]:
    try:
        return _CachedSTEmbedder(name, batch_size=batch_size)
    except Exception as e:  # pragma: no cover — diagnostic only
        print(
            f"WARN: failed to load sentence-transformers embedding model "
            f"{name!r}: {e}; falling back to deterministic hash embedding.",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# CacheBlend / FullReuse driver.
# ---------------------------------------------------------------------------
def _build_cacheblend_pipeline(
    model,
    *,
    use_full_reuse: bool,
    check_layers: List[int],
    recomp_ratios: List[float],
    instance_id: str,
):
    """
    Build a fresh (engine, connector, blender) tuple for this method.
    `instance_id` is used as the HFModelTracker / LMCBlenderBuilder key;
    use a different id between methods so the blender is rebuilt.
    """
    from lmc.cache_engine import LMCacheEngine
    from lmc.compute.blend.blender import LMCBlender
    from lmc.compute.blend.full_reuse_blender import LMCFullReuseBlender
    from lmc.compute.blend.utils import LMCBlenderBuilder
    from lmc.compute.positional_encoding import get_fused_rope
    from lmc.config import LMCacheEngineConfig
    from lmc.gpu_connector import HFBufferLayerwiseGPUConnector
    from lmc.integration.hf.utils import HFModelTracker
    from lmc.storage import LMCacheMetadata, LocalCPUBackend
    from lmc.token_database import SegmentTokenDatabase

    cfg = LMCacheEngineConfig(
        enable_blending=True,
        blend_special_str=getattr(model.config, "_phase4_blend_special_str", "# #"),
        blend_check_layers=list(check_layers),
        blend_recompute_ratios=list(recomp_ratios),
        use_layerwise=True,
    )
    mcfg = model.config
    num_layers = mcfg.num_hidden_layers
    num_kv = mcfg.num_key_value_heads
    head_dim = getattr(mcfg, "head_dim", None) or (
        mcfg.hidden_size // mcfg.num_attention_heads
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Register the model and rebuild cleanly each method.
    HFModelTracker._hf_models.pop(instance_id, None)
    HFModelTracker.register_model(instance_id, model)
    LMCBlenderBuilder._blenders.pop(instance_id, None)

    metadata = LMCacheMetadata(
        model_name=mcfg.name_or_path or instance_id, kv_dtype=dtype,
    )
    token_db = SegmentTokenDatabase(cfg, metadata)
    storage = LocalCPUBackend()
    fused_rope = get_fused_rope(hf_model=model, head_dim=head_dim, is_neox_style=True)
    connector = HFBufferLayerwiseGPUConnector(
        num_layers=num_layers, num_kv_heads=num_kv, head_dim=head_dim,
        dtype=dtype, device=device, fused_rotary_emb=fused_rope,
    )
    engine = LMCacheEngine(
        token_database=token_db, storage=storage,
        gpu_connector=connector, num_layers=num_layers,
    )

    blender_cls = LMCFullReuseBlender if use_full_reuse else LMCBlender
    blender = blender_cls(
        cache_engine=engine, gpu_connector=connector, hf_model=model, config=cfg,
    )
    return engine, connector, blender


# ---------------------------------------------------------------------------
# Segment cache-hit diagnostic.
# ---------------------------------------------------------------------------
def _compute_cache_hit_end(engine, second_prompt_ids: torch.Tensor) -> int:
    """
    Walk the SegmentTokenDatabase over the second prompt and return the
    end position of the LAST cache-hit segment (i.e. the position where
    the first cache MISS starts). Returns 0 if even the first segment
    misses.

    This boundary is where the blender's prefill must stop. Tokens past
    this position must be prefilled by stock HF forward (since
    CacheBlend's connector buffer can only cover positions up to the
    last cache hit).
    """
    token_db = engine.token_database
    num_layers = engine.num_layers
    second_cpu = second_prompt_ids.detach().to(device="cpu", dtype=torch.long)
    last_hit_end = 0
    for start, end, key in token_db.process_tokens(tokens=second_cpu):
        layer_keys = key.split_layers(num_layers)
        if engine.storage.contains(layer_keys[0]):
            last_hit_end = int(end)
        else:
            break
    return last_hit_end


@torch.no_grad()
def _manual_prefill_extend(
    model,
    *,
    tail_ids: torch.Tensor,         # 1-D long tensor on device
    base_cache,                     # DynamicCache with seq_len = prefix_len
    prefix_len: int,
) -> "DynamicCache":
    """
    Run stock HF forward on `tail_ids` starting from `base_cache`.
    Returns the extended cache (length = prefix_len + tail_ids.numel()).
    Model must be UNPATCHED at entry.
    """
    device = next(model.parameters()).device
    L_tail = int(tail_ids.numel())
    position_ids = torch.arange(
        prefix_len, prefix_len + L_tail, dtype=torch.long, device=device,
    ).view(1, -1)
    out = model(
        input_ids=tail_ids.view(1, -1),
        past_key_values=base_cache,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    return out.past_key_values


def _diagnose_segment_hits(
    engine,
    *,
    second_prompt_ids: torch.Tensor,
    materialized: MaterializedCase,
) -> Dict[str, Any]:
    """
    Walk the SegmentTokenDatabase over the SECOND prompt and for each
    segment check whether layer-0's key is present in the backend.

    The expected pattern after warmup with the first prompt:
      prefix              → HIT (same prefix in both prompts)
      chunk_<each>        → HIT (same per-chunk token ids in both prompts)
      real_question       → MISS (the first prompt's tail was the dummy
                            warmup query, which hashes differently)
    """
    token_db = engine.token_database
    num_layers = engine.num_layers

    # The token_database's process_tokens splits on `sep_ids` and yields
    # one (start, end, key) per segment in order. The materialized case
    # tells us how to name each segment.
    second_ids_cpu = second_prompt_ids.detach().to(device="cpu", dtype=torch.long)
    segment_names: List[str] = ["prefix"] + [
        f"chunk:{cid}" for cid in materialized.case.second_order
    ] + ["real_question"]

    per_segment_hit: Dict[str, bool] = {}
    per_segment_len: Dict[str, int] = {}
    retrieved_tokens = 0

    seg_iter = list(token_db.process_tokens(tokens=second_ids_cpu))
    if len(seg_iter) != len(segment_names):
        # If the segment count doesn't match what we expect, report as
        # an unknown layout rather than silently mislabel hits.
        return {
            "segments": [],
            "segment_token_lengths": {},
            "segment_cache_hit": {},
            "expected_segment_count": len(segment_names),
            "actual_segment_count": len(seg_iter),
            "retrieved_token_count": 0,
            "real_question_segment_cache_hit": None,
            "all_six_chunks_cache_hit": False,
            "prefix_cache_hit": None,
            "layout_mismatch": True,
        }

    for name, (start, end, key) in zip(segment_names, seg_iter):
        layer_keys = key.split_layers(num_layers)
        hit = engine.storage.contains(layer_keys[0])
        per_segment_hit[name] = bool(hit)
        per_segment_len[name] = int(end - start)
        if hit:
            retrieved_tokens += int(end - start)

    chunk_segment_names = [f"chunk:{cid}" for cid in materialized.case.second_order]
    all_chunks_hit = all(per_segment_hit.get(n, False) for n in chunk_segment_names)

    return {
        "segments": segment_names,
        "segment_token_lengths": per_segment_len,
        "segment_cache_hit": per_segment_hit,
        "retrieved_token_count": retrieved_tokens,
        "real_question_segment_cache_hit": per_segment_hit.get("real_question"),
        "all_six_chunks_cache_hit": all_chunks_hit,
        "prefix_cache_hit": per_segment_hit.get("prefix"),
        "expected_segment_count": len(segment_names),
        "actual_segment_count": len(seg_iter),
        "layout_mismatch": False,
    }


@torch.no_grad()
def _run_cache_method(
    model,
    tokenizer,
    materialized: MaterializedCase,
    *,
    use_full_reuse: bool,
    check_layers: List[int],
    recomp_ratios: List[float],
    max_new_tokens: int,
    stock_kv_first_prompt: List[Tuple[torch.Tensor, torch.Tensor]],
    instance_id: str,
    return_segment_diagnostics: bool = False,
) -> Tuple[str, float, List[int], Optional[Dict[str, Any]]]:
    """
    Run either Method 2 (Full reuse) or one ratio of Method 3 (CacheBlend)
    on the second prompt. Returns
        (generated_text, prefill_ms, generated_ids, segment_diagnostics_or_None).

    Assumes `model` is UNPATCHED at entry; this function patches it
    (via building the blender pipeline) and unpatches before decode.
    """
    device = next(model.parameters()).device

    engine, connector, blender = _build_cacheblend_pipeline(
        model,
        use_full_reuse=use_full_reuse,
        check_layers=check_layers,
        recomp_ratios=recomp_ratios,
        instance_id=instance_id,
    )

    first_prompt_ids = torch.tensor(
        materialized.first_prompt_ids, dtype=torch.long, device=device,
    )
    second_prompt_ids = torch.tensor(
        materialized.second_prompt_ids, dtype=torch.long, device=device,
    )

    # ----- Warmup (no F1 measured): populate the cache from the FIRST
    # (warmup-only) prompt. -----
    engine.store_from_prefill(first_prompt_ids, stock_kv_first_prompt)

    # ----- Segment cache-hit diagnostic (post-warmup, pre-blend) -----
    seg_diag: Optional[Dict[str, Any]] = None
    if return_segment_diagnostics:
        seg_diag = _diagnose_segment_hits(
            engine,
            second_prompt_ids=second_prompt_ids,
            materialized=materialized,
        )

    # ----- Decide blender vs. miss-tail split -----
    # The connector buffer can only cover positions up to the last cache
    # hit. Find that boundary and split the second prompt accordingly.
    L_full = int(second_prompt_ids.numel())
    cache_end = _compute_cache_hit_end(engine, second_prompt_ids)
    if cache_end <= 0:
        raise RuntimeError(
            "second prompt has no cache hits — warmup failed or sep/prefix "
            "mismatched. Cannot run cache method."
        )
    blender_input_ids = second_prompt_ids[:cache_end]
    miss_tail_ids = second_prompt_ids[cache_end:]   # may be empty

    # ----- Evaluation prefill on the cache-hit prefix -----
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    blender.blend(blender_input_ids)
    if cuda:
        torch.cuda.synchronize()
    prefill_ms = (time.time() - t0) * 1000.0

    # Pull the fused KV out of the connector before unpatching the model.
    mcfg = model.config
    num_layers = mcfg.num_hidden_layers
    num_kv = mcfg.num_key_value_heads
    head_dim = getattr(mcfg, "head_dim", None) or (
        mcfg.hidden_size // mcfg.num_attention_heads
    )
    fused_cache = connector_to_dyncache(
        connector, num_layers, num_kv, head_dim,
        target_dtype=next(model.parameters()).dtype,
        device=device,
    )

    unpatch_hf_model(model)

    # ----- Manual prefill of the miss tail (sep + real question) onto
    # the fused cache so it covers the full second prompt. -----
    if int(miss_tail_ids.numel()) > 0:
        if cuda:
            torch.cuda.synchronize()
        t_tail = time.time()
        fused_cache = _manual_prefill_extend(
            model,
            tail_ids=miss_tail_ids,
            base_cache=fused_cache,
            prefix_len=cache_end,
        )
        if cuda:
            torch.cuda.synchronize()
        prefill_ms += (time.time() - t_tail) * 1000.0

    assert fused_cache.get_seq_length() == L_full, (
        f"cache length {fused_cache.get_seq_length()} != prompt length {L_full} "
        f"(cache_end={cache_end}, tail_len={int(miss_tail_ids.numel())})"
    )

    text, gen_ids = decode_from_full_cache(
        model, tokenizer, second_prompt_ids, fused_cache, max_new_tokens,
    )

    connector.reset()
    del connector, engine, blender, fused_cache
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    return text, prefill_ms, gen_ids, seg_diag


# ---------------------------------------------------------------------------
# Per-example driver (all methods + bookkeeping).
# ---------------------------------------------------------------------------
def _stock_prefill(model, prompt_ids: torch.Tensor):
    """Run stock forward and return per-layer (K, V) tensors for warmup."""
    with torch.no_grad():
        out = model(
            input_ids=prompt_ids.view(1, -1), use_cache=True, return_dict=True,
        )
    cache = out.past_key_values
    layers = []
    nL = model.config.num_hidden_layers
    for i in range(nL):
        if hasattr(cache, "layers"):
            k, v = cache.layers[i].keys, cache.layers[i].values
        elif hasattr(cache, "key_cache"):
            k, v = cache.key_cache[i], cache.value_cache[i]
        else:
            k, v = cache[i][0], cache[i][1]
        layers.append((k.detach().clone(), v.detach().clone()))
    return layers


def _first_diff_index(a: List[int], b: List[int]) -> Optional[int]:
    """Return the first index where two token-id lists differ, or None."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _per_example(
    model,
    tokenizer,
    materialized: MaterializedCase,
    *,
    max_new_tokens: int,
    cacheblend_check_layers: List[int],
    cacheblend_recompute_ratios: List[float],
) -> Dict[str, Any]:
    """
    Run all methods on **the second prompt** of one example.

    Role split (unchanged):
      - `first_prompt_ids`  → **warmup only**. Stock HF prefill populates
        the cache with per-chunk KV. The first prompt's tail is a *dummy
        warmup query*, not the real MuSiQue question. No F1 is measured.
      - `second_prompt_ids` → **evaluation only**. All methods generate
        from this prompt; F1 is computed on those generations. The
        second prompt's tail is the *real* MuSiQue question segment;
        this segment must be a cache MISS.
    """
    device = next(model.parameters()).device
    case = materialized.case

    second_ids = torch.tensor(
        materialized.second_prompt_ids, dtype=torch.long, device=device,
    )
    first_ids = torch.tensor(
        materialized.first_prompt_ids, dtype=torch.long, device=device,
    )

    # ---- Method 1 (evaluation): Full KV recompute on the second prompt. ----
    text1, prefill1, gen1 = full_recompute_run(model, tokenizer, second_ids, max_new_tokens)
    f1_full = f1_with_aliases(text1, case.answer, case.answer_aliases)

    # ---- Warmup (no F1 measured): stock prefill on the FIRST prompt. ----
    stock_kv_first = _stock_prefill(model, first_ids)

    # ---- Method 2 (evaluation): Full KV reuse on the second prompt. ----
    text2, prefill2, gen2, seg_diag = _run_cache_method(
        model, tokenizer, materialized,
        use_full_reuse=True,
        check_layers=cacheblend_check_layers,
        recomp_ratios=[0.0],   # ignored by FullReuse but still required.
        max_new_tokens=max_new_tokens,
        stock_kv_first_prompt=stock_kv_first,
        instance_id="hf_cacheblend_phase4_full_reuse",
        return_segment_diagnostics=True,
    )
    f1_reuse = f1_with_aliases(text2, case.answer, case.answer_aliases)

    # ---- Method 3 (evaluation): CacheBlend across all ratios. ----
    cacheblend_results: Dict[str, Dict[str, Any]] = {}
    for r in cacheblend_recompute_ratios:
        text3, prefill3, gen3, _ = _run_cache_method(
            model, tokenizer, materialized,
            use_full_reuse=False,
            check_layers=cacheblend_check_layers,
            recomp_ratios=[r],
            max_new_tokens=max_new_tokens,
            stock_kv_first_prompt=stock_kv_first,
            instance_id=f"hf_cacheblend_phase4_blend_r{int(r*10000):05d}",
            return_segment_diagnostics=False,
        )
        f1_blend_r = f1_with_aliases(text3, case.answer, case.answer_aliases)
        cacheblend_results[cacheblend_method_name(r)] = {
            "ratio": float(r),
            "generated_text": text3,
            "generated_ids": gen3,
            "f1": f1_blend_r,
            "prefill_ms": prefill3,
        }

    methods: Dict[str, Any] = {
        METHOD_FULL_RECOMPUTE: {
            "generated_text": text1, "generated_ids": gen1,
            "f1": f1_full, "prefill_ms": prefill1,
        },
        METHOD_FULL_KV_REUSE: {
            "generated_text": text2, "generated_ids": gen2,
            "f1": f1_reuse, "prefill_ms": prefill2,
        },
    }
    methods.update(cacheblend_results)

    # ---- Generated-token comparisons (for diagnostic interpretation) ----
    comparisons: Dict[str, Any] = {}
    if cacheblend_method_name(0.15) in cacheblend_results:
        cb015 = cacheblend_results[cacheblend_method_name(0.15)]
        comparisons["reuse_eq_blend_r0.15"] = (gen2 == cb015["generated_ids"])
        comparisons["first_diff_token_reuse_vs_blend_r0.15"] = _first_diff_index(
            gen2, cb015["generated_ids"],
        )
    if cacheblend_method_name(1.00) in cacheblend_results:
        cb100 = cacheblend_results[cacheblend_method_name(1.00)]
        comparisons["full_eq_blend_r1.00"] = (gen1 == cb100["generated_ids"])
        comparisons["first_diff_token_full_vs_blend_r1.00"] = _first_diff_index(
            gen1, cb100["generated_ids"],
        )

    return {
        "id": case.id,
        "dataset": case.dataset,
        "question": case.question,
        "answer": case.answer,
        "answer_aliases": case.answer_aliases,
        "dummy_warmup_query": case.dummy_warmup_query_text,
        "real_question_segment": case.real_question_segment_text,
        "selected_chunks": [
            {
                "chunk_id": c.chunk_id,
                "paragraph_idx": c.paragraph_idx,
                "title": c.title,
                "is_supporting": c.is_supporting,
                "l2_distance_to_question": c.l2_distance_to_question,
                "token_count": len(materialized.chunk_ids_by_id[c.chunk_id]),
            }
            for c in case.selected_chunks
        ],
        "supporting_chunk_ids": [
            c.chunk_id for c in case.selected_chunks if c.is_supporting
        ],
        "retrieved_non_supporting_chunk_ids": [
            c.chunk_id for c in case.selected_chunks if not c.is_supporting
        ],
        "orders": {
            "first": case.first_order,
            "second": case.second_order,
            "policy": case.order_policy,
        },
        "tokenization": {
            **materialized.to_metadata_dict(),
            "separator_count_first": materialized.separator_count_first,
            "separator_count_second": materialized.separator_count_second,
            "first_prompt_token_length": len(materialized.first_prompt_ids),
            "second_prompt_token_length": len(materialized.second_prompt_ids),
        },
        "validation": {
            "separator_count_first": materialized.separator_count_first,
            "separator_count_second": materialized.separator_count_second,
            "orders_are_reverse": materialized.orders_are_reverse,
            "no_internal_separator": materialized.no_internal_separator,
        },
        "segment_diagnostics": seg_diag or {},
        "methods": methods,
        "generated_comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Aggregate + markdown.
# ---------------------------------------------------------------------------
def _agg(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(mean(values)), float(median(values))


def compute_failure_subset(
    per_example_records: List[Dict[str, Any]],
    method_names: List[str],
) -> Dict[str, Any]:
    """
    Examples where Full KV reuse F1 < Full recompute F1 — the subset
    where naive reuse loses quality. Aggregate every method's F1 over
    just that subset and report it.
    """
    subset_records: List[Dict[str, Any]] = []
    for rec in per_example_records:
        m = rec.get("methods", {})
        if METHOD_FULL_RECOMPUTE not in m or METHOD_FULL_KV_REUSE not in m:
            continue
        if m[METHOD_FULL_KV_REUSE]["f1"] < m[METHOD_FULL_RECOMPUTE]["f1"]:
            subset_records.append(rec)

    per_method_means: Dict[str, float] = {}
    for name in method_names:
        vals = []
        for rec in subset_records:
            if name in rec.get("methods", {}):
                vals.append(rec["methods"][name]["f1"])
        per_method_means[name] = float(mean(vals)) if vals else float("nan")

    # Best CacheBlend ratio on the subset (max F1 mean among r=... entries).
    cb_means = {n: v for n, v in per_method_means.items() if n.startswith("cacheblend_r")}
    if cb_means:
        best_name = max(cb_means, key=lambda k: (cb_means[k] if math.isfinite(cb_means[k]) else -1.0))
        best_f1 = cb_means[best_name]
    else:
        best_name, best_f1 = None, float("nan")

    return {
        "num_examples": len(subset_records),
        "per_method_f1_mean": per_method_means,
        "best_cacheblend_method": best_name,
        "best_cacheblend_f1_mean": best_f1,
    }


def _write_markdown(
    output_path: str,
    *,
    args: argparse.Namespace,
    used: int,
    skipped: int,
    skip_counts: Dict[str, int],
    per_method: Dict[str, Dict[str, List[float]]],
    method_names: List[str],
    cacheblend_ratios: List[float],
    seg_diag_summary: Dict[str, Any],
    failure_subset: Dict[str, Any],
    model_max_len: int,
    budget: int,
    bucket_per_method: Optional[Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _table_row(method_display: str, m: Dict[str, List[float]]) -> str:
        f1m, f1p = _agg(m["f1"])
        lm, lp = _agg(m["prefill_ms"])
        return (
            f"| {method_display} | {f1m:.4f} | {f1p:.4f} | "
            f"{lm:.2f} | {lp:.2f} |"
        )

    def _f1_mean(name: str) -> float:
        vals = per_method.get(name, {}).get("f1", [])
        return float(mean(vals)) if vals else float("nan")

    lines: List[str] = []
    if args.dataset == "loong":
        lines.append("# Phase 4 — Loong RAG comparison")
    else:
        lines.append("# Phase 4 — MuSiQue RAG comparison")
    lines.append("")
    lines.append(f"Model: `{args.model}`")
    lines.append(f"dtype: `{args.dtype}`")
    lines.append(f"dataset: `{args.dataset}`")
    lines.append(f"input: `{args.input_jsonl}`")
    lines.append(f"num_examples_requested: {args.num_examples}")
    lines.append(f"num_examples_used: {used}")
    lines.append(f"num_examples_skipped: {skipped}")
    lines.append(f"seed: {args.seed}")
    lines.append(f"blend_special_str: `{args.blend_special_str}`")
    if args.dataset == "musique":
        lines.append(f"embedding_model: `{args.embedding_model}`")
        lines.append(f"embedding_normalize: {args.embedding_normalize}")
    if args.dataset == "loong":
        lines.append(f"loong_num_chunks: {args.loong_num_chunks}")
        lines.append(f"loong_first_order: {args.loong_first_order}")
        lines.append(f"loong_on_extra_chunks: {args.loong_on_extra_chunks}")
        lines.append(f"loong_on_fewer_chunks: {args.loong_on_fewer_chunks}")
    lines.append(f"max_model_len: {model_max_len}")
    lines.append(f"max_new_tokens: {args.max_new_tokens}")
    lines.append(f"safety_margin: {args.safety_margin}")
    lines.append(f"prompt_token_budget: {budget}")
    lines.append(f"dummy_warmup_query: `{args.dummy_warmup_query}`")
    lines.append(f"cacheblend_recompute_ratios: `{','.join(f'{r:.2f}' for r in cacheblend_ratios)}`")
    lines.append("")

    # ----- main aggregate -----
    lines.append("| Method | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |")
    lines.append("|--------|---------|--------|-----------------|----------------|")
    lines.append(_table_row("Full recompute", per_method[METHOD_FULL_RECOMPUTE]))
    lines.append(_table_row("Full KV reuse",  per_method[METHOD_FULL_KV_REUSE]))
    # Always include r=0.15 if requested; for the rest, defer to the ratio sweep section.
    if cacheblend_method_name(0.15) in per_method:
        lines.append(_table_row("CacheBlend r=0.15", per_method[cacheblend_method_name(0.15)]))
    lines.append("")

    # ----- ratio sweep -----
    if len(cacheblend_ratios) > 1:
        lines.append("## CacheBlend ratio sweep")
        lines.append("")
        lines.append("| Ratio | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |")
        lines.append("|-------|---------|--------|-----------------|----------------|")
        for r in cacheblend_ratios:
            name = cacheblend_method_name(r)
            if name in per_method:
                f1m, f1p = _agg(per_method[name]["f1"])
                lm, lp = _agg(per_method[name]["prefill_ms"])
                lines.append(f"| r={r:.2f} | {f1m:.4f} | {f1p:.4f} | {lm:.2f} | {lp:.2f} |")
        lines.append("")

    # ----- failure-only subset -----
    lines.append("## Failure-only subset")
    lines.append("")
    lines.append(
        "Examples where Full KV reuse F1 < Full recompute F1. This is "
        "the subset where naive reuse loses quality, so it is where we "
        "expect CacheBlend's selective recomputation to help."
    )
    lines.append("")
    lines.append(f"- 갯수: {failure_subset['num_examples']}")
    lines.append(f"- Full recompute F1 mean (subset): "
                 f"{failure_subset['per_method_f1_mean'].get(METHOD_FULL_RECOMPUTE, float('nan')):.4f}")
    lines.append(f"- Full KV reuse F1 mean (subset): "
                 f"{failure_subset['per_method_f1_mean'].get(METHOD_FULL_KV_REUSE, float('nan')):.4f}")
    for r in cacheblend_ratios:
        name = cacheblend_method_name(r)
        if name in failure_subset["per_method_f1_mean"]:
            lines.append(
                f"- CacheBlend r={r:.2f} F1 mean (subset): "
                f"{failure_subset['per_method_f1_mean'][name]:.4f}"
            )
    if failure_subset["best_cacheblend_method"]:
        lines.append(
            f"- Best CacheBlend ratio (subset): "
            f"`{failure_subset['best_cacheblend_method']}` "
            f"@ {failure_subset['best_cacheblend_f1_mean']:.4f}"
        )
    lines.append("")

    # ----- prompt length buckets (Loong only) -----
    if args.dataset == "loong" and bucket_per_method:
        lines.append("## Prompt length buckets")
        lines.append("")
        lines.append("| Bucket | n | Full F1 | Reuse F1 | r=0.15 F1 | Full ms | Reuse ms | r=0.15 ms |")
        lines.append("|--------|---|---------|----------|-----------|---------|----------|-----------|")
        bucket_order = ["0-8k", "8-16k", "16-24k", "24-32k", "over_budget"]
        for bucket in bucket_order:
            if bucket not in bucket_per_method:
                continue
            b = bucket_per_method[bucket]
            n_b = len(b[METHOD_FULL_RECOMPUTE]["f1"])
            f1_full_b = mean(b[METHOD_FULL_RECOMPUTE]["f1"]) if n_b else float("nan")
            f1_reuse_b = mean(b[METHOD_FULL_KV_REUSE]["f1"]) if n_b else float("nan")
            r015 = cacheblend_method_name(0.15)
            f1_r015_b = mean(b[r015]["f1"]) if (r015 in b and b[r015]["f1"]) else float("nan")
            lm_full = mean(b[METHOD_FULL_RECOMPUTE]["prefill_ms"]) if n_b else float("nan")
            lm_reuse = mean(b[METHOD_FULL_KV_REUSE]["prefill_ms"]) if n_b else float("nan")
            lm_r015 = mean(b[r015]["prefill_ms"]) if (r015 in b and b[r015]["prefill_ms"]) else float("nan")
            lines.append(
                f"| {bucket} | {n_b} | {f1_full_b:.4f} | {f1_reuse_b:.4f} | "
                f"{f1_r015_b:.4f} | {lm_full:.1f} | {lm_reuse:.1f} | {lm_r015:.1f} |"
            )
        lines.append("")

    # ----- segment cache diagnostics -----
    lines.append("## Segment cache diagnostics")
    lines.append("")
    lines.append(f"- evaluated examples: {seg_diag_summary['num_evaluated']}")
    lines.append(f"- all chunks cache-hit: {seg_diag_summary['all_chunks_hit_count']}")
    lines.append(f"- real question segment cache-hit: {seg_diag_summary['real_question_hit_count']}")
    lines.append(f"- prefix cache-hit: {seg_diag_summary['prefix_hit_count']}")
    if seg_diag_summary["real_question_hit_count"] > 0:
        lines.append("")
        lines.append(
            "**⚠ WARNING**: real question segment was cache-hit on "
            f"{seg_diag_summary['real_question_hit_count']} example(s). "
            "This invalidates the cache-eval interpretation — the model "
            "would have seen the real question's KV reused from warmup."
        )
    lines.append("")

    # ----- sanity checks -----
    lines.append("## Sanity checks")
    lines.append("")
    f1_full = _f1_mean(METHOD_FULL_RECOMPUTE)
    f1_reuse = _f1_mean(METHOD_FULL_KV_REUSE)
    f1_r0 = _f1_mean(cacheblend_method_name(0.00)) if 0.00 in cacheblend_ratios else float("nan")
    f1_r1 = _f1_mean(cacheblend_method_name(1.00)) if 1.00 in cacheblend_ratios else float("nan")
    lines.append(f"- CacheBlend r=0.00 vs Full KV reuse F1 gap: "
                 f"{(f1_r0 - f1_reuse):+.4f}")
    lines.append(f"- CacheBlend r=1.00 vs Full recompute F1 gap: "
                 f"{(f1_r1 - f1_full):+.4f}")
    if math.isfinite(f1_r1) and math.isfinite(f1_full):
        if abs(f1_r1 - f1_full) > 0.03:
            lines.append("")
            lines.append(
                "**⚠ WARNING**: CacheBlend r=1.00 should approach Full recompute "
                "(|gap| ≤ 0.03). The observed gap exceeds 0.03 — possible causes:"
            )
            lines.append("- HKVD selection still excludes some tokens at r=1.00.")
            lines.append("- RoPE shift or layer-0 KV reuse leaves a residual signal.")
            lines.append("- Prompt materialisation difference (sep id off-by-one, BOS).")
            lines.append("- Generation seed mismatch (greedy should be deterministic).")
    if seg_diag_summary["real_question_hit_count"] > 0:
        lines.append("")
        lines.append("**⚠ WARNING**: real question segment was cache-hit (see above).")
    lines.append("")

    # ----- quality gaps & latency ratios (kept from original) -----
    lines.append("## Quality gaps (overall)")
    f1_blend_015 = _f1_mean(cacheblend_method_name(0.15)) if 0.15 in cacheblend_ratios else float("nan")
    lines.append(f"- CacheBlend r=0.15 minus Full recompute: {f1_blend_015 - f1_full:+.4f}")
    lines.append(f"- Full KV reuse minus Full recompute:     {f1_reuse - f1_full:+.4f}")
    lines.append(f"- CacheBlend r=0.15 minus Full KV reuse:  {f1_blend_015 - f1_reuse:+.4f}")
    lines.append("")

    def _safe_ratio(a: List[float], b: List[float]) -> float:
        ma = mean(a) if a else float("nan")
        mb = mean(b) if b else float("nan")
        if not math.isfinite(mb) or mb == 0:
            return float("nan")
        return ma / mb

    full_lat = per_method[METHOD_FULL_RECOMPUTE]["prefill_ms"]
    reuse_lat = per_method[METHOD_FULL_KV_REUSE]["prefill_ms"]
    lines.append("## Latency ratios (overall)")
    lines.append(f"- Full KV reuse / Full recompute: {_safe_ratio(reuse_lat, full_lat):.3f}")
    for r in cacheblend_ratios:
        name = cacheblend_method_name(r)
        if name in per_method:
            lines.append(
                f"- CacheBlend r={r:.2f} / Full recompute: "
                f"{_safe_ratio(per_method[name]['prefill_ms'], full_lat):.3f}"
            )
    lines.append("")

    # ----- skips -----
    lines.append("## Skip reasons")
    lines.append("")
    if skip_counts:
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for reason, count in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("(no skips)")
    lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main().
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    cacheblend_ratios = parse_recompute_ratios(args.cacheblend_recompute_ratios)

    if args.prefix_file is not None:
        prefix = Path(args.prefix_file).read_text(encoding="utf-8")
    elif args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = DEFAULT_PREFIX

    embedding_normalize = args.embedding_normalize.lower() in ("1", "true", "yes", "on")

    print(f"[phase4] loading tokenizer + model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()

    model.config._phase4_blend_special_str = args.blend_special_str

    # Resolve effective context budget. For Loong + Mistral-7B-v0.2 we
    # default to 32k (model's native max is 32768); for everything else
    # we fall back to whatever the config exposes.
    cfg_max = getattr(model.config, "max_position_embeddings", 4096)
    if args.max_model_len is not None:
        model_max_len = args.max_model_len
    elif args.dataset == "loong":
        model_max_len = min(32768, cfg_max) if cfg_max else 32768
    else:
        model_max_len = cfg_max
    budget = prompt_token_budget(model_max_len, args.max_new_tokens, args.safety_margin)
    print(
        f"[phase4] dataset={args.dataset} model_max_len={model_max_len} "
        f"max_new_tokens={args.max_new_tokens} safety_margin={args.safety_margin} "
        f"prompt_token_budget={budget}"
    )

    embedder = None
    if args.dataset == "musique":
        embedder = _build_embedder(args.embedding_model, args.embedding_batch_size)

    method_names: List[str] = [METHOD_FULL_RECOMPUTE, METHOD_FULL_KV_REUSE]
    for r in cacheblend_ratios:
        method_names.append(cacheblend_method_name(r))

    per_method: Dict[str, Dict[str, List[float]]] = {
        name: {"f1": [], "prefill_ms": []} for name in method_names
    }
    per_example_records: List[Dict[str, Any]] = []
    skip_counts: Dict[str, int] = {}
    used = 0
    skipped = 0
    real_question_hit_count = 0
    all_chunks_hit_count = 0
    prefix_hit_count = 0
    # Loong: per-bucket aggregates.
    bucket_per_method: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    details_fp = None
    if args.write_jsonl_details is not None:
        Path(args.write_jsonl_details).parent.mkdir(parents=True, exist_ok=True)
        details_fp = open(args.write_jsonl_details, "w", encoding="utf-8")

    try:
        if args.dataset == "musique":
            case_iter = iter_cases(
                args.input_jsonl,
                prefix_text=prefix,
                blend_special_str=args.blend_special_str,
                question_template=args.question_template,
                dummy_warmup_query=args.dummy_warmup_query,
                num_chunks=args.num_chunks,
                seed=args.seed,
                embedder=embedder,
                embedding_normalize=embedding_normalize,
                on_too_many_supporting=args.on_too_many_supporting,
            )
        elif args.dataset == "loong":
            case_iter = iter_loong_cases(
                args.input_jsonl,
                prefix_text=prefix,
                blend_special_str=args.blend_special_str,
                question_template=args.question_template,
                dummy_warmup_query=args.dummy_warmup_query,
                num_chunks=args.loong_num_chunks,
                seed=args.seed,
                first_order_policy=args.loong_first_order,
                on_extra_chunks=args.loong_on_extra_chunks,
                on_fewer_chunks=args.loong_on_fewer_chunks,
            )
        else:
            raise ValueError(f"unknown dataset {args.dataset!r}")

        for case, skip_reason in case_iter:
            if case is None:
                reason = skip_reason or "unknown"
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                skipped += 1
                continue

            try:
                materialized = materialize(
                    case, tokenizer,
                    tokenizer_name_or_path=args.model,
                    on_internal_separator=args.on_internal_separator,
                )
            except InternalSeparatorError:
                skip_counts["internal_separator_error"] = skip_counts.get("internal_separator_error", 0) + 1
                skipped += 1
                continue
            if materialized is None:
                skip_counts["internal_separator"] = skip_counts.get("internal_separator", 0) + 1
                skipped += 1
                continue

            total_first = len(materialized.first_prompt_ids)
            total_second = len(materialized.second_prompt_ids)
            # The full budget covers `prompt + max_new_tokens + safety_margin`.
            if max(total_first, total_second) > budget:
                skip_counts["prompt_too_long"] = skip_counts.get("prompt_too_long", 0) + 1
                skipped += 1
                continue

            try:
                result = _per_example(
                    model, tokenizer, materialized,
                    max_new_tokens=args.max_new_tokens,
                    cacheblend_check_layers=[args.cacheblend_check_layers],
                    cacheblend_recompute_ratios=cacheblend_ratios,
                )
            except torch.cuda.OutOfMemoryError:
                skip_counts["oom"] = skip_counts.get("oom", 0) + 1
                skipped += 1
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for name in method_names:
                if name in result["methods"]:
                    per_method[name]["f1"].append(result["methods"][name]["f1"])
                    per_method[name]["prefill_ms"].append(result["methods"][name]["prefill_ms"])
            used += 1

            seg_diag = result.get("segment_diagnostics") or {}
            if seg_diag.get("real_question_segment_cache_hit"):
                real_question_hit_count += 1
                seg_diag["invalid_example"] = True
                print(
                    f"WARN: example {case.id!r} — real question segment was a "
                    "cache hit. Marking as invalid.",
                    file=sys.stderr,
                )
            if seg_diag.get("all_six_chunks_cache_hit"):
                all_chunks_hit_count += 1
            if seg_diag.get("prefix_cache_hit"):
                prefix_hit_count += 1

            # Length bucket bookkeeping (used in the Loong markdown).
            if args.dataset == "loong":
                bucket = assign_length_bucket(total_second)
                bucket_per_method.setdefault(bucket, {
                    name: {"f1": [], "prefill_ms": []} for name in method_names
                })
                for name in method_names:
                    if name in result["methods"]:
                        bucket_per_method[bucket][name]["f1"].append(
                            result["methods"][name]["f1"]
                        )
                        bucket_per_method[bucket][name]["prefill_ms"].append(
                            result["methods"][name]["prefill_ms"]
                        )

            # Add length / budget metadata to the per-example JSONL output.
            result["prompt_lengths"] = {
                "first": total_first,
                "second": total_second,
                "max_model_len": model_max_len,
                "max_new_tokens": args.max_new_tokens,
                "safety_margin": args.safety_margin,
                "prompt_token_budget": budget,
            }
            if args.dataset == "loong":
                result["length_bucket"] = assign_length_bucket(total_second)

            if details_fp is not None:
                # Strip generated_ids out before JSONL write (they're huge).
                slim = json.loads(json.dumps(result, ensure_ascii=False))
                for name in slim.get("methods", {}):
                    slim["methods"][name].pop("generated_ids", None)
                details_fp.write(json.dumps(slim, ensure_ascii=False) + "\n")
                details_fp.flush()
            per_example_records.append(result)

            f1_log = " ".join(
                f"{name}={result['methods'][name]['f1']:.3f}"
                for name in method_names if name in result["methods"]
            )
            print(
                f"[phase4] used={used}/{args.num_examples} skipped={skipped} "
                f"id={case.id} | {f1_log}"
            )

            if used >= args.num_examples:
                break
    finally:
        if details_fp is not None:
            details_fp.close()

    seg_diag_summary = {
        "num_evaluated": used,
        "real_question_hit_count": real_question_hit_count,
        "all_chunks_hit_count": all_chunks_hit_count,
        "prefix_hit_count": prefix_hit_count,
    }
    failure_subset = compute_failure_subset(per_example_records, method_names)

    _write_markdown(
        args.output, args=args, used=used, skipped=skipped,
        skip_counts=skip_counts, per_method=per_method,
        method_names=method_names,
        cacheblend_ratios=cacheblend_ratios,
        seg_diag_summary=seg_diag_summary,
        failure_subset=failure_subset,
        model_max_len=model_max_len,
        budget=budget,
        bucket_per_method=bucket_per_method if args.dataset == "loong" else None,
    )

    print(f"[phase4] wrote markdown → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
