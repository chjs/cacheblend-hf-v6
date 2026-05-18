"""
Phase 4 — MuSiQue RAG quality comparison for CacheBlend-HF.

Compares three prefill methods on MuSiQue Answerable:
  1. Full KV recompute (stock HF prefill on the second prompt).
  2. Full KV reuse (Prompt Cache baseline: cached chunk KVs + RoPE
     shift, no recomputation).
  3. CacheBlend (Phase 3 LMCBlender, check_layers=[1], recomp_ratios=[0.15]).

For each evaluated example the script:
  - generates an answer greedily (do_sample=False)
  - scores token-level F1 against the gold answer + aliases
  - measures prefill latency
and writes an aggregate markdown + optional per-example JSONL.

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
from scripts._phase4_musique import (             # noqa: E402
    DEFAULT_PREFIX, DEFAULT_QUESTION_TEMPLATE, MusiqueCase, iter_cases,
)
from scripts._phase4_runtime import (             # noqa: E402
    connector_to_dyncache, decode_from_full_cache, full_recompute_run,
    unpatch_hf_model,
)
from scripts._phase4_tokenize import (            # noqa: E402
    InternalSeparatorError, MaterializedCase, materialize,
)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 — MuSiQue RAG comparison")
    p.add_argument("--model", required=True, choices=[
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ])
    p.add_argument("--input-jsonl", required=True,
                   help="Path to musique_ans_v1.0_train.jsonl")
    p.add_argument("--num-examples", type=int, required=True,
                   help="Maximum number of MuSiQue examples to USE (after skips).")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
    p.add_argument("--output", required=True,
                   help="Path to write the aggregate markdown report.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--num-chunks", type=int, default=6)
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
    p.add_argument("--on-too-many-supporting", choices=["skip", "error"], default="skip")
    p.add_argument("--on-internal-separator", choices=["skip", "error"], default="skip")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="If set, override the model's max position embedding for the "
                        "prompt-fits-in-context check.")
    p.add_argument("--write-jsonl-details", default=None,
                   help="Optional path for per-example JSONL details.")
    p.add_argument("--cacheblend-check-layers", type=int, default=1)
    p.add_argument("--cacheblend-recomp-ratio", type=float, default=0.15)

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
    # Build directly (skip the LMCBlenderBuilder cache so two different
    # blender classes can both be built within one run).
    blender = blender_cls(
        cache_engine=engine, gpu_connector=connector, hf_model=model, config=cfg,
    )
    return engine, connector, blender


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
) -> Tuple[str, float, List[int]]:
    """
    Run either Method 2 (Full reuse) or Method 3 (CacheBlend) on the
    second prompt. Returns `(generated_text, prefill_ms, generated_ids)`.

    Assumes `model` is UNPATCHED at entry; this function patches it
    (via building the blender pipeline) and unpatches before decode.
    """
    device = next(model.parameters()).device

    # Build the pipeline (this patches the model in place via
    # LMCBaseModel.__init__).
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

    # ----- Warmup (no F1 measured) -----
    # `stock_kv_first_prompt` was captured by a stock HF prefill on the
    # FIRST prompt (`_per_example._stock_prefill(model, first_ids)`).
    # We stash those per-chunk KVs into the LocalCPUBackend, keyed by
    # each chunk's hash. The first prompt is never generated from.
    engine.store_from_prefill(first_prompt_ids, stock_kv_first_prompt)

    # ----- Evaluation prefill on the SECOND prompt -----
    # `blender.blend(...)` runs the layerwise prefill on the second
    # prompt, reusing the cached chunk KVs (RoPE-shifted into the
    # second prompt's chunk positions). The resulting fused KV is
    # fed to the decoder to generate the answer that gets F1-scored.
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    blender.blend(second_prompt_ids)
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

    # Important: model must be UNPATCHED for the manual decode loop to
    # work (the patched o_proj / RMSNorm wrappers break stock forward).
    unpatch_hf_model(model)

    text, gen_ids = decode_from_full_cache(
        model, tokenizer, second_prompt_ids, fused_cache, max_new_tokens,
    )

    # Release the connector's GPU buffers and the fused cache before
    # the next example.
    connector.reset()
    del connector, engine, blender, fused_cache
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    return text, prefill_ms, gen_ids


# ---------------------------------------------------------------------------
# Per-example driver (all three methods + bookkeeping).
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


def _per_example(
    model,
    tokenizer,
    materialized: MaterializedCase,
    *,
    max_new_tokens: int,
    cacheblend_check_layers: List[int],
    cacheblend_recomp_ratios: List[float],
) -> Dict[str, Any]:
    """
    Run all three methods on **the second prompt** of one example.

    Role split (made explicit by user request):
      - `first_prompt_ids`  → **warmup only**. Stock HF prefill on this
        prompt populates `LMCacheEngine` with per-chunk KV under the
        chunks' hashes. No F1 is computed on this prompt.
      - `second_prompt_ids` → **evaluation only**. All three methods
        (Full recompute / Full KV reuse / CacheBlend) generate from
        this prompt; F1 vs gold answer is computed on those generations.

    The two prompts share prefix / separator / per-chunk token sequence /
    question. The only thing that differs is chunk order:
    `second_order = reversed(first_order)`. So cached chunk KVs from the
    first prompt are reused at *different positions* in the second
    prompt (the GPU connector applies `FusedRope` to shift K). The
    quality of that position-shifted reuse is what F1 reflects.
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
    # Model is unpatched. F1 reference value for this example.
    text1, prefill1, gen1 = full_recompute_run(model, tokenizer, second_ids, max_new_tokens)
    f1_full = f1_with_aliases(text1, case.answer, case.answer_aliases)

    # ---- Warmup (no F1 measured): stock prefill on the FIRST prompt. ----
    # Captures per-layer K, V to seed the cache. Methods 2 and 3 reuse
    # these chunk KVs (shifted to the second prompt's chunk positions
    # via FusedRope inside the GPU connector). The first prompt is never
    # generated from and never F1-scored.
    stock_kv_first = _stock_prefill(model, first_ids)

    # ---- Method 2 (evaluation): Full KV reuse on the second prompt. ----
    # Uses warmup KV from the first prompt; skips HKVD recomputation.
    text2, prefill2, gen2 = _run_cache_method(
        model, tokenizer, materialized,
        use_full_reuse=True,
        check_layers=cacheblend_check_layers,
        recomp_ratios=cacheblend_recomp_ratios,
        max_new_tokens=max_new_tokens,
        stock_kv_first_prompt=stock_kv_first,
        instance_id="hf_cacheblend_phase4_full_reuse",
    )
    f1_reuse = f1_with_aliases(text2, case.answer, case.answer_aliases)

    # ---- Method 3 (evaluation): CacheBlend on the second prompt. ----
    # Uses the same warmup KV, plus HKVD recomputation of 15% tokens.
    text3, prefill3, gen3 = _run_cache_method(
        model, tokenizer, materialized,
        use_full_reuse=False,
        check_layers=cacheblend_check_layers,
        recomp_ratios=cacheblend_recomp_ratios,
        max_new_tokens=max_new_tokens,
        stock_kv_first_prompt=stock_kv_first,
        instance_id="hf_cacheblend_phase4_blend",
    )
    f1_blend = f1_with_aliases(text3, case.answer, case.answer_aliases)

    return {
        "id": case.id,
        "dataset": case.dataset,
        "question": case.question,
        "answer": case.answer,
        "answer_aliases": case.answer_aliases,
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
        "tokenization": materialized.to_metadata_dict(),
        "prompt_lengths": {
            "first": len(materialized.first_prompt_ids),
            "second": len(materialized.second_prompt_ids),
        },
        "validation": {
            "separator_count_first": materialized.separator_count_first,
            "separator_count_second": materialized.separator_count_second,
            "orders_are_reverse": materialized.orders_are_reverse,
            "no_internal_separator": materialized.no_internal_separator,
        },
        "methods": {
            "full_recompute": {
                "generated_text": text1, "f1": f1_full, "prefill_ms": prefill1,
            },
            "full_kv_reuse": {
                "generated_text": text2, "f1": f1_reuse, "prefill_ms": prefill2,
            },
            "cacheblend": {
                "generated_text": text3, "f1": f1_blend, "prefill_ms": prefill3,
            },
        },
    }


# ---------------------------------------------------------------------------
# Aggregate + markdown.
# ---------------------------------------------------------------------------
def _agg(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(mean(values)), float(median(values))


def _write_markdown(
    output_path: str,
    *,
    args: argparse.Namespace,
    used: int,
    skipped: int,
    skip_counts: Dict[str, int],
    per_method: Dict[str, Dict[str, List[float]]],
) -> None:
    """Per the spec; numbers are pre-aggregated by caller."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _table_row(method_name: str, m: Dict[str, List[float]]) -> str:
        f1m, f1p = _agg(m["f1"])
        lm, lp = _agg(m["prefill_ms"])
        return (
            f"| {method_name} | {f1m:.4f} | {f1p:.4f} | "
            f"{lm:.2f} | {lp:.2f} |"
        )

    lines: List[str] = []
    lines.append("# Phase 4 — MuSiQue RAG comparison")
    lines.append("")
    lines.append(f"Model: `{args.model}`")
    lines.append(f"dtype: `{args.dtype}`")
    lines.append(f"input: `{args.input_jsonl}`")
    lines.append(f"num_examples_requested: {args.num_examples}")
    lines.append(f"num_examples_used: {used}")
    lines.append(f"num_examples_skipped: {skipped}")
    lines.append(f"seed: {args.seed}")
    lines.append(f"blend_special_str: `{args.blend_special_str}`")
    lines.append(f"embedding_model: `{args.embedding_model}`")
    lines.append(f"embedding_normalize: {args.embedding_normalize}")
    lines.append(f"max_new_tokens: {args.max_new_tokens}")
    lines.append("")
    lines.append("| Method | F1 mean | F1 p50 | Prefill ms mean | Prefill ms p50 |")
    lines.append("|--------|---------|--------|-----------------|----------------|")
    lines.append(_table_row("Full recompute", per_method["full_recompute"]))
    lines.append(_table_row("Full KV reuse",  per_method["full_kv_reuse"]))
    lines.append(_table_row("CacheBlend",     per_method["cacheblend"]))
    lines.append("")
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
    lines.append("## Quality gaps")
    f1_full_m = mean(per_method["full_recompute"]["f1"]) if per_method["full_recompute"]["f1"] else float("nan")
    f1_reuse_m = mean(per_method["full_kv_reuse"]["f1"]) if per_method["full_kv_reuse"]["f1"] else float("nan")
    f1_blend_m = mean(per_method["cacheblend"]["f1"]) if per_method["cacheblend"]["f1"] else float("nan")
    lines.append(f"- CacheBlend minus Full recompute: {f1_blend_m - f1_full_m:+.4f}")
    lines.append(f"- Full KV reuse minus Full recompute: {f1_reuse_m - f1_full_m:+.4f}")
    lines.append(f"- CacheBlend minus Full KV reuse: {f1_blend_m - f1_reuse_m:+.4f}")
    lines.append("")
    # Latency ratios.
    def _safe_ratio(a: List[float], b: List[float]) -> float:
        ma = mean(a) if a else float("nan")
        mb = mean(b) if b else float("nan")
        if not math.isfinite(mb) or mb == 0:
            return float("nan")
        return ma / mb
    lines.append("## Latency ratios")
    lines.append(f"- Full KV reuse / Full recompute: {_safe_ratio(per_method['full_kv_reuse']['prefill_ms'], per_method['full_recompute']['prefill_ms']):.3f}")
    lines.append(f"- CacheBlend / Full recompute:   {_safe_ratio(per_method['cacheblend']['prefill_ms'], per_method['full_recompute']['prefill_ms']):.3f}")
    lines.append(f"- CacheBlend / Full KV reuse:    {_safe_ratio(per_method['cacheblend']['prefill_ms'], per_method['full_kv_reuse']['prefill_ms']):.3f}")
    lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main().
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Resolve prefix text.
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

    # Cache the blend_special_str in the model's config so the pipeline
    # builder can read it without threading args through.
    model.config._phase4_blend_special_str = args.blend_special_str

    model_max_len = args.max_model_len or getattr(model.config, "max_position_embeddings", 4096)

    # Build the embedding model. Falls back to a deterministic
    # hash-derived "embedding" if sentence-transformers isn't installed.
    embedder = _build_embedder(args.embedding_model, args.embedding_batch_size)

    per_method = {
        "full_recompute": {"f1": [], "prefill_ms": []},
        "full_kv_reuse":  {"f1": [], "prefill_ms": []},
        "cacheblend":     {"f1": [], "prefill_ms": []},
    }
    skip_counts: Dict[str, int] = {}
    used = 0
    skipped = 0
    details_fp = None
    if args.write_jsonl_details is not None:
        Path(args.write_jsonl_details).parent.mkdir(parents=True, exist_ok=True)
        details_fp = open(args.write_jsonl_details, "w", encoding="utf-8")

    try:
        case_iter = iter_cases(
            args.input_jsonl,
            prefix_text=prefix,
            blend_special_str=args.blend_special_str,
            question_template=args.question_template,
            num_chunks=args.num_chunks,
            seed=args.seed,
            embedder=embedder,
            embedding_normalize=embedding_normalize,
            on_too_many_supporting=args.on_too_many_supporting,
        )

        for case, skip_reason in case_iter:
            if case is None:
                reason = skip_reason or "unknown"
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                skipped += 1
                continue

            # Stage B: tokenize.
            try:
                materialized = materialize(
                    case, tokenizer,
                    tokenizer_name_or_path=args.model,
                    on_internal_separator=args.on_internal_separator,
                )
            except InternalSeparatorError as e:
                skip_counts["internal_separator_error"] = skip_counts.get("internal_separator_error", 0) + 1
                skipped += 1
                continue
            if materialized is None:
                skip_counts["internal_separator"] = skip_counts.get("internal_separator", 0) + 1
                skipped += 1
                continue

            # Prompt-fits-in-context check.
            total_first = len(materialized.first_prompt_ids)
            total_second = len(materialized.second_prompt_ids)
            if max(total_first, total_second) + args.max_new_tokens > model_max_len:
                skip_counts["prompt_too_long"] = skip_counts.get("prompt_too_long", 0) + 1
                skipped += 1
                continue

            # Run all three methods.
            try:
                result = _per_example(
                    model, tokenizer, materialized,
                    max_new_tokens=args.max_new_tokens,
                    cacheblend_check_layers=[args.cacheblend_check_layers],
                    cacheblend_recomp_ratios=[args.cacheblend_recomp_ratio],
                )
            except torch.cuda.OutOfMemoryError as e:  # noqa: F841
                skip_counts["oom"] = skip_counts.get("oom", 0) + 1
                skipped += 1
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for name in ("full_recompute", "full_kv_reuse", "cacheblend"):
                per_method[name]["f1"].append(result["methods"][name]["f1"])
                per_method[name]["prefill_ms"].append(result["methods"][name]["prefill_ms"])
            used += 1

            if details_fp is not None:
                details_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                details_fp.flush()

            print(
                f"[phase4] used={used}/{args.num_examples} skipped={skipped} "
                f"f1: full={result['methods']['full_recompute']['f1']:.3f} "
                f"reuse={result['methods']['full_kv_reuse']['f1']:.3f} "
                f"blend={result['methods']['cacheblend']['f1']:.3f} "
                f"id={case.id}"
            )

            if used >= args.num_examples:
                break
    finally:
        if details_fp is not None:
            details_fp.close()

    _write_markdown(
        args.output, args=args, used=used, skipped=skipped,
        skip_counts=skip_counts, per_method=per_method,
    )
    print(f"[phase4] wrote markdown → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
