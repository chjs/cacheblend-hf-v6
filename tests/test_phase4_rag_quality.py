"""
Phase 4 tests — MuSiQue prompt construction + smoke comparison.

Unit tests are CPU-only and don't need the MuSiQue dataset on disk:
they exercise chunk selection, reverse-order policy, separator count
validation, tokenizer-independence of Stage A, dummy-warmup vs real-
question separation, recompute-ratio parser, and the failure-only
subset computation. The integration smoke test runs the actual script
under `MUSIQUE_ANS_TRAIN_JSONL` if present (skip otherwise).
"""
from __future__ import annotations

import math
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Make sibling-package imports work in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._phase4_f1 import f1_one, f1_with_aliases
from scripts._phase4_musique import (
    DEFAULT_DUMMY_WARMUP_QUERY, DEFAULT_PREFIX, DEFAULT_QUESTION_TEMPLATE,
    MusiqueCase, SelectedChunk, build_case, select_chunks,
)
from scripts._phase4_tokenize import (
    InternalSeparatorError, count_subsequence, materialize,
)
from scripts.run_rag_comparison import (
    METHOD_FULL_KV_REUSE, METHOD_FULL_RECOMPUTE,
    cacheblend_method_name, compute_failure_subset, parse_recompute_ratios,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal fake MuSiQue examples.
# ---------------------------------------------------------------------------
def _fake_example(
    *,
    id: str = "ex0",
    question: str = "Which mammal is named here?",
    answer: str = "whale",
    aliases: List[str] = (),
    paragraphs: List[dict] = None,
    answerable: bool = True,
) -> dict:
    if paragraphs is None:
        paragraphs = [
            {"idx": 0, "title": "Apples", "paragraph_text": "Apples are fruit.", "is_supporting": False},
            {"idx": 1, "title": "Whales", "paragraph_text": "Whales are mammals.", "is_supporting": True},
            {"idx": 2, "title": "Towers", "paragraph_text": "The Eiffel Tower is in Paris.", "is_supporting": False},
            {"idx": 3, "title": "Dogs",   "paragraph_text": "Dogs are loyal.",     "is_supporting": True},
            {"idx": 4, "title": "Cats",   "paragraph_text": "Cats are independent.","is_supporting": False},
            {"idx": 5, "title": "Trees",  "paragraph_text": "Trees grow tall.",   "is_supporting": False},
            {"idx": 6, "title": "Stars",  "paragraph_text": "Stars are far away.","is_supporting": False},
        ]
    return {
        "id": id, "question": question, "answer": answer,
        "answer_aliases": list(aliases), "paragraphs": paragraphs,
        "answerable": answerable,
    }


# ---------------------------------------------------------------------------
# Mock tokenizer.
# ---------------------------------------------------------------------------
class MockTokenizer:
    """Each character becomes one token id; `encode` prepends BOS=1."""
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        return [1] + [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i not in (1, 2))


# ===========================================================================
# 1. test_musique_selects_all_supporting_paragraphs
# ===========================================================================
def test_musique_selects_all_supporting_paragraphs():
    ex = _fake_example()
    chunks = select_chunks(ex, num_chunks=6, embedder=None)
    assert chunks is not None
    assert len(chunks) == 6
    supporting_ids = {c.paragraph_idx for c in chunks if c.is_supporting}
    assert supporting_ids == {1, 3}


# ===========================================================================
# 2. test_musique_fills_to_six_by_l2
# ===========================================================================
def test_musique_fills_to_six_by_l2():
    ex = _fake_example()
    fake_vecs = {
        "_q_": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        0: np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        4: np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
        5: np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        6: np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    expected_order = [2, 4, 6, 5]

    def embedder(texts):
        out = [fake_vecs["_q_"]]
        ns_idxs = [0, 2, 4, 5, 6]
        for idx in ns_idxs:
            out.append(fake_vecs[idx])
        return np.stack(out)

    chunks = select_chunks(ex, num_chunks=6, embedder=embedder)
    assert chunks is not None
    non_supp = [c for c in chunks if not c.is_supporting]
    assert sorted([c.paragraph_idx for c in non_supp]) == sorted(expected_order)
    for c in non_supp:
        assert c.l2_distance_to_question is not None


# ===========================================================================
# 3. test_too_many_supporting_skip / error
# ===========================================================================
def test_too_many_supporting_skip():
    paragraphs = [
        {"idx": i, "title": f"T{i}", "paragraph_text": "x", "is_supporting": True}
        for i in range(7)
    ]
    ex = _fake_example(paragraphs=paragraphs)
    out = select_chunks(ex, num_chunks=6, embedder=None, on_too_many_supporting="skip")
    assert out is None


def test_too_many_supporting_error():
    paragraphs = [
        {"idx": i, "title": f"T{i}", "paragraph_text": "x", "is_supporting": True}
        for i in range(7)
    ]
    ex = _fake_example(paragraphs=paragraphs)
    from scripts._phase4_musique import ChunkSelectionError
    with pytest.raises(ChunkSelectionError):
        select_chunks(ex, num_chunks=6, embedder=None, on_too_many_supporting="error")


# ===========================================================================
# 4. test_reverse_order_policy
# ===========================================================================
def test_reverse_order_policy():
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    assert case is not None
    assert case.second_order == list(reversed(case.first_order))
    assert set(case.first_order) == set(case.second_order)
    assert case.order_policy == "reverse_of_first_order"


# ===========================================================================
# 5. test_separator_count_still_seven (renamed; same intent)
# ===========================================================================
def test_separator_count_still_seven():
    """Both prompts have exactly 1 (post-prefix) + 6 (per chunk) = 7 separators."""
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    assert case is not None
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None
    assert mat.separator_count_first == 7
    assert mat.separator_count_second == 7


# ===========================================================================
# 6. test_no_internal_separator_detection
# ===========================================================================
def test_no_internal_separator_skip():
    paragraphs = [
        {"idx": 0, "title": "T0", "paragraph_text": "harmless",         "is_supporting": True},
        {"idx": 1, "title": "T1", "paragraph_text": "x # # y",          "is_supporting": True},
        {"idx": 2, "title": "T2", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 3, "title": "T3", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 4, "title": "T4", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 5, "title": "T5", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 6, "title": "T6", "paragraph_text": "harmless",         "is_supporting": False},
    ]
    ex = _fake_example(paragraphs=paragraphs)
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    assert case is not None
    tok = MockTokenizer()
    out = materialize(case, tok, tokenizer_name_or_path="mock", on_internal_separator="skip")
    assert out is None


def test_no_internal_separator_error():
    paragraphs = [
        {"idx": 0, "title": "T0", "paragraph_text": "harmless",         "is_supporting": True},
        {"idx": 1, "title": "T1", "paragraph_text": "x # # y",          "is_supporting": True},
        {"idx": 2, "title": "T2", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 3, "title": "T3", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 4, "title": "T4", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 5, "title": "T5", "paragraph_text": "harmless",         "is_supporting": False},
        {"idx": 6, "title": "T6", "paragraph_text": "harmless",         "is_supporting": False},
    ]
    ex = _fake_example(paragraphs=paragraphs)
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    tok = MockTokenizer()
    with pytest.raises(InternalSeparatorError):
        materialize(case, tok, tokenizer_name_or_path="mock", on_internal_separator="error")


# ===========================================================================
# 7. test_tokenizer_independent_case_generation
# ===========================================================================
def test_tokenizer_independent_case_generation():
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    assert case is not None
    assert not hasattr(case, "first_prompt_ids")
    assert not hasattr(case, "second_prompt_ids")
    for c in case.selected_chunks:
        assert isinstance(c.text, str)
    # MusiqueCase exposes both query texts, not token ids.
    assert isinstance(case.dummy_warmup_query_text, str)
    assert isinstance(case.real_question_segment_text, str)


# ===========================================================================
# 8. test_chunk_token_ids_identical_across_orders
# ===========================================================================
def test_chunk_token_ids_identical_across_orders():
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None

    sep = mat.sep_ids
    L_sep = len(sep)

    def chunk_positions(token_ids: List[int], order: List[str]):
        pos = len(mat.prefix_ids) + L_sep
        for cid in order:
            cids = mat.chunk_ids_by_id[cid]
            assert token_ids[pos:pos + len(cids)] == cids, (
                f"chunk {cid} mismatch in prompt at offset {pos}"
            )
            pos += len(cids) + L_sep

    chunk_positions(mat.first_prompt_ids, case.first_order)
    chunk_positions(mat.second_prompt_ids, case.second_order)


# ===========================================================================
# NEW: dummy / real question separation
# ===========================================================================
def _is_subsequence(haystack: List[int], needle: List[int]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return True
    return False


def test_first_prompt_uses_dummy_query():
    """First prompt's tail must be the dummy warmup query, not the real question."""
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None,
                      dummy_warmup_query="WARMUP QUERY DO NOT ANSWER")
    assert case is not None
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None

    assert _is_subsequence(mat.first_prompt_ids, mat.dummy_query_ids)
    assert not _is_subsequence(mat.first_prompt_ids, mat.real_question_ids)


def test_second_prompt_uses_real_question():
    """Second prompt's tail must be the real MuSiQue question segment, not the dummy."""
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None,
                      dummy_warmup_query="WARMUP QUERY DO NOT ANSWER")
    assert case is not None
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None

    assert _is_subsequence(mat.second_prompt_ids, mat.real_question_ids)
    assert not _is_subsequence(mat.second_prompt_ids, mat.dummy_query_ids)


def test_real_question_not_equal_dummy_query():
    """Default warmup query must differ from a synthetic example's real question."""
    ex = _fake_example(question="Which mammal is named here?")
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    assert case is not None
    assert case.dummy_warmup_query_text != case.real_question_segment_text
    # The default warmup query text starts with the cache-warmup sentence,
    # while the question segment uses the `Question:\nAnswer:` template.
    assert "warmup" in case.dummy_warmup_query_text.lower()
    assert "Question:" in case.real_question_segment_text
    assert ex["question"] not in case.dummy_warmup_query_text


def test_dummy_query_default_is_sentinel():
    """Sanity: the project's default dummy warmup query is the documented sentence."""
    assert DEFAULT_DUMMY_WARMUP_QUERY == "This is a cache warmup query. Do not answer."


# ===========================================================================
# NEW: ratio parser
# ===========================================================================
def test_cacheblend_ratio_parser():
    assert parse_recompute_ratios("0.0,0.05,0.15,0.30,0.50,1.00") == \
        [0.0, 0.05, 0.15, 0.30, 0.50, 1.00]
    # Order preserved even when unsorted.
    assert parse_recompute_ratios("0.5,0.1") == [0.5, 0.1]
    # Single value.
    assert parse_recompute_ratios("0.15") == [0.15]
    # Whitespace tolerated; empty fragments ignored.
    assert parse_recompute_ratios(" 0.0 , 0.15 ") == [0.0, 0.15]
    # Dedupe.
    assert parse_recompute_ratios("0.15,0.15,0.30") == [0.15, 0.30]
    # Out-of-range rejected.
    for bad in ("-0.1", "1.5", "abc", ""):
        with pytest.raises(ValueError):
            parse_recompute_ratios(bad)


# ===========================================================================
# NEW: failure-only subset computation
# ===========================================================================
def test_failure_subset_computation():
    """Synthetic per-example records; verify subset selection + per-method
    means are computed only over the subset."""
    def rec(eid: str, *, full: float, reuse: float, blend015: float, blend100: float):
        return {
            "id": eid,
            "methods": {
                METHOD_FULL_RECOMPUTE: {"f1": full,    "prefill_ms": 100.0},
                METHOD_FULL_KV_REUSE:  {"f1": reuse,   "prefill_ms": 10.0},
                cacheblend_method_name(0.15): {"f1": blend015, "prefill_ms": 20.0},
                cacheblend_method_name(1.00): {"f1": blend100, "prefill_ms": 95.0},
            },
        }

    records = [
        # ex0: reuse worse than full → goes into subset.
        rec("ex0", full=1.0, reuse=0.2, blend015=0.6, blend100=1.0),
        # ex1: reuse = full → NOT in subset.
        rec("ex1", full=0.8, reuse=0.8, blend015=0.8, blend100=0.8),
        # ex2: reuse better than full → NOT in subset.
        rec("ex2", full=0.5, reuse=0.9, blend015=0.7, blend100=0.5),
        # ex3: reuse worse than full → in subset.
        rec("ex3", full=1.0, reuse=0.0, blend015=0.5, blend100=1.0),
    ]
    methods = [
        METHOD_FULL_RECOMPUTE, METHOD_FULL_KV_REUSE,
        cacheblend_method_name(0.15), cacheblend_method_name(1.00),
    ]
    out = compute_failure_subset(records, methods)
    assert out["num_examples"] == 2  # ex0, ex3.
    # Subset means: full = (1.0 + 1.0)/2 = 1.0; reuse = (0.2 + 0.0)/2 = 0.1.
    assert out["per_method_f1_mean"][METHOD_FULL_RECOMPUTE] == pytest.approx(1.0)
    assert out["per_method_f1_mean"][METHOD_FULL_KV_REUSE] == pytest.approx(0.1)
    # CacheBlend r=0.15: (0.6 + 0.5)/2 = 0.55; r=1.00: (1.0 + 1.0)/2 = 1.0.
    assert out["per_method_f1_mean"][cacheblend_method_name(0.15)] == pytest.approx(0.55)
    assert out["per_method_f1_mean"][cacheblend_method_name(1.00)] == pytest.approx(1.0)
    # Best CacheBlend ratio is r=1.00 (1.0 > 0.55).
    assert out["best_cacheblend_method"] == cacheblend_method_name(1.00)
    assert out["best_cacheblend_f1_mean"] == pytest.approx(1.0)


def test_failure_subset_empty_subset():
    """When no example fails, subset is empty and means are nan."""
    rec = {
        "id": "ex0",
        "methods": {
            METHOD_FULL_RECOMPUTE: {"f1": 0.5, "prefill_ms": 1.0},
            METHOD_FULL_KV_REUSE:  {"f1": 0.5, "prefill_ms": 1.0},
            cacheblend_method_name(0.15): {"f1": 0.5, "prefill_ms": 1.0},
        },
    }
    out = compute_failure_subset(
        [rec],
        [METHOD_FULL_RECOMPUTE, METHOD_FULL_KV_REUSE, cacheblend_method_name(0.15)],
    )
    assert out["num_examples"] == 0
    for name in (METHOD_FULL_RECOMPUTE, METHOD_FULL_KV_REUSE, cacheblend_method_name(0.15)):
        v = out["per_method_f1_mean"][name]
        assert math.isnan(v)


# ===========================================================================
# Extras: F1 sanity, count_subsequence.
# ===========================================================================
def test_f1_basics():
    assert f1_one("the whale", "whale") == pytest.approx(2 * (1/2) * (1/1) / (1/2 + 1/1))
    assert f1_with_aliases("dog", "whale", aliases=["mammal", "DOG"]) == 1.0
    assert f1_one("", "anything") == 0.0
    assert f1_one("", "") == 1.0


def test_count_subsequence():
    assert count_subsequence([1, 2, 3, 1, 2, 3], [1, 2]) == 2
    assert count_subsequence([1, 1, 1, 1], [1, 1]) == 3
    with pytest.raises(ValueError):
        count_subsequence([1, 2], [])


# ===========================================================================
# Integration smoke (skipped unless MUSIQUE_ANS_TRAIN_JSONL set).
# ===========================================================================
MUSIQUE_PATH = os.environ.get("MUSIQUE_ANS_TRAIN_JSONL")


@pytest.mark.skipif(
    not (MUSIQUE_PATH and Path(MUSIQUE_PATH).exists() and os.environ.get("LMC_PHASE4_REAL")),
    reason=(
        "Integration smoke requires MUSIQUE_ANS_TRAIN_JSONL pointing at a "
        "downloaded MuSiQue file AND LMC_PHASE4_REAL=1. Skipping."
    ),
)
def test_integration_smoke():
    """End-to-end script run on a tiny slice — checks the script runs and
    emits the expected segment-diagnostic invariants."""
    with tempfile.TemporaryDirectory() as tmp:
        md_out = Path(tmp) / "phase4_smoke.md"
        jsonl_out = Path(tmp) / "details.jsonl"
        cmd = [
            sys.executable, "scripts/run_rag_comparison.py",
            "--model", "mistralai/Mistral-7B-Instruct-v0.2",
            "--input-jsonl", MUSIQUE_PATH,
            "--num-examples", "5",
            "--dtype", "bfloat16",
            "--output", str(md_out),
            "--write-jsonl-details", str(jsonl_out),
            "--cacheblend-recompute-ratios", "0.0,0.15,1.00",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"script failed: stderr={res.stderr[-2000:]}"
        assert md_out.exists() and md_out.stat().st_size > 0
        assert jsonl_out.exists() and jsonl_out.stat().st_size > 0

        import json
        records = []
        with jsonl_out.open() as fp:
            for line in fp:
                records.append(json.loads(line))
        assert records, "no per-example records emitted"

        for rec in records:
            seg = rec.get("segment_diagnostics", {})
            # Cache-hit invariant: real question segment must MISS.
            # If it doesn't, this run is invalid as a RAG-cache test.
            assert seg.get("real_question_segment_cache_hit") in (False, None), (
                f"example {rec.get('id')}: real question segment was a cache HIT — "
                "this defeats the cache-eval intent"
            )
            # Every CacheBlend ratio key must be present.
            for r in (0.0, 0.15, 1.00):
                name = cacheblend_method_name(r)
                assert name in rec["methods"], f"missing method {name}"
                assert math.isfinite(rec["methods"][name]["f1"])
                assert rec["methods"][name]["f1"] >= 0
