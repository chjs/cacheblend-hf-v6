"""
Phase 4 tests — MuSiQue prompt construction + smoke comparison.

Unit tests are CPU-only and don't need the MuSiQue dataset on disk:
they exercise chunk selection, reverse-order policy, separator count
validation, tokenizer-independence of Stage A, etc. The integration
smoke test runs the actual script under `MUSIQUE_ANS_TRAIN_JSONL` if
present (skip otherwise).
"""
from __future__ import annotations

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
    DEFAULT_PREFIX, DEFAULT_QUESTION_TEMPLATE, MusiqueCase,
    SelectedChunk, build_case, select_chunks,
)
from scripts._phase4_tokenize import (
    InternalSeparatorError, count_subsequence, materialize,
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
# Mock tokenizer — for tokenizer-related tests so we don't need to
# download anything.
# ---------------------------------------------------------------------------
class MockTokenizer:
    """
    Minimal tokenizer for Stage B tests. Each character becomes one
    token id, and `encode` prepends `[BOS]` (id=1) the way real
    SentencePiece tokenizers do. The decoder is not needed for these
    tests. The "blend special" token sequence is whatever
    `encode("# #")[1:]` returns.
    """
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        # 1 (BOS) + ord(char) for each character.
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
    assert supporting_ids == {1, 3}, (
        "both is_supporting paragraphs (idx 1, 3) must be included"
    )


# ===========================================================================
# 2. test_musique_fills_to_six_by_l2
# ===========================================================================
def test_musique_fills_to_six_by_l2():
    ex = _fake_example()
    # Mock embedder: assign known L2 distances. Make idx 2, 4, 6, 5, 0 close
    # in that priority. The 4 nearest non-supporting (after taking the 2
    # supporting) should be idx 2, 4, 6, 5.
    # Embedding shape: arbitrary 4-d; first vector is the question.
    fake_vecs = {
        # question
        "_q_": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        # non-supporting candidates (paragraph idx → distance from origin)
        0: np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        4: np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
        5: np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        6: np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    expected_order = [2, 4, 6, 5]  # the 4 nearest non-supporting, by L2

    def embedder(texts):
        out = [fake_vecs["_q_"]]
        # texts[1:] correspond to non-supporting paragraphs in their list
        # order. We need to map them back to idx; tests for that mapping:
        # the non-supporting list is [0, 2, 4, 5, 6] (in idx order).
        ns_idxs = [0, 2, 4, 5, 6]
        for idx in ns_idxs:
            out.append(fake_vecs[idx])
        return np.stack(out)

    chunks = select_chunks(ex, num_chunks=6, embedder=embedder)
    assert chunks is not None
    non_supp = [c for c in chunks if not c.is_supporting]
    assert sorted([c.paragraph_idx for c in non_supp]) == sorted(expected_order)
    # And every non-supporting chunk has a recorded L2 distance.
    for c in non_supp:
        assert c.l2_distance_to_question is not None


# ===========================================================================
# 3. test_too_many_supporting_skip_or_error
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
# 5. test_separator_count_six_chunks
# ===========================================================================
def test_separator_count_six_chunks():
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    assert case is not None
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None
    assert mat.separator_count_first == 7   # 1 after prefix + 6 chunks
    assert mat.separator_count_second == 7


# ===========================================================================
# 6. test_no_internal_separator_detection
# ===========================================================================
def test_no_internal_separator_skip():
    """If a chunk contains the separator subsequence, the example must skip."""
    # Build a case whose chunk text literally contains "# #" — that's
    # the same byte sequence the separator tokenizes to under MockTokenizer.
    paragraphs = [
        {"idx": 0, "title": "T0", "paragraph_text": "harmless",         "is_supporting": True},
        # Chunk 1's body contains the literal separator string.
        {"idx": 1, "title": "T1", "paragraph_text": "x # # y",           "is_supporting": True},
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
    out = materialize(case, tok, tokenizer_name_or_path="mock",
                      on_internal_separator="skip")
    assert out is None


def test_no_internal_separator_error():
    paragraphs = [
        {"idx": 0, "title": "T0", "paragraph_text": "harmless",         "is_supporting": True},
        {"idx": 1, "title": "T1", "paragraph_text": "x # # y",           "is_supporting": True},
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
        materialize(case, tok, tokenizer_name_or_path="mock",
                    on_internal_separator="error")


# ===========================================================================
# 7. test_tokenizer_independent_case_generation
# ===========================================================================
def test_tokenizer_independent_case_generation():
    """`build_case` must produce a MusiqueCase without ever touching a tokenizer."""
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    assert case is not None
    # MusiqueCase exposes texts, not token ids.
    assert not hasattr(case, "first_prompt_ids")
    assert not hasattr(case, "second_prompt_ids")
    # selected_chunks must carry text, not tokens.
    for c in case.selected_chunks:
        assert isinstance(c.text, str)


# ===========================================================================
# 8. test_chunk_token_ids_identical_across_orders
# ===========================================================================
def test_chunk_token_ids_identical_across_orders():
    ex = _fake_example()
    case = build_case(ex, num_chunks=6, seed=42, embedder=None)
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None

    # For every chunk id, find its position in first_prompt_ids and second_prompt_ids
    # and verify the token sub-sequences match.
    sep = mat.sep_ids
    L_sep = len(sep)

    def chunk_positions(token_ids: List[int], order: List[str]):
        # Walk: prefix, sep, chunk1, sep, chunk2, sep, ..., chunk6, sep, question.
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
# Extras: F1 sanity, count_subsequence.
# ===========================================================================
def test_f1_basics():
    assert f1_one("the whale", "whale") == pytest.approx(2 * (1/2) * (1/1) / (1/2 + 1/1))
    assert f1_with_aliases("dog", "whale", aliases=["mammal", "DOG"]) == 1.0
    assert f1_one("", "anything") == 0.0
    # Empty against empty is degenerate but defined.
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
    """End-to-end script run on a tiny slice."""
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
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, (
            f"script failed: stderr={res.stderr[-2000:]}"
        )
        assert md_out.exists() and md_out.stat().st_size > 0
        assert jsonl_out.exists() and jsonl_out.stat().st_size > 0

        # Pull F1 values and assert smoke gates.
        import json
        f1s = {"full_recompute": [], "full_kv_reuse": [], "cacheblend": []}
        with jsonl_out.open() as fp:
            for line in fp:
                rec = json.loads(line)
                for m in f1s:
                    f1s[m].append(rec["methods"][m]["f1"])
                    assert math.isfinite(rec["methods"][m]["f1"])
                    assert rec["methods"][m]["f1"] >= 0
        from statistics import mean
        f1_full = mean(f1s["full_recompute"])
        f1_blend = mean(f1s["cacheblend"])
        f1_reuse = mean(f1s["full_kv_reuse"])
        # CacheBlend within 0.05 of full recompute.
        assert abs(f1_blend - f1_full) <= 0.05, (
            f"CacheBlend F1 {f1_blend:.3f} too far from Full {f1_full:.3f}"
        )
        # Full reuse should not beat CacheBlend by more than 0.03.
        assert f1_reuse - f1_blend <= 0.03, (
            f"Full reuse F1 {f1_reuse:.3f} beats CacheBlend {f1_blend:.3f} by more than 0.03"
        )


# math import only needed by the integration smoke; quietly inert otherwise.
import math   # noqa: E402  (top of integration smoke needs it)
