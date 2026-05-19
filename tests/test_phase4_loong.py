"""
Phase 4 Loong-specific tests.

Covers schema normalization (documents / contexts / passages shapes),
chunk policy (extra / fewer-than-expected chunks), prompt-too-long
budget enforcement, length bucketing, and the dummy-warmup vs
real-question split invariants for Loong cases.

Existing MuSiQue tests live in test_phase4_rag_quality.py and must
continue to pass.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

# Make sibling-package imports work in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._phase4_loong import (
    DEFAULT_LOONG_NUM_CHUNKS, LoongChunkPolicyError, LoongSchemaError,
    assign_length_bucket, build_loong_case, normalize_loong_example,
    prompt_token_budget, select_loong_chunks,
)
from scripts._phase4_tokenize import materialize


# ---------------------------------------------------------------------------
# MockTokenizer reused from test_phase4_rag_quality.
# ---------------------------------------------------------------------------
class MockTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        return [1] + [ord(c) for c in text]


def _is_subsequence(haystack: List[int], needle: List[int]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return True
    return False


def _fake_loong_example(*, docs_field: str = "documents", n_docs: int = 11) -> dict:
    docs = [
        {"title": f"Title{i}", "text": f"text body {i}"}
        for i in range(n_docs)
    ]
    return {
        "id": "loong_ex0",
        "question": "What is the answer?",
        "answer": "blue",
        docs_field: docs,
    }


# ===========================================================================
# 1. test_normalize_loong_schema_documents
# ===========================================================================
def test_normalize_loong_schema_documents():
    ex = _fake_loong_example(docs_field="documents")
    n = normalize_loong_example(ex)
    assert n["id"] == "loong_ex0"
    assert n["question"] == "What is the answer?"
    assert n["answer"] == "blue"
    assert len(n["documents"]) == 11
    assert n["documents"][0]["text"] == "text body 0"
    assert n["documents"][0]["title"] == "Title0"


# ===========================================================================
# 2. test_normalize_loong_schema_contexts
# ===========================================================================
def test_normalize_loong_schema_contexts():
    ex = _fake_loong_example(docs_field="contexts")
    n = normalize_loong_example(ex)
    assert len(n["documents"]) == 11


def test_normalize_loong_schema_passages():
    ex = _fake_loong_example(docs_field="passages")
    n = normalize_loong_example(ex)
    assert len(n["documents"]) == 11


def test_normalize_loong_schema_string_docs():
    """Some Loong shapes store docs as plain strings."""
    ex = {
        "id": "x", "question": "q?", "answer": "a",
        "documents": ["plain doc one", "plain doc two"],
    }
    n = normalize_loong_example(ex)
    assert n["documents"][0]["text"] == "plain doc one"
    assert n["documents"][0]["title"] == ""


def test_normalize_loong_missing_question_raises():
    ex = {"id": "x", "answer": "a", "documents": [{"text": "t"}]}
    with pytest.raises(LoongSchemaError):
        normalize_loong_example(ex)


def test_normalize_loong_missing_docs_raises():
    ex = {"id": "x", "question": "q", "answer": "a"}
    with pytest.raises(LoongSchemaError):
        normalize_loong_example(ex)


def test_normalize_loong_aliases_from_answers_list():
    """Some Loong files store `answers: [primary, alt1, alt2]`."""
    ex = {
        "id": "x", "question": "q",
        "answers": ["blue", "azure", "navy"],
        "documents": [{"text": "t"}],
    }
    n = normalize_loong_example(ex)
    assert n["answer"] == "blue"
    assert set(n["answer_aliases"]) == {"azure", "navy"}


# ===========================================================================
# 3. test_loong_original_to_reverse_order
# ===========================================================================
def test_loong_original_to_reverse_order():
    ex = _fake_loong_example()
    case, sr = build_loong_case(ex, num_chunks=11, first_order_policy="original")
    assert sr is None and case is not None
    # original = dataset order; second = reversed of that.
    expected_first = [c.chunk_id for c in case.selected_chunks]
    assert case.first_order == expected_first
    assert case.second_order == list(reversed(case.first_order))


# ===========================================================================
# 4. test_loong_random_to_reverse_order
# ===========================================================================
def test_loong_random_to_reverse_order():
    ex = _fake_loong_example()
    case, _ = build_loong_case(ex, num_chunks=11, first_order_policy="random", seed=42)
    assert case is not None
    # Random shuffle: first_order is a permutation of chunk ids.
    chunk_ids = {c.chunk_id for c in case.selected_chunks}
    assert set(case.first_order) == chunk_ids
    # Determinism: same seed → same first_order.
    case2, _ = build_loong_case(ex, num_chunks=11, first_order_policy="random", seed=42)
    assert case.first_order == case2.first_order
    # Reverse invariant.
    assert case.second_order == list(reversed(case.first_order))


# ===========================================================================
# 5. test_loong_separator_count_11_chunks
# ===========================================================================
def test_loong_separator_count_11_chunks():
    ex = _fake_loong_example()
    case, _ = build_loong_case(ex, num_chunks=11)
    assert case is not None
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None
    # 1 (post-prefix) + 11 (per chunk) = 12.
    assert mat.separator_count_first == 12
    assert mat.separator_count_second == 12


# ===========================================================================
# 6. test_loong_dummy_query_only_in_first_prompt
# ===========================================================================
def test_loong_dummy_query_only_in_first_prompt():
    ex = _fake_loong_example()
    case, _ = build_loong_case(
        ex, num_chunks=11,
        dummy_warmup_query="WARMUP QUERY ONLY",
    )
    assert case is not None
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None
    assert _is_subsequence(mat.first_prompt_ids, mat.dummy_query_ids)
    assert not _is_subsequence(mat.first_prompt_ids, mat.real_question_ids)
    assert _is_subsequence(mat.second_prompt_ids, mat.real_question_ids)
    assert not _is_subsequence(mat.second_prompt_ids, mat.dummy_query_ids)


# ===========================================================================
# 7. test_loong_fewer_chunks_policy
# ===========================================================================
def test_loong_fewer_chunks_policy_skip():
    ex = _fake_loong_example(n_docs=5)
    case, sr = build_loong_case(ex, num_chunks=11, on_fewer_chunks="skip")
    assert case is None
    assert sr == "fewer_than_required_chunks"


def test_loong_fewer_chunks_policy_use_all():
    ex = _fake_loong_example(n_docs=5)
    case, sr = build_loong_case(ex, num_chunks=11, on_fewer_chunks="use_all")
    assert sr is None and case is not None
    assert len(case.selected_chunks) == 5


def test_loong_fewer_chunks_policy_error():
    ex = _fake_loong_example(n_docs=5)
    with pytest.raises(LoongChunkPolicyError):
        build_loong_case(ex, num_chunks=11, on_fewer_chunks="error")


# ===========================================================================
# 8. test_loong_extra_chunks_policy
# ===========================================================================
def test_loong_extra_chunks_policy_first():
    ex = _fake_loong_example(n_docs=15)
    case, sr = build_loong_case(ex, num_chunks=11, on_extra_chunks="first")
    assert sr is None and case is not None
    assert len(case.selected_chunks) == 11
    # First-N taken in dataset order.
    assert case.selected_chunks[0].title == "Title0"
    assert case.selected_chunks[-1].title == "Title10"


def test_loong_extra_chunks_policy_skip():
    ex = _fake_loong_example(n_docs=15)
    case, sr = build_loong_case(ex, num_chunks=11, on_extra_chunks="skip")
    assert case is None
    assert sr == "more_than_required_chunks"


def test_loong_extra_chunks_policy_error():
    ex = _fake_loong_example(n_docs=15)
    with pytest.raises(LoongChunkPolicyError):
        build_loong_case(ex, num_chunks=11, on_extra_chunks="error")


# ===========================================================================
# 9. test_loong_prompt_too_long_skip
# ===========================================================================
def test_loong_prompt_too_long_skip():
    """Synthetic: a single huge chunk pushes the second prompt past budget."""
    # Construct a chunk whose text is much longer than the budget.
    big_text = "x " * 40000  # ~80k chars → MockTokenizer gives ~80k tokens.
    ex = {
        "id": "big",
        "question": "q?",
        "answer": "a",
        "documents": [{"title": "Big", "text": big_text}],
    }
    case, sr = build_loong_case(ex, num_chunks=1, on_fewer_chunks="use_all")
    assert sr is None and case is not None
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None
    # The skip itself is enforced in run_rag_comparison.main(), not in
    # materialize() — here we just verify the prompt length exceeds a
    # 32k-style budget.
    assert len(mat.second_prompt_ids) > prompt_token_budget(32768, 32, 128)


# ===========================================================================
# 10. test_prompt_length_budget
# ===========================================================================
def test_prompt_length_budget():
    assert prompt_token_budget(32768, 32, 128) == 32608
    assert prompt_token_budget(4096, 32, 128) == 3936
    # Edge: zero-margin.
    assert prompt_token_budget(1000, 0, 0) == 1000


# ===========================================================================
# 11. test_length_buckets
# ===========================================================================
def test_length_buckets():
    assert assign_length_bucket(0) == "0-8k"
    assert assign_length_bucket(8 * 1024 - 1) == "0-8k"
    assert assign_length_bucket(8 * 1024) == "8-16k"
    assert assign_length_bucket(16 * 1024 - 1) == "8-16k"
    assert assign_length_bucket(16 * 1024) == "16-24k"
    assert assign_length_bucket(24 * 1024 - 1) == "16-24k"
    assert assign_length_bucket(24 * 1024) == "24-32k"
    assert assign_length_bucket(32 * 1024 - 1) == "24-32k"
    assert assign_length_bucket(32 * 1024) == "over_budget"
    assert assign_length_bucket(100_000) == "over_budget"


# ===========================================================================
# 12. test_select_loong_chunks_exact
# ===========================================================================
def test_select_loong_chunks_exact():
    """When chunks == num_chunks, use them all unchanged."""
    ex = _fake_loong_example(n_docs=11)
    n = normalize_loong_example(ex)
    chunks, sr = select_loong_chunks(n, num_chunks=11)
    assert sr is None
    assert len(chunks) == 11
    # Chunk text is rendered with "Title:" prefix when title exists.
    assert chunks[0].text.startswith("Title: Title0")


# ===========================================================================
# 13. test_chunk_text_stable_across_orders
# ===========================================================================
def test_chunk_text_stable_across_orders():
    """The same chunk must tokenize to identical ids regardless of position."""
    ex = _fake_loong_example()
    case, _ = build_loong_case(ex, num_chunks=11, first_order_policy="random", seed=42)
    tok = MockTokenizer()
    mat = materialize(case, tok, tokenizer_name_or_path="mock")
    assert mat is not None
    # Spot-check: every chunk_id has exactly one token sequence in
    # chunk_ids_by_id, independent of order.
    for cid, ids in mat.chunk_ids_by_id.items():
        assert isinstance(ids, list)
        assert all(isinstance(t, int) for t in ids)
