"""
Phase 4 Stage A — MuSiQue Answerable case construction.

Tokenizer-independent: reads `musique_ans_v1.0_train.jsonl`, selects
exactly N chunks per example (all `is_supporting` paragraphs + nearest
non-supporting by embedding L2), shuffles once with deterministic
randomness, and produces a `MusiqueCase` dataclass containing only
text + chunk metadata. Token id materialisation is Stage B's job
(`_phase4_tokenize.py`).

Design echoes docs/LMCACHE_IMPLEMENTATION.md §2.7's SegmentTokenDatabase
semantics: each chunk is *its own segment* — we never tokenize the
full concatenated prompt and split afterwards.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------
@dataclass
class SelectedChunk:
    chunk_id: str
    paragraph_idx: int
    title: str
    text: str                       # rendered chunk text (Title:\n\n{paragraph_text})
    is_supporting: bool
    l2_distance_to_question: Optional[float]


@dataclass
class MusiqueCase:
    """Tokenizer-independent CacheBlend test case for one MuSiQue example.

    The first prompt is a *warmup-only* prompt: it ends with
    `dummy_warmup_query_text` instead of the real MuSiQue question, so
    that the question segment does NOT get cached under a key that the
    second prompt would hit. The second prompt ends with
    `real_question_segment_text` (the actual MuSiQue question) and is
    the *only* prompt scored for F1.
    """
    id: str
    dataset: str
    question: str
    answer: str
    answer_aliases: List[str]
    prefix_text: str
    blend_special_str: str
    real_question_segment_text: str      # second prompt's final segment
    dummy_warmup_query_text: str         # first prompt's final segment
    selected_chunks: List[SelectedChunk]
    first_order: List[str]          # chunk_ids in first-prompt order
    second_order: List[str]         # chunk_ids in second-prompt order
    order_policy: str = "reverse_of_first_order"

    def __post_init__(self):
        # Invariant: second_order is exactly the reverse of first_order.
        assert self.second_order == list(reversed(self.first_order)), (
            f"second_order must be reverse of first_order; "
            f"first={self.first_order} second={self.second_order}"
        )
        # Same id set.
        assert set(self.first_order) == set(self.second_order)
        # The warmup query must differ from the real question, otherwise
        # the real question segment would be a cache hit (defeats the
        # cache-eval intent).
        assert (
            self.dummy_warmup_query_text != self.real_question_segment_text
        ), (
            "dummy_warmup_query_text must differ from real_question_segment_text "
            "so the real-question segment is a cache MISS"
        )

    def chunk_by_id(self, cid: str) -> SelectedChunk:
        for c in self.selected_chunks:
            if c.chunk_id == cid:
                return c
        raise KeyError(cid)


# ---------------------------------------------------------------------------
# Default templates.
# ---------------------------------------------------------------------------
DEFAULT_PREFIX = (
    # MuSiQue gold answers are mostly short factoid spans; long-form
    # explanations from the model would tank F1 precision because of
    # the extra tokens. We instruct for a short direct answer but
    # deliberately avoid hard word-count constraints like "exactly 3
    # words" — some valid MuSiQue answers exceed 3 tokens and a hard
    # cap would clip them.
    "You are a question-answering assistant. Use the provided passages "
    "to answer the final question. Answer with only the final answer. "
    "Use the shortest possible phrase. Do not explain."
)

DEFAULT_QUESTION_TEMPLATE = "Question: {question}\n\nAnswer:"

DEFAULT_CHUNK_TEMPLATE = "Title: {title}\n\n{paragraph_text}"

# The first prompt is a *warmup-only* prompt — it populates the
# per-chunk KV cache but is not used for F1. The trailing segment must
# differ from the real MuSiQue question so the real question segment
# of the second prompt is NOT a cache hit. This sentinel-style query
# does that and tells the model not to answer.
DEFAULT_DUMMY_WARMUP_QUERY = "This is a cache warmup query. Do not answer."


# ---------------------------------------------------------------------------
# JSONL iteration.
# ---------------------------------------------------------------------------
def iter_musique_jsonl(path: str) -> Iterator[dict]:
    """Yield one JSON object per non-empty line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Chunk selection.
# ---------------------------------------------------------------------------
def _render_chunk(title: str, paragraph_text: str) -> str:
    return DEFAULT_CHUNK_TEMPLATE.format(
        title=title.strip(), paragraph_text=paragraph_text.strip(),
    )


def _chunk_id_for(example_id: str, paragraph_idx: int) -> str:
    return f"{example_id}::paragraph::{paragraph_idx}"


class ChunkSelectionError(Exception):
    """Raised when chunk selection would emit a non-skip error condition."""


def select_chunks(
    example: dict,
    *,
    num_chunks: int = 6,
    embedder=None,
    embedding_normalize: bool = False,
    on_too_many_supporting: str = "skip",
) -> Optional[List[SelectedChunk]]:
    """
    Pick exactly `num_chunks` SelectedChunks for one MuSiQue example.

    Returns None when the example should be skipped.

    `embedder` is any callable taking `list[str]` and returning a
    numpy array of shape `(n, d)`. If it is None, the function falls
    back to a deterministic order-based "embedding" that lets unit
    tests run without sentence-transformers.
    """
    paragraphs: List[dict] = example.get("paragraphs", []) or []
    if not paragraphs:
        return None

    supporting = [p for p in paragraphs if p.get("is_supporting")]
    non_supporting = [p for p in paragraphs if not p.get("is_supporting")]

    if len(supporting) > num_chunks:
        if on_too_many_supporting == "error":
            raise ChunkSelectionError(
                f"example {example.get('id')!r} has "
                f"{len(supporting)} > {num_chunks} supporting paragraphs"
            )
        return None

    selected: List[SelectedChunk] = []
    for p in supporting:
        selected.append(SelectedChunk(
            chunk_id=_chunk_id_for(example["id"], p["idx"]),
            paragraph_idx=int(p["idx"]),
            title=str(p.get("title", "")),
            text=_render_chunk(p.get("title", ""), p.get("paragraph_text", "")),
            is_supporting=True,
            l2_distance_to_question=None,
        ))

    remaining = num_chunks - len(selected)
    if remaining > 0:
        if len(non_supporting) < remaining:
            # Not enough non-supporting paragraphs to fill out 6 chunks.
            return None
        question_text = str(example["question"])
        non_supp_texts = [
            _render_chunk(p.get("title", ""), p.get("paragraph_text", ""))
            for p in non_supporting
        ]
        distances = _question_paragraph_l2(
            question_text, non_supp_texts,
            embedder=embedder, normalize=embedding_normalize,
        )
        # ascending L2; tie-break by ascending paragraph idx.
        order = sorted(
            range(len(non_supporting)),
            key=lambda i: (distances[i], int(non_supporting[i]["idx"])),
        )
        for i in order[:remaining]:
            p = non_supporting[i]
            selected.append(SelectedChunk(
                chunk_id=_chunk_id_for(example["id"], p["idx"]),
                paragraph_idx=int(p["idx"]),
                title=str(p.get("title", "")),
                text=_render_chunk(p.get("title", ""), p.get("paragraph_text", "")),
                is_supporting=False,
                l2_distance_to_question=float(distances[i]),
            ))

    assert len(selected) == num_chunks
    return selected


def _question_paragraph_l2(
    question: str,
    paragraphs: Sequence[str],
    *,
    embedder=None,
    normalize: bool = False,
) -> List[float]:
    """
    L2 distance between question embedding and each paragraph embedding.
    Falls back to a deterministic "fake embedding" if `embedder` is
    None — useful for unit tests that don't have sentence-transformers
    installed.
    """
    if not paragraphs:
        return []

    if embedder is None:
        # Deterministic fallback: hash-derived 8-d vector per text.
        # Stable across runs (Python builtin `hash` honours PYTHONHASHSEED).
        def fake_embed(text: str) -> np.ndarray:
            h = hash(text)
            rng = np.random.default_rng(abs(h) & 0xFFFFFFFF)
            return rng.standard_normal(8).astype(np.float32)

        q_vec = fake_embed(question)
        p_vecs = np.stack([fake_embed(p) for p in paragraphs])
    else:
        all_vecs = embedder([question, *paragraphs])
        q_vec = np.asarray(all_vecs[0], dtype=np.float32)
        p_vecs = np.asarray(all_vecs[1:], dtype=np.float32)

    if normalize:
        def _l2norm(x: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(x, axis=-1, keepdims=True)
            n[n == 0] = 1.0
            return x / n
        q_vec = _l2norm(q_vec[None])[0]
        p_vecs = _l2norm(p_vecs)

    diffs = p_vecs - q_vec[None]
    return list(np.linalg.norm(diffs, axis=-1).astype(float))


# ---------------------------------------------------------------------------
# Case construction (Stage A entry point).
# ---------------------------------------------------------------------------
def build_case(
    example: dict,
    *,
    prefix_text: str = DEFAULT_PREFIX,
    blend_special_str: str = "# #",
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
    dummy_warmup_query: str = DEFAULT_DUMMY_WARMUP_QUERY,
    num_chunks: int = 6,
    seed: int = 42,
    embedder=None,
    embedding_normalize: bool = False,
    on_too_many_supporting: str = "skip",
) -> Optional[MusiqueCase]:
    """Stage A: build a tokenizer-independent MusiqueCase.

    Returns None when the example must be skipped.
    """
    if example.get("answerable") is False:
        return None
    chunks = select_chunks(
        example,
        num_chunks=num_chunks,
        embedder=embedder,
        embedding_normalize=embedding_normalize,
        on_too_many_supporting=on_too_many_supporting,
    )
    if chunks is None:
        return None
    if len(chunks) != num_chunks:
        return None

    # Deterministic per-example random order.
    rng = random.Random(f"{seed}:{example['id']}")
    permuted = list(chunks)
    rng.shuffle(permuted)
    first_order = [c.chunk_id for c in permuted]
    second_order = list(reversed(first_order))

    real_question_segment_text = question_template.format(
        question=example["question"].strip(),
    )

    if dummy_warmup_query == real_question_segment_text:
        # Extremely defensive: a user could pass --dummy-warmup-query that
        # collides with this example's real question template. Skip the
        # example rather than corrupting the cache-hit semantics.
        return None

    return MusiqueCase(
        id=str(example["id"]),
        dataset="musique_ans_v1.0_train",
        question=str(example["question"]),
        answer=str(example["answer"]),
        answer_aliases=list(example.get("answer_aliases") or []),
        prefix_text=prefix_text,
        blend_special_str=blend_special_str,
        real_question_segment_text=real_question_segment_text,
        dummy_warmup_query_text=dummy_warmup_query,
        selected_chunks=chunks,
        first_order=first_order,
        second_order=second_order,
    )


def iter_cases(
    path: str,
    *,
    embedder=None,
    **kwargs,
) -> Iterator[tuple[Optional[MusiqueCase], Optional[str]]]:
    """
    Yield `(case, skip_reason)` per example. `case` is None when the
    example was skipped; `skip_reason` is a short tag describing why.
    """
    for ex in iter_musique_jsonl(path):
        if ex.get("answerable") is False:
            yield None, "not_answerable"
            continue
        try:
            case = build_case(ex, embedder=embedder, **kwargs)
        except ChunkSelectionError as e:
            yield None, f"chunk_selection_error: {e}"
            continue
        if case is None:
            # Most common case: too many supporting paragraphs or
            # not enough non-supporting paragraphs to fill 6.
            if len(ex.get("paragraphs", [])) == 0:
                yield None, "no_paragraphs"
            else:
                supporting = sum(1 for p in ex["paragraphs"] if p.get("is_supporting"))
                num_chunks = kwargs.get("num_chunks", 6)
                if supporting > num_chunks:
                    yield None, f"too_many_supporting:{supporting}"
                else:
                    yield None, "insufficient_non_supporting"
            continue
        yield case, None
