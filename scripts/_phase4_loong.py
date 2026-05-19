"""
Phase 4 Stage A — Loong dataset case construction.

Loong is an extended multi-document QA benchmark with ~11 documents per
example. Unlike MuSiQue (which uses supporting-paragraph + L2 retrieval
to pick 6 chunks), Loong uses *all* provided documents as chunks
because the benchmark assumes every document may carry partial answer
signal.

Layered design — Stage A is tokenizer-independent. It reads Loong JSONL,
normalizes the example schema (Loong files vary; `documents` /
`contexts` / `passages` / `docs` / `chunks` all observed in the wild),
picks N chunks per the chunk-count policy, and emits a
`MusiqueCase`-compatible dataclass with `dataset="loong"`. Stage B
(`scripts/_phase4_tokenize.py:materialize`) is fully dataset-agnostic
and is reused unchanged.
"""
from __future__ import annotations

import json
import random
from typing import Iterator, List, Optional, Tuple

from scripts._phase4_musique import (
    DEFAULT_DUMMY_WARMUP_QUERY, DEFAULT_PREFIX, DEFAULT_QUESTION_TEMPLATE,
    MusiqueCase, SelectedChunk,
)


# ---------------------------------------------------------------------------
# Schema normalization — Loong JSONL files vary, so the field names
# we extract from are searched in priority order.
# ---------------------------------------------------------------------------
_QUESTION_KEYS = ("question", "query", "input", "prompt")
_ANSWER_KEYS = ("answer", "gold_answer", "ground_truth", "output", "label")
_ANSWERS_LIST_KEYS = ("answers", "gold_answers", "ground_truths", "labels")
_ALIASES_KEYS = ("answer_aliases", "aliases", "alternative_answers")
_DOCS_KEYS = ("documents", "docs", "context", "contexts", "passages", "chunks")
_ID_KEYS = ("id", "_id", "example_id", "question_id", "qid")
_DOC_TEXT_KEYS = ("text", "content", "document", "passage", "body", "doc")
_DOC_TITLE_KEYS = ("title", "name", "heading", "doc_title")
_DOC_ID_KEYS = ("id", "_id", "doc_id", "document_id", "passage_id")


class LoongSchemaError(ValueError):
    """Raised when the Loong example schema cannot be unambiguously normalized."""


def _first_present(d: dict, keys: Tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _coerce_to_str(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # Common pattern: {"text": "..."} wrapped.
        for k in _DOC_TEXT_KEYS:
            if k in x and isinstance(x[k], str):
                return x[k]
    if isinstance(x, list) and x and isinstance(x[0], str):
        return "\n".join(x)
    return str(x)


def _normalize_doc(doc, idx: int, example_id: str) -> dict:
    """Extract (chunk_id, title, text) from a Loong document.

    Handles dict-with-title-text, plain-string, and list-of-strings shapes.
    """
    if isinstance(doc, str):
        return {
            "chunk_id": f"{example_id}::document::{idx}",
            "paragraph_idx": idx,
            "title": "",
            "text": doc,
        }
    if not isinstance(doc, dict):
        raise LoongSchemaError(
            f"unsupported document shape {type(doc).__name__} at index {idx}"
        )
    text = _first_present(doc, _DOC_TEXT_KEYS)
    if text is None:
        raise LoongSchemaError(
            f"document at index {idx} has no text-bearing field. "
            f"keys: {sorted(doc.keys())}"
        )
    title = _first_present(doc, _DOC_TITLE_KEYS) or ""
    doc_id = _first_present(doc, _DOC_ID_KEYS) or f"document_{idx}"
    return {
        "chunk_id": f"{example_id}::{doc_id}",
        "paragraph_idx": idx,
        "title": _coerce_to_str(title).strip(),
        "text": _coerce_to_str(text).strip(),
    }


def normalize_loong_example(raw: dict) -> dict:
    """Convert a raw Loong example to a uniform dict.

    Returned schema:
        {
          "id": str,
          "question": str,
          "answer": str,
          "answer_aliases": List[str],
          "documents": List[{"chunk_id","paragraph_idx","title","text"}],
          "metadata": dict   # extra fields preserved for diagnostics
        }

    Raises LoongSchemaError if the example cannot be unambiguously
    normalized (missing question, answer, or documents).
    """
    if not isinstance(raw, dict):
        raise LoongSchemaError(f"example is not a dict: {type(raw).__name__}")

    example_id = _first_present(raw, _ID_KEYS)
    if example_id is None:
        # Fall back to a hash-derived id; downstream still needs *something*
        # stable to key the chunk hashes by.
        example_id = f"loong_{abs(hash(json.dumps(raw, sort_keys=True))) % 10**12}"
    example_id = str(example_id)

    question = _first_present(raw, _QUESTION_KEYS)
    if question is None:
        raise LoongSchemaError(
            f"example {example_id}: no question field. "
            f"top-level keys: {sorted(raw.keys())}"
        )
    question = _coerce_to_str(question).strip()
    if not question:
        raise LoongSchemaError(f"example {example_id}: empty question")

    answer_raw = _first_present(raw, _ANSWER_KEYS)
    aliases_raw = _first_present(raw, _ALIASES_KEYS)

    if answer_raw is None:
        # Try the "answers" list shape.
        answers_list = _first_present(raw, _ANSWERS_LIST_KEYS)
        if answers_list is not None:
            if not isinstance(answers_list, list) or not answers_list:
                raise LoongSchemaError(
                    f"example {example_id}: answers list is empty or not a list"
                )
            answer_raw = answers_list[0]
            extra_aliases = [str(a) for a in answers_list[1:] if a is not None]
            aliases_raw = (aliases_raw or []) + extra_aliases  # type: ignore[operator]
        else:
            raise LoongSchemaError(
                f"example {example_id}: no answer/answers field. "
                f"top-level keys: {sorted(raw.keys())}"
            )
    answer = _coerce_to_str(answer_raw).strip()

    if aliases_raw is None:
        aliases: List[str] = []
    elif isinstance(aliases_raw, list):
        aliases = [_coerce_to_str(a).strip() for a in aliases_raw if a is not None]
    else:
        aliases = [_coerce_to_str(aliases_raw).strip()]

    docs_raw = _first_present(raw, _DOCS_KEYS)
    if docs_raw is None:
        raise LoongSchemaError(
            f"example {example_id}: no documents/contexts/passages field. "
            f"top-level keys: {sorted(raw.keys())}"
        )
    if not isinstance(docs_raw, list):
        # Some Loong variants put context as a single big string. Wrap.
        docs_raw = [docs_raw]
    if not docs_raw:
        raise LoongSchemaError(f"example {example_id}: empty document list")

    documents = [_normalize_doc(doc, i, example_id) for i, doc in enumerate(docs_raw)]

    metadata = {
        k: raw[k] for k in raw
        if k not in (
            *_QUESTION_KEYS, *_ANSWER_KEYS, *_ANSWERS_LIST_KEYS,
            *_ALIASES_KEYS, *_DOCS_KEYS, *_ID_KEYS,
        )
    }

    return {
        "id": example_id,
        "question": question,
        "answer": answer,
        "answer_aliases": aliases,
        "documents": documents,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Chunk selection policy.
# ---------------------------------------------------------------------------
DEFAULT_LOONG_NUM_CHUNKS = 11
DEFAULT_LOONG_CHUNK_TEMPLATE_TITLED = "Title: {title}\n\n{text}"
DEFAULT_LOONG_CHUNK_TEMPLATE_UNTITLED = "{text}"


class LoongChunkPolicyError(Exception):
    pass


def _render_chunk_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title:
        return DEFAULT_LOONG_CHUNK_TEMPLATE_TITLED.format(title=title, text=text)
    return DEFAULT_LOONG_CHUNK_TEMPLATE_UNTITLED.format(text=text)


def select_loong_chunks(
    normalized: dict,
    *,
    num_chunks: int = DEFAULT_LOONG_NUM_CHUNKS,
    on_extra_chunks: str = "first",
    on_fewer_chunks: str = "skip",
) -> Optional[Tuple[List[SelectedChunk], str]]:
    """
    Returns (chunks, skip_reason_or_None). When the example must be
    skipped, returns (None, skip_reason_str).
    """
    docs = normalized["documents"]
    n = len(docs)
    if n == num_chunks:
        chosen = docs
    elif n > num_chunks:
        if on_extra_chunks == "first":
            chosen = docs[:num_chunks]
        elif on_extra_chunks == "skip":
            return None, "more_than_required_chunks"
        elif on_extra_chunks == "error":
            raise LoongChunkPolicyError(
                f"example {normalized['id']}: {n} chunks > {num_chunks}"
            )
        else:
            raise ValueError(f"unknown on_extra_chunks={on_extra_chunks!r}")
    else:  # n < num_chunks
        if on_fewer_chunks == "skip":
            return None, "fewer_than_required_chunks"
        elif on_fewer_chunks == "error":
            raise LoongChunkPolicyError(
                f"example {normalized['id']}: {n} chunks < {num_chunks}"
            )
        elif on_fewer_chunks == "use_all":
            chosen = docs
        else:
            raise ValueError(f"unknown on_fewer_chunks={on_fewer_chunks!r}")

    selected: List[SelectedChunk] = []
    for doc in chosen:
        selected.append(SelectedChunk(
            chunk_id=doc["chunk_id"],
            paragraph_idx=doc["paragraph_idx"],
            title=doc["title"],
            text=_render_chunk_text(doc),
            # Loong doesn't expose supporting/non-supporting labels; the
            # benchmark assumes every provided document is potentially
            # relevant.
            is_supporting=True,
            l2_distance_to_question=None,
        ))
    return selected, None


# ---------------------------------------------------------------------------
# Case construction (Stage A entry point).
# ---------------------------------------------------------------------------
def build_loong_case(
    raw_example: dict,
    *,
    prefix_text: str = DEFAULT_PREFIX,
    blend_special_str: str = "# #",
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
    dummy_warmup_query: str = DEFAULT_DUMMY_WARMUP_QUERY,
    num_chunks: int = DEFAULT_LOONG_NUM_CHUNKS,
    seed: int = 42,
    first_order_policy: str = "original",
    on_extra_chunks: str = "first",
    on_fewer_chunks: str = "skip",
) -> Tuple[Optional[MusiqueCase], Optional[str]]:
    """Stage A: build a tokenizer-independent MusiqueCase from a raw
    Loong example.

    Returns (case, skip_reason). On success, case is non-None and
    skip_reason is None.
    """
    try:
        normalized = normalize_loong_example(raw_example)
    except LoongSchemaError as e:
        return None, f"unsupported_schema: {e}"

    chunks_tuple = select_loong_chunks(
        normalized,
        num_chunks=num_chunks,
        on_extra_chunks=on_extra_chunks,
        on_fewer_chunks=on_fewer_chunks,
    )
    if chunks_tuple is None:
        return None, "unknown_chunk_selection_error"
    chunks, skip_reason = chunks_tuple
    if chunks is None:
        return None, skip_reason

    # Chunk order policy.
    chunk_ids = [c.chunk_id for c in chunks]
    if first_order_policy == "original":
        first_order = list(chunk_ids)
    elif first_order_policy == "random":
        rng = random.Random(f"{seed}:{normalized['id']}")
        first_order = list(chunk_ids)
        rng.shuffle(first_order)
    else:
        raise ValueError(f"unknown first_order_policy={first_order_policy!r}")
    second_order = list(reversed(first_order))

    real_question_segment_text = question_template.format(
        question=normalized["question"].strip(),
    )
    if dummy_warmup_query == real_question_segment_text:
        return None, "dummy_collides_with_real_question"

    case = MusiqueCase(
        id=normalized["id"],
        dataset="loong",
        question=normalized["question"],
        answer=normalized["answer"],
        answer_aliases=normalized["answer_aliases"],
        prefix_text=prefix_text,
        blend_special_str=blend_special_str,
        real_question_segment_text=real_question_segment_text,
        dummy_warmup_query_text=dummy_warmup_query,
        selected_chunks=chunks,
        first_order=first_order,
        second_order=second_order,
        order_policy="reverse_of_first_order",
    )
    return case, None


def iter_loong_cases(
    path: str,
    **kwargs,
) -> Iterator[Tuple[Optional[MusiqueCase], Optional[str]]]:
    """Yield (case, skip_reason_or_None) for every line of a Loong JSONL.

    `skip_reason` is None when `case` is built successfully.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                yield None, f"json_decode_error: {e}"
                continue
            yield build_loong_case(raw, **kwargs)


# ---------------------------------------------------------------------------
# Prompt length bucketing (used in markdown aggregation, exposed here
# so tests can exercise it without spinning up the full script).
# ---------------------------------------------------------------------------
def assign_length_bucket(second_prompt_token_length: int) -> str:
    """Map a second-prompt token length to one of the spec's buckets."""
    L = int(second_prompt_token_length)
    if L < 8 * 1024:
        return "0-8k"
    if L < 16 * 1024:
        return "8-16k"
    if L < 24 * 1024:
        return "16-24k"
    if L < 32 * 1024:
        return "24-32k"
    return "over_budget"


def prompt_token_budget(max_model_len: int, max_new_tokens: int, safety_margin: int) -> int:
    return int(max_model_len) - int(max_new_tokens) - int(safety_margin)
