"""
Phase 4 Stage B — tokenizer-specific materialisation.

Implements LMCache's `sep_ids = tokenizer.encode(blend_special_str)[1:]`
behaviour: each segment is tokenized *independently* with the BOS
stripped, then the lists are concatenated manually. Never tokenize
the whole concatenated prompt as the source of truth (see
docs/phases/PHASE4_PROMPT.md "Separator and CacheBlend segment
semantics" + `examples/blend_kv_v1/blend.py:147-186`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scripts._phase4_musique import MusiqueCase


def _encode_strip(tokenizer, text: str) -> List[int]:
    """`tokenizer.encode(text)[1:]` — strips one leading auto-BOS."""
    return list(tokenizer.encode(text))[1:]


def count_subsequence(xs: List[int], pat: List[int]) -> int:
    """Number of (overlapping-allowed) occurrences of `pat` in `xs`."""
    if not pat:
        raise ValueError("empty separator token sequence")
    n = len(xs)
    L = len(pat)
    return sum(1 for i in range(n - L + 1) if xs[i:i + L] == pat)


@dataclass
class MaterializedCase:
    case: MusiqueCase

    # Per-segment token id lists (all post-BOS-strip).
    sep_ids: List[int]
    prefix_ids: List[int]
    question_ids: List[int]
    chunk_ids_by_id: Dict[str, List[int]]

    # Concatenated full prompt token id lists.
    first_prompt_ids: List[int]
    second_prompt_ids: List[int]

    # Tokenizer metadata.
    tokenizer_name_or_path: str
    construction: str = "segmentwise_tokenize_then_concat"
    bos_removed_after_encode: bool = True

    # Validation summary.
    separator_count_first: int = 0
    separator_count_second: int = 0
    orders_are_reverse: bool = True
    no_internal_separator: bool = True

    def to_metadata_dict(self) -> dict:
        return {
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "blend_special_str": self.case.blend_special_str,
            "sep_ids": self.sep_ids,
            "construction": self.construction,
            "bos_removed_after_encode": self.bos_removed_after_encode,
        }


class InternalSeparatorError(Exception):
    pass


def materialize(
    case: MusiqueCase,
    tokenizer,
    *,
    tokenizer_name_or_path: str,
    on_internal_separator: str = "skip",
) -> Optional[MaterializedCase]:
    """
    Stage B entrypoint. Returns the MaterializedCase, or None when the
    example should be skipped (e.g. internal separator detected and
    policy is `skip`).
    """
    sep_ids = _encode_strip(tokenizer, case.blend_special_str)
    if not sep_ids:
        raise ValueError(
            f"separator {case.blend_special_str!r} tokenized to empty after BOS strip"
        )

    prefix_ids = _encode_strip(tokenizer, case.prefix_text)
    question_ids = _encode_strip(tokenizer, case.question_segment_text)

    chunk_ids_by_id: Dict[str, List[int]] = {}
    for chunk in case.selected_chunks:
        chunk_ids_by_id[chunk.chunk_id] = _encode_strip(tokenizer, chunk.text)

    # Internal-separator check: no individual segment may contain `sep_ids`.
    bad_segments: List[str] = []
    if count_subsequence(prefix_ids, sep_ids) > 0:
        bad_segments.append("prefix")
    if count_subsequence(question_ids, sep_ids) > 0:
        bad_segments.append("question")
    for cid, ids in chunk_ids_by_id.items():
        if count_subsequence(ids, sep_ids) > 0:
            bad_segments.append(f"chunk:{cid}")

    no_internal_separator = not bad_segments
    if bad_segments:
        if on_internal_separator == "error":
            raise InternalSeparatorError(
                f"case {case.id!r}: internal separator detected in {bad_segments}"
            )
        return None

    # Concatenate per the LMCache pattern.
    def concat(order: List[str]) -> List[int]:
        out: List[int] = []
        out.extend(prefix_ids)
        out.extend(sep_ids)
        for cid in order:
            out.extend(chunk_ids_by_id[cid])
            out.extend(sep_ids)
        out.extend(question_ids)
        return out

    first_prompt_ids = concat(case.first_order)
    second_prompt_ids = concat(case.second_order)

    # Sep count validation: 1 (post-prefix) + N (after each chunk) = N+1.
    expected_sep_count = 1 + len(case.first_order)
    first_count = count_subsequence(first_prompt_ids, sep_ids)
    second_count = count_subsequence(second_prompt_ids, sep_ids)
    assert first_count == expected_sep_count, (
        f"first prompt sep count {first_count} != {expected_sep_count}"
    )
    assert second_count == expected_sep_count, (
        f"second prompt sep count {second_count} != {expected_sep_count}"
    )

    # Order invariants.
    orders_are_reverse = case.second_order == list(reversed(case.first_order))
    assert orders_are_reverse
    assert set(case.first_order) == set(case.second_order)

    return MaterializedCase(
        case=case,
        sep_ids=sep_ids,
        prefix_ids=prefix_ids,
        question_ids=question_ids,
        chunk_ids_by_id=chunk_ids_by_id,
        first_prompt_ids=first_prompt_ids,
        second_prompt_ids=second_prompt_ids,
        tokenizer_name_or_path=tokenizer_name_or_path,
        separator_count_first=first_count,
        separator_count_second=second_count,
        orders_are_reverse=orders_are_reverse,
        no_internal_separator=no_internal_separator,
    )
