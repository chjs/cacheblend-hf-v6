"""
Phase 4 — LongBench-style token-level F1 evaluation.

Lowercase + strip punctuation + whitespace-tokenize + precision/recall/F1.
When `aliases` is non-empty we score against each (gold + aliases) and
take the max.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List


_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    return _normalize(text).split()


def f1_one(pred: str, gold: str) -> float:
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not pred_tokens and not gold_tokens:
        # Both empty after normalisation: perfect match (degenerate).
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def f1_with_aliases(pred: str, gold: str, aliases: Iterable[str] = ()) -> float:
    best = f1_one(pred, gold)
    for alias in aliases or ():
        if not isinstance(alias, str):
            continue
        score = f1_one(pred, alias)
        if score > best:
            best = score
    return best
