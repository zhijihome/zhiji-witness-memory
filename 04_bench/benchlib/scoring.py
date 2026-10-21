"""Scoring metrics for the LOCAL proxy benchmark (SQuAD-style, fixed)."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if not pred_tokens and not gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_recall(returned_dia_ids: list[str], gold_dia_ids: list[str]) -> float:
    """Fraction of gold evidence dia_ids present in the returned surface."""
    if not gold_dia_ids:
        return 0.0
    got = set(returned_dia_ids)
    return len(got & set(gold_dia_ids)) / len(set(gold_dia_ids))
