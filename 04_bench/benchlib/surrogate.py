"""Fixed deterministic surrogate answerer (LOCAL coordinate only).

This stands in for the platform's fixed answer model so that LOCAL runs are a
fair ablation surface: it is IDENTICAL across all our configurations and it
never sees gold answers, gold evidence lists, or any LLM. Its outputs are only
ever compared within the LOCAL coordinate — they are not, and must not be
reported as, AML scores (Architecture C3 / Contract §5).
"""

from __future__ import annotations

import re

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TEMPORAL_Q = re.compile(
    r"\b(when|what year|which year|what date|how (long|old|many years)|last|next|ago|since)\b", re.IGNORECASE
)
_DATE_LIKE = re.compile(
    r"(\b(?:19|20)\d{2}\b)"
    r"|(\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b\.?(?:,?\s*\d{4})?)"
    r"|(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)"
    r"|(\b\d{4}-\d{2}-\d{2}\b)",
    re.IGNORECASE,
)
_MAX_ANSWER_WORDS = 30


def _norm_tokens(text: str) -> list[str]:
    text = _PUNCT.sub(" ", text.lower())
    text = _ARTICLES.sub(" ", text)
    return text.split()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def surrogate_answer(question: str, options: list[str] | None, evidence_texts: list[str]) -> str:
    """Deterministic extractive answer from retrieved evidence (never gold)."""
    q_tokens = set(_norm_tokens(question))
    if options:
        joined = " ".join(evidence_texts[:10]).lower()
        ev_tokens = set(_norm_tokens(joined))
        best, best_score = "", -1.0
        for opt in options:
            s = _jaccard(set(_norm_tokens(opt)), ev_tokens) + 0.001 * len(set(_norm_tokens(opt)) & q_tokens)
            if s > best_score:
                best, best_score = opt, s
        return best

    if not evidence_texts:
        return ""

    temporal = bool(_TEMPORAL_Q.search(question))
    candidates: list[tuple[float, int, int, str]] = []  # (score, rank, sent_rank, sentence)
    for rank, evidence in enumerate(evidence_texts[:10]):
        for sent_rank, sentence in enumerate(_split_sentences(evidence)):
            s = _jaccard(q_tokens, set(_norm_tokens(sentence)))
            if temporal and _DATE_LIKE.search(sentence):
                s += 0.15
            candidates.append((round(s, 6), rank, sent_rank, sentence))
    if not candidates:
        candidates = [(0.0, 0, 0, evidence_texts[0])]
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    best_sentence = candidates[0][3]

    if temporal:
        m = _DATE_LIKE.search(best_sentence)
        if m:
            start = m.start()
            # widen to word boundaries around the date-like span
            while start > 0 and best_sentence[start - 1].isalnum():
                start -= 1
            end = m.end()
            while end < len(best_sentence) and best_sentence[end].isalnum():
                end += 1
            return best_sentence[start:end].strip(" .,!?")

    words = best_sentence.split()
    return " ".join(words[:_MAX_ANSWER_WORDS])
