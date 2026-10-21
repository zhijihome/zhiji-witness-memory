"""Exact / literal-phrase retriever (layer 02).

Scoring (deterministic, documented):
  * 1.0 when the whole normalized query is a substring of the normalized unit
    content (the exact return surface — no role prefix here: we match literally
    what we would return);
  * otherwise greedy longest-phrase coverage over query tokens left→right:
    consume the longest contiguous run (≤8 tokens) found verbatim as a
    substring of the normalized unit. Stopword tokens never count toward
    coverage on their own (a bare "is the" match is not evidence); they only
    ride along inside a longer matched run. score = matched_content_tokens /
    total_content_tokens;
  * units below `min_coverage` are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .tokenize import STOPWORDS, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from .engine import SearchIndex

_MAX_RUN = 8


def phrase_coverage(norm_content: str, q_norm: str, q_tokens: list[str]) -> float:
    if q_norm and q_norm in norm_content:
        return 1.0
    content_positions = [i for i, t in enumerate(q_tokens) if t not in STOPWORDS]
    total = len(content_positions)
    if total == 0:
        return 0.0
    matched_positions: set[int] = set()
    i = 0
    n = len(q_tokens)
    while i < n:
        matched = 0
        for run in range(min(_MAX_RUN, n - i), 0, -1):
            phrase = " ".join(q_tokens[i : i + run])
            if phrase in norm_content:
                matched = run
                break
        if matched:
            matched_positions.update(range(i, i + matched))
            i += matched
        else:
            i += 1
    hit = sum(1 for p in matched_positions if p in set(content_positions))
    return hit / total


class ExactRetriever:
    name = "exact"

    def __init__(self, min_coverage: float = 0.25) -> None:
        self.min_coverage = min_coverage

    def score(self, index: SearchIndex, query: str) -> dict[int, float]:
        from aml_store.store import normalize_text  # same definition, avoids drift

        q_norm = normalize_text(query)
        if not q_norm:
            return {}
        q_tokens = tokenize(query)
        out: dict[int, float] = {}
        for idx, norm_content in enumerate(index.ui.norm_contents):
            s = phrase_coverage(norm_content, q_norm, q_tokens)
            if s >= self.min_coverage and s > 0:
                out[idx] = s
        return out
