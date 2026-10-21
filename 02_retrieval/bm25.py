"""BM25 (Okapi) retriever, self-implemented for full determinism and version
independence (workorder deliverable 7).

Parameters k1 / b come from config. IDF uses the Lucene-style smoothed
formulation ln(1 + (N - df + 0.5) / (df + 0.5)), which is always positive.
Postings live on the SearchIndex (per-user corpus — isolation keeps BM25
statistics scoped to one user's memory, by construction).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .tokenize import content_tokens

if TYPE_CHECKING:  # pragma: no cover
    from .engine import SearchIndex


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, stemming: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self.stemming = stemming

    def score(self, index: SearchIndex, query: str) -> dict[int, float]:
        terms = content_tokens(query, self.stemming)
        if not terms or index.n_docs == 0:
            return {}
        k1, b = self.k1, self.b
        avgdl = index.avgdl or 1.0
        out: dict[int, float] = {}
        seen: set[str] = set()
        for term in terms:  # unique query terms only; duplicates do not double-count
            if term in seen or term not in index.bm25_postings:
                continue
            seen.add(term)
            postings = index.bm25_postings[term]
            df = len(postings)
            idf = math.log(1.0 + (index.n_docs - df + 0.5) / (df + 0.5))
            for idx, tf in postings:
                dl = index.doc_len[idx]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                out[idx] = out.get(idx, 0.0) + idf * tf * (k1 + 1.0) / denom
        return out
