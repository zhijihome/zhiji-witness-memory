"""Shared deterministic tokenization for all retrievers.

Same input string → same token list, always, on every machine (no locale, no
seeded hashing anywhere in this package).
"""

from __future__ import annotations

import re
from unicodedata import normalize as unicodedata_normalize

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)

# Compact English stopword list — deliberately small: aggressive stopword removal
# hurts exact-phrase semantics and conversational short answers.
STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have he her hers him his i if in into is it
    its me my of on or our she so than that the their them then there these they this to
    was we were what which who will with you your
    """.split()
)


def norm_text(text: str) -> str:
    """NFKC + casefold + whitespace collapse (mirrors aml_store.normalize_text)."""
    return " ".join(unicodedata_normalize("NFKC", text).casefold().split())


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(norm_text(text))


def light_stem(token: str) -> str:
    """Conservative English normalizer: possessive + plural only.

    Deliberately NOT a full Porter stemmer — aggressive stemming trades
    precision for recall; character n-grams in the dense retriever already
    bridge most morphological variants.
    """
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_tokens(text: str, stemming: bool = True) -> list[str]:
    """Stopword-filtered tokens used by BM25 and dense word features."""
    return [light_stem(t) if stemming else t for t in tokenize(text) if t not in STOPWORDS]
