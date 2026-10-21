"""aml_retrieval — retrieval core (architecture layer 02, V0/V0.5 scope).

Exact / BM25 / dense retrieval, fusion (RRF or parameter-free CombMNZ
norm_sum), session-neighbor expansion, temporal anchors, and the
return-surface packer (select by relevance, then order by policy). No LLM,
no network, no writeback (INV2): search never mutates the store.
"""

from .bm25 import BM25Retriever
from .dense import DenseRetriever, HashedNgramEmbedder, build_embedder, resolve_embedding_device
from .engine import (
    DERIVED_BASE,
    DerivedDoc,
    PackedItem,
    RetrievalEngine,
    SearchIndex,
    SearchOutcome,
    fuse_norm_sum,
    fuse_rrf,
)
from .exact import ExactRetriever
from .temporal import TimeAnchor, extract_anchors, query_anchors

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HashedNgramEmbedder",
    "build_embedder",
    "resolve_embedding_device",
    "DERIVED_BASE",
    "DerivedDoc",
    "PackedItem",
    "RetrievalEngine",
    "SearchIndex",
    "SearchOutcome",
    "fuse_rrf",
    "fuse_norm_sum",
    "ExactRetriever",
    "TimeAnchor",
    "extract_anchors",
    "query_anchors",
]
