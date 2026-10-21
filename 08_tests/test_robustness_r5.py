"""Robustness gates from the r5 external audit (AUDIT-VERDICT-R1):

* R1 — the embedding cache connection must survive cross-thread use
  (FastAPI sync endpoints run on an AnyIO worker pool).
* O1 — an embed/index failure mid-Add must not leave a permanent
  unsearchable hole; the next write rebuilds from the canonical store.
* R2 — dense backends store vectors in a packed matrix (bounded memory),
  not in per-dimension Python postings.
"""

from __future__ import annotations

import threading

import pytest


class _TinyDenseEmbedder:
    """Deterministic tiny dense backend (8-dim) — exercises the matrix path
    the way the 1024-dim DashScope backend does, without network or models."""

    dimensions = 8

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        import math

        vals = [(hash(text) >> (i * 7)) % 10 + 1 for i in range(self.dimensions)]
        n = math.sqrt(sum(v * v for v in vals))
        return [v / n for v in vals]

    def embed(self, text: str) -> dict[int, float]:
        self.calls += 1
        return {i: v for i, v in enumerate(self._vec(text)) if v != 0}

    def embed_many(self, texts: list[str]) -> list[dict[int, float]]:
        return [self.embed(t) for t in texts]


def test_embed_cache_cross_thread(tmp_path, monkeypatch):
    """R1: cache connection created on one worker thread must serve all."""
    monkeypatch.setenv("FAKE_EMB_KEY", "x")
    from aml_retrieval.dense import DashScopeEmbedder

    db = tmp_path / "emb.sqlite"
    holder: dict = {}

    def build_and_seed():
        e = DashScopeEmbedder("http://127.0.0.1:9", "FAKE_EMB_KEY", "m", 4, str(db))
        e._db.execute("INSERT INTO emb VALUES (?,?)", (e._key("hello"), e._pack([1.0, 0, 0, 0])))
        e._db.commit()
        holder["e"] = e

    t = threading.Thread(target=build_and_seed)
    t.start()
    t.join()
    # main thread is a *different* worker — pre-fix this raised ProgrammingError
    vec = holder["e"].embed("hello")
    assert vec and vec[0] == pytest.approx(1.0)


def test_dense_matrix_storage_replaces_postings():
    """R2: a dense backend populates a packed matrix, not bucket postings."""
    import numpy as np
    from aml_retrieval.dense import DenseRetriever
    from aml_retrieval.engine import SearchIndex
    from aml_store.model import MemoryUnit
    from aml_store.store import UserIndex

    idx = SearchIndex(UserIndex(user_id="u"), temporal_enabled=False, stemming=True)
    units = [
        MemoryUnit(id=f"u{i}", user_id="u", session_id="s", request_id="r",
                   msg_index=i, role="user", content=t, ts_ms=None,
                   created_at="", arrival_seq=i + 1, session_seq=i, content_sha256="")
        for i, t in enumerate(["alpha text one", "beta text two", "alpha again"])
    ]
    emb = _TinyDenseEmbedder()
    idx.extend(units, emb)
    assert isinstance(idx.dense_matrix, np.ndarray)
    assert idx.dense_matrix.shape == (3, 8)
    assert not idx.dense_postings  # matrix mode never touches postings
    scores = DenseRetriever(emb, min_score=0.0).score(idx, emb.embed("alpha"))
    assert len(scores) == 3 and max(scores.values()) <= 1.0 + 1e-5


def test_index_hole_self_heals_after_embed_failure(tmp_config):
    """O1: an embed failure on extend evicts the index — the next Add rebuilds
    from the canonical store and the once-lost unit becomes searchable."""
    from aml_api.service import MemoryService
    from aml_store import AddPayload, MessageIn

    cfg = tmp_config({"dense": {"backend": "hashed_ngram"}})
    svc = MemoryService(cfg)
    try:
        svc.add(AddPayload(request_id="r0", user_id="u", session_id="s",
                         messages=[MessageIn(role="user", content="durable fact about otters")]))

        real = svc.engine.index_embedder()

        class Boom:
            def embed_many(self, texts):
                raise RuntimeError("embed backend exploded")

            def embed(self, text):
                raise RuntimeError("embed backend exploded")

        svc.engine._embedder = Boom()
        with pytest.raises(Exception):
            svc.add(AddPayload(request_id="r1", user_id="u", session_id="s",
                               messages=[MessageIn(role="user", content="fact about herons")]))
        svc.engine._embedder = real

        out = svc.add(AddPayload(request_id="r2", user_id="u", session_id="s",
                                 messages=[MessageIn(role="user", content="fact about ibises")]))
        assert out.stored
        hits = svc.search("u", "otters", 10).items
        assert any("otters" in i.content for i in hits), [i.content for i in hits]
        hits = svc.search("u", "herons", 10).items
        assert any("herons" in i.content for i in hits), [i.content for i in hits]
    finally:
        svc.close()
