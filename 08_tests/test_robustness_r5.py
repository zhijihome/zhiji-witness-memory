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


# --- lenient-ingest edges (audit O5 + judge re-check) -------------------------


def test_ts_ms_never_500s_on_pathological_values(tmp_config):
    """inf / 1e999 / microsecond-confused / negative timestamps must not crash:
    schema degrades them to None instead of raising OverflowError out of
    pydantic (-> 500) or overflowing datetime at store insert (-> 500)."""
    from aml_api.service import MemoryService
    from aml_api.schemas import AddRequest
    from aml_store import AddPayload

    cfg = tmp_config({"dense": {"backend": "hashed_ngram"}})
    svc = MemoryService(cfg)
    pathological = [float("inf"), float("-inf"), 1e999, -1e999, 1_760_000_000_000_000_000, -5, 253_402_300_800_000]
    for i, ts in enumerate(pathological):
        req = AddRequest.model_validate({
            "request_id": f"r{i}", "user_id": "u", "session_id": "s",
            "messages": [{"role": "user", "content": f"fact {i} about cats", "timestamp": ts}],
        })
        out = svc.add(AddPayload(request_id=req.request_id, user_id=req.user_id,
                                 session_id=req.session_id,
                                 messages=[__import__("aml_store", fromlist=["MessageIn"]).MessageIn(
                                     role=m.role, content=m.content, timestamp=m.timestamp)
                                     for m in req.messages]))
        assert out.stored, f"ts={ts!r} must still store"
    hits = svc.search("u", "cats", 20).items
    assert len(hits) >= len(pathological)
    svc.close()


def test_empty_content_dropped_before_store(tmp_config):
    """Empty/whitespace content is schema-accepted but must never be stored —
    neighbor expansion would otherwise surface content:'' in responses (G15)."""
    from aml_api.service import MemoryService
    from aml_api.schemas import AddRequest
    from aml_store import AddPayload, MessageIn

    cfg = tmp_config({"dense": {"backend": "hashed_ngram"}})
    svc = MemoryService(cfg)
    req = AddRequest.model_validate({
        "request_id": "r", "user_id": "u", "session_id": "s",
        "messages": [
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "real fact about dogs"},
        ],
    })
    out = svc.add(AddPayload(request_id=req.request_id, user_id=req.user_id,
                             session_id=req.session_id,
                             messages=[MessageIn(role=m.role, content=m.content, timestamp=m.timestamp)
                                       for m in req.messages]))
    assert len(out.stored) == 1  # only the real unit
    hits = svc.search("u", "dogs", 10).items
    assert all(i.content.strip() for i in hits), [i.content for i in hits]
    svc.close()


def test_min_score_config_validation():
    """min_score<=0 would diverge matrix vs postings hit-sets — config rejects."""
    import json
    import tempfile
    from pathlib import Path

    from aml_config.config import ConfigError, load_config

    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.json"
        base = json.loads(Path("10_config/defaults.json").read_text())
        base["dense"]["min_score"] = 0
        bad.write_text(json.dumps(base))
        with pytest.raises((ConfigError, Exception)):
            load_config(path=str(bad))


# --- r8 audit follow-ups (AUDIT-VERDICT-R3 leftovers) -------------------------


def test_search_options_non_string_never_422(client_factory):
    """MCQ options are accepted-but-ignored: numeric / mixed / scalar options
    must not 422 the whole Search (no-retry class → question gets no memory)."""
    with client_factory() as c:
        c.post("/add", json={"request_id": "r", "user_id": "u", "session_id": "s",
                             "messages": [{"role": "user", "content": "I was born in 1990 in Lyon"}]})
        for opts in ([1990, 1991], [1990.5, None, True], "A. only", 7, ["A. 1990", "B. 1991"]):
            resp = c.post("/search", json={"query": "where was I born", "user_id": "u", "top_k": 10, "options": opts})
            assert resp.status_code == 200, (opts, resp.text)
            assert any("1990" in i["content"] for i in resp.json()["data"])


def test_null_content_degrades_to_dropped_not_422(client_factory):
    """content:null in one message must not 422 the chunk; the null message is
    dropped like empty content, its siblings are stored and searchable."""
    with client_factory() as c:
        resp = c.post("/add", json={"request_id": "r", "user_id": "u", "session_id": "s",
                                    "messages": [{"role": "user", "content": None},
                                                 {"role": "assistant", "content": "the boat is named Marlin"}]})
        assert resp.status_code == 200, resp.text
        data = c.post("/search", json={"query": "boat name", "user_id": "u", "top_k": 10}).json()["data"]
        assert data and all(i["content"].strip() for i in data)
        assert any("Marlin" in i["content"] for i in data)


def test_batched_embed_vectors_reused_not_refetched():
    """Rebuild speed-up: each text is embedded exactly once (the batched call's
    vectors are used directly), and the packed matrix is identical to the
    per-item path."""
    import numpy as np
    from aml_retrieval.engine import SearchIndex
    from aml_store.model import MemoryUnit
    from aml_store.store import UserIndex

    class Counting(_TinyDenseEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.single = 0

        def embed(self, text):
            self.single += 1
            return super().embed(text)

        def embed_many(self, texts):
            return [_TinyDenseEmbedder.embed(self, t) for t in texts]

    class NoBatch(_TinyDenseEmbedder):
        embed_many = None  # forces the per-item fallback path

    units = [MemoryUnit(id=f"u{i}", user_id="u", session_id="s", request_id="r", msg_index=i,
                        role="user", content=f"text number {i}", ts_ms=None, created_at="",
                        arrival_seq=i + 1, session_seq=i, content_sha256="") for i in range(600)]
    a = SearchIndex(UserIndex(user_id="u"), temporal_enabled=False)
    emb = Counting()
    a.extend(units, emb)
    assert emb.single == 0 and emb.calls == len(units)  # one batched embedding per text, no re-read
    b = SearchIndex(UserIndex(user_id="u"), temporal_enabled=False)
    b.extend(units, NoBatch())
    assert np.array_equal(a.dense_matrix, b.dense_matrix) and a.dense_rows == b.dense_rows


def test_manifest_image_pins_read_from_dockerfile():
    """Manifest image pins are parsed from the Dockerfile (single source), both
    on the default path and when runtime facts are injected."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "competition"))
    import generate_manifest as gm

    pins = gm.dockerfile_pins()
    text = (root / "05_docker" / "Dockerfile").read_text()
    assert pins["base_image_digest"].split("@")[1] in text and pins["base_image_digest"] != "UNKNOWN"
    assert pins["ce_rerank_revision"] in text and pins["ce_rerank_revision"] != "UNKNOWN"
    cfg = root / "10_config" / "competition.a1ds.json"
    m = gm.build_manifest(cfg, environment_summary={"runtime": "x", "ce_rerank_revision": "stale"})
    assert m["environment_summary"]["ce_rerank_revision"] == pins["ce_rerank_revision"]
    assert m["environment_summary"]["runtime"] == "x"
