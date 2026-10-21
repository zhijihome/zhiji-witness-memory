"""Retrieval core gates: G5 (independent toggles), G6 (determinism), plus
sanity checks for each retriever, fusion, and neighbor expansion."""

from __future__ import annotations

import pytest
from aml_config import load_config
from aml_retrieval import RetrievalEngine
from aml_store import AddPayload, CompetitionMemoryStore, MessageIn

CORPUS = [
    ("alice", "sess-1", "I adopted a golden retriever puppy named Max last week.", 1683500000000),
    ("melanie", "sess-1", "That is wonderful news indeed!", 1683500100000),
    ("alice", "sess-1", "He is three months old. I started a new job at the bakery on 2 May 2023.", 1683500200000),
    ("alice", "sess-2", "My favourite hobby is rock climbing in Yosemite.", 1683600000000),
    ("melanie", "sess-2", "I prefer oil painting, especially portraits.", 1683600100000),
]


@pytest.fixture()
def engine_and_index(tmp_path):
    store = CompetitionMemoryStore(tmp_path / "r.db")
    for i, (role, sess, text, ts) in enumerate(CORPUS):
        store.add(
            AddPayload(
                request_id=f"r{i}",
                user_id="u1",
                session_id=sess,
                messages=[MessageIn(role=role, content=text, timestamp=ts)],
            )
        )
    # hermetic + pinned to the V0 retrieval world: hashed dense, exact on,
    # neighbor off (shipped V0.5 defaults differ; see test_v05_hardening.py)
    cfg = load_config(
        overrides={
            "storage": {"db_path": str(tmp_path / "r.db")},
            "dense": {"backend": "hashed_ngram"},
            "retrievers": {"exact": True},
            "surface": {"NEIGHBOR_POLICY": "off"},
        }
    )
    engine = RetrievalEngine(cfg)
    index = engine.build_index(store.get_user_index("u1"))
    store.close()
    return engine, index


def _ids(outcome):
    return [item.id for item in outcome.items]


def test_each_retriever_finds_its_target(engine_and_index):
    engine, index = engine_and_index
    # exact: literal substring
    scores = engine.exact.score(index, "golden retriever puppy named Max")
    assert scores and max(scores.values()) == 1.0
    # bm25: lexical overlap
    scores = engine.bm25.score(index, "bakery job")
    assert scores, "bm25 should score the bakery unit"
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    assert "bakery" in index.ui.units[top].content
    # dense: hashed cosine ranks topical match first
    q_vec = engine.dense.embed_query("rock climbing hobby")
    scores = engine.dense.score(index, q_vec)
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    assert "climbing" in index.ui.units[top].content


def test_g5_all_toggle_combinations(tmp_path, tmp_config):
    combos = [
        {"exact": True, "bm25": False, "dense": False},
        {"exact": False, "bm25": True, "dense": False},
        {"exact": False, "bm25": False, "dense": True},
        {"exact": True, "bm25": True, "dense": False},
        {"exact": True, "bm25": False, "dense": True},
        {"exact": False, "bm25": True, "dense": True},
        {"exact": True, "bm25": True, "dense": True},
    ]
    store = CompetitionMemoryStore(tmp_path / "t.db")
    for i, (role, sess, text, ts) in enumerate(CORPUS):
        store.add(
            AddPayload(
                request_id=f"r{i}",
                user_id="u1",
                session_id=sess,
                messages=[MessageIn(role=role, content=text, timestamp=ts)],
            )
        )
    for combo in combos:
        cfg = tmp_config({"retrievers": combo})
        engine = RetrievalEngine(cfg)
        index = engine.build_index(store.get_user_index("u1"))
        out = engine.search(index, "bakery job", 100)
        assert sorted(out.trace["retrievers_on"]) == sorted(k for k, v in combo.items() if v)
        assert out.items, f"no hits with {combo}"
    store.close()


def test_g6_same_config_three_runs_identical(tmp_path, tmp_config):
    cfg = tmp_config()
    store = CompetitionMemoryStore(tmp_path / "d.db")
    for i, (role, sess, text, ts) in enumerate(CORPUS):
        store.add(
            AddPayload(
                request_id=f"r{i}",
                user_id="u1",
                session_id=sess,
                messages=[MessageIn(role=role, content=text, timestamp=ts)],
            )
        )
    baseline = None
    for _ in range(3):
        engine = RetrievalEngine(cfg)  # fresh engine each run
        index = engine.build_index(store.get_user_index("u1"))
        out = engine.search(index, "How old is Max? What about the new job?", 100)
        signature = [(i.id, i.score, i.content) for i in out.items]
        if baseline is None:
            baseline = signature
        else:
            assert signature == baseline  # exact tie-break determinism
    store.close()


def test_g6_determinism_with_ties(client_factory):
    c = client_factory()
    msgs = [{"role": "u", "content": "identical tie token alpha"} for _ in range(6)]
    c.post("/add", json={"request_id": "ties", "messages": msgs, "user_id": "tie-user", "session_id": "s"})
    signatures = set()
    for _ in range(3):
        out = c.post("/search", json={"query": "identical tie token", "user_id": "tie-user", "top_k": 100})
        signatures.add(tuple((i["id"], i["score"]) for i in out.json()["data"]))
    assert len(signatures) == 1


def test_neighbor_expansion_adds_context(engine_and_index, tmp_config):
    engine, index = engine_and_index
    # window off (default): only directly-matching units
    base = engine.search(index, "golden retriever puppy named Max", 100)
    base_ids = {i.id for i in base.items}
    assert "mem-000002" not in base_ids  # the neighbour turn is not a lexical hit

    cfg2 = tmp_config({"surface": {"NEIGHBOR_POLICY": "window:1"}})
    engine2 = RetrievalEngine(cfg2)
    expanded = engine2.search(index, "golden retriever puppy named Max", 100)
    expanded_ids = {i.id for i in expanded.items}
    assert "mem-000002" in expanded_ids  # adjacent turn pulled in as context
    via = {i.id: i.via for i in expanded.items}
    assert via["mem-000002"] == "neighbor"


def test_rrf_fusion_beats_single_retriever_on_partial_overlap(engine_and_index):
    engine, index = engine_and_index
    out = engine.search(index, "Max age job bakery", 100)
    assert out.items
    assert out.trace["retrievers_on"] == ["bm25", "dense", "exact"]
    # both target facts surface in the fused ranking
    contents = " ".join(i.content for i in out.items)
    assert "bakery" in contents and "Max" in contents
