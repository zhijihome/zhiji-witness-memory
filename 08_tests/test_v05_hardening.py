"""V0.5 retrieval-hardening gates: H1 select/order decoupling (G30),
H3 norm_sum fusion (parameter-free, deterministic, rrf untouched),
H7 SERVER_CONFIG relation accounting (G34), O2 measured usage aggregation,
and the shipped-defaults sanity pin.

Hermetic by construction: everything here runs on the hashed_ngram backend
(via the conftest pin or explicit overrides) — no model downloads.
"""

from __future__ import annotations

import pytest
from aml_config import load_config
from aml_config.config import ConfigError
from aml_retrieval import RetrievalEngine, fuse_norm_sum, fuse_rrf
from aml_store import AddPayload, CompetitionMemoryStore, MessageIn

import _bootstrap  # noqa: F401

# ---------------------------------------------------------------------------
# H1 / G30 — ORDERING_POLICY only re-ranks the relevance-selected surface


CORPUS_6 = [
    ("u", "target alpha bravo strong match one", 1700000000000),
    ("u", "target alpha bravo strong match two", 1700000100000),
    ("u", "unrelated filler about gardening tools", 1700000200000),
    ("u", "target alpha bravo strong match three", 1700000300000),
    ("u", "another unrelated note about accounting", 1700000400000),
    ("u", "target alpha bravo strong match four", 1700000500000),
]


def _engine_with_corpus(tmp_path, overrides=None):
    store = CompetitionMemoryStore(tmp_path / "r.db")
    for i, (role, text, ts) in enumerate(CORPUS_6):
        store.add(
            AddPayload(
                request_id=f"r{i}",
                user_id="u1",
                session_id="s",
                messages=[MessageIn(role=role, content=text, timestamp=ts)],
            )
        )
    cfg = load_config(
        overrides={
            "storage": {"db_path": str(tmp_path / "r.db")},
            "dense": {"backend": "hashed_ngram"},
            **(overrides or {}),
        }
    )
    engine = RetrievalEngine(cfg)
    index = engine.build_index(store.get_user_index("u1"))
    store.close()
    return engine, index


def test_g30_ordering_policy_never_changes_the_selected_set(tmp_path):
    engine, index = _engine_with_corpus(tmp_path)
    rel = engine.search(index, "target alpha bravo strong match", 100)
    tmp_cfg = load_config(
        overrides={
            "storage": {"db_path": str(tmp_path / "r.db")},
            "dense": {"backend": "hashed_ngram"},
            "surface": {"ORDERING_POLICY": "temporal_asc"},
        }
    )
    engine2 = RetrievalEngine(tmp_cfg)
    index2 = engine2.build_index(index.ui)  # same underlying user index
    temporal = engine2.search(index2, "target alpha bravo strong match", 100)

    # identical id SET, temporal order ascending within the surface
    assert {i.id for i in rel.items} == {i.id for i in temporal.items}
    assert [i.created_at for i in temporal.items] == sorted(i.created_at for i in temporal.items)


def test_g30_temporal_asc_selects_by_relevance_not_by_age(tmp_path):
    """The V0 defect: temporal_asc picked the OLDEST docs. Now a small
    RETURN_SIZE must return the same top-N as relevance_desc, re-ordered."""
    engine, index = _engine_with_corpus(tmp_path, {"surface": {"RETURN_SIZE": 2}})
    rel = engine.search(index, "target alpha bravo strong match", 100)

    engine2 = RetrievalEngine(
        load_config(
            overrides={
                "storage": {"db_path": str(tmp_path / "r.db")},
                "dense": {"backend": "hashed_ngram"},
                "surface": {"RETURN_SIZE": 2, "ORDERING_POLICY": "temporal_asc"},
            }
        )
    )
    index2 = engine2.build_index(index.ui)
    temporal = engine2.search(index2, "target alpha bravo strong match", 100)

    # same selected pair, only the order differs — selection was by relevance
    assert {i.id for i in rel.items} == {i.id for i in temporal.items}
    assert len(temporal.items) == 2
    by_age = sorted(temporal.items, key=lambda i: i.created_at)
    assert [i.id for i in temporal.items] == [i.id for i in by_age]


# ---------------------------------------------------------------------------
# H3 — fusion: rrf golden (V0 semantics pinned), norm_sum hand-computed


def test_rrf_golden_v0_semantics_unchanged():
    ranked = {"a": [1, 2], "b": [2, 3]}
    fused = fuse_rrf(ranked, rrf_k=60, weights={})
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[3] == pytest.approx(1 / 62)


def test_norm_sum_hand_computed_max_norm():
    channels = {"bm25": {1: 10.0, 2: 5.0}, "dense": {2: 1.0, 3: 0.5}}
    fused = fuse_norm_sum(channels, norm="max")
    assert fused[1] == pytest.approx(1 * (10.0 / 10.0))
    assert fused[2] == pytest.approx(2 * (5.0 / 10.0 + 1.0 / 1.0))
    assert fused[3] == pytest.approx(1 * (0.5 / 1.0))


def test_norm_sum_hand_computed_minmax_norm():
    channels = {"bm25": {1: 10.0, 2: 5.0}, "dense": {2: 1.0, 3: 0.5}}
    fused = fuse_norm_sum(channels, norm="minmax")
    assert fused[1] == pytest.approx(1 * 1.0)
    assert fused[2] == pytest.approx(2 * (0.0 + 1.0))
    assert fused[3] == pytest.approx(1 * 0.0)


def test_norm_sum_agreement_beats_single_channel_top_hit():
    """A doc ranked top by a weak channel alone must not outrank a doc that two
    channels agree on — the V0 equal-RRF failure mode (Finalizer F2)."""
    channels = {"bm25": {1: 12.0, 2: 6.0}, "dense": {3: 0.9, 1: 0.4}}
    fused = fuse_norm_sum(channels, norm="max")
    assert fused[1] > fused[3]  # agreement beats a dense-only top hit
    assert fused[1] > fused[2]


def test_norm_sum_deterministic_and_single_channel_rank_preserving():
    channels = {"bm25": {5: 3.0, 9: 7.0, 2: 5.0}}
    a = fuse_norm_sum(channels, norm="max")
    b = fuse_norm_sum(dict(channels), norm="max")
    assert a == b
    assert max(a, key=lambda k: a[k]) == 9  # raw order preserved within channel


def test_norm_sum_rejects_unknown_norm():
    with pytest.raises(ConfigError):
        fuse_norm_sum({"bm25": {1: 1.0}}, norm="bogus")


def test_engine_norm_sum_end_to_end(tmp_path):
    engine, index = _engine_with_corpus(tmp_path, {"fusion": {"method": "norm_sum"}})
    out = engine.search(index, "target alpha bravo strong match", 100)
    assert out.trace["fusion_method"] == "norm_sum"
    assert out.items
    assert [i.score for i in out.items] == sorted((i.score for i in out.items), reverse=True)


def test_engine_rrf_default_trace(tmp_path):
    engine, index = _engine_with_corpus(tmp_path, {"fusion": {"method": "rrf", "norm": "max"}})
    out = engine.search(index, "target alpha bravo strong match", 100)
    assert out.trace["fusion_method"] == "rrf"
    assert out.trace["select_policy"] == "relevance_top_n"


# ---------------------------------------------------------------------------
# H2 — shipped defaults sanity (what "default config" means as of V0.5)


def test_v05_shipped_defaults():
    """The shipped V0.5 default configuration, pinned to the H5 data outcome
    (arm C7 in the ledger): semantic dense, norm_sum/minmax fusion, exact
    dropped, neighbor window:1, boost 0.01."""
    cfg = load_config()
    assert cfg.dense_cfg["backend"] == "sentence_transformers"
    assert cfg.dense_cfg["device"] == "auto"
    assert cfg.temporal_cfg["boost_delta"] == 0.01
    assert cfg.fusion_cfg["method"] == "norm_sum"
    assert cfg.fusion_cfg["norm"] == "minmax"
    assert cfg.retrievers.exact is False
    assert cfg.retrievers.bm25 is True and cfg.retrievers.dense is True
    assert cfg.surface.NEIGHBOR_POLICY == "window:1"
    assert cfg.surface.ORDERING_POLICY == "relevance_desc"
    assert cfg.surface.RETURN_SIZE == "top_k"


def test_v05_config_validation_rejects_bad_values():
    with pytest.raises(ConfigError):
        load_config(overrides={"fusion": {"method": "wrrf"}})
    with pytest.raises(ConfigError):
        load_config(overrides={"fusion": {"norm": "zscore"}})
    with pytest.raises(ConfigError):
        load_config(overrides={"dense": {"device": "tpu"}})


# ---------------------------------------------------------------------------
# H7 / G34 — SERVER_CONFIG relation accounting (no .get() folding)


def test_g34_relation_classes_are_distinct():
    from benchlib.ledger import (
        REL_NO_KEY,
        REL_NULL_VALUE,
        REL_STRICT_MATCH,
        REL_VALUE_MISMATCH,
        server_config_relation,
    )

    assert server_config_relation({"CONFIG": "x", "SERVER_CONFIG": "x"}) == REL_STRICT_MATCH
    assert server_config_relation({"CONFIG": "x"}) == REL_NO_KEY  # missing key is its own class
    assert server_config_relation({"CONFIG": "x", "SERVER_CONFIG": None}) == REL_NULL_VALUE
    assert server_config_relation({"CONFIG": "x", "SERVER_CONFIG": "y"}) == REL_VALUE_MISMATCH
    assert REL_NO_KEY != REL_NULL_VALUE != REL_STRICT_MATCH


def test_g34_missing_key_is_not_folded_into_null_or_match():
    from benchlib.ledger import REL_NO_KEY, REL_NULL_VALUE, REL_STRICT_MATCH, server_config_relation

    legacy = {"CONFIG": "old-full-config-fingerprint"}  # ledger rows 1–3 shape
    assert server_config_relation(legacy) == REL_NO_KEY
    assert server_config_relation(legacy) != REL_NULL_VALUE
    assert server_config_relation(legacy) != REL_STRICT_MATCH


def test_g34_audit_ledger_counts():
    from benchlib.ledger import audit_ledger_server_config

    records = [
        {"CONFIG": "a", "SERVER_CONFIG": "a"},
        {"CONFIG": "b"},  # legacy no-key
        {"CONFIG": "c"},  # legacy no-key
        {"CONFIG": "d", "SERVER_CONFIG": None},  # violation
        {"CONFIG": "e", "SERVER_CONFIG": "zz"},  # violation
        {"CONFIG": "f", "SERVER_CONFIG": "f"},
    ]
    audit = audit_ledger_server_config(records)
    assert audit["total_rows"] == 6
    assert audit["counts"] == {
        "strict_match": 2,
        "no_server_config_key": 2,
        "null_value": 1,
        "value_mismatch": 1,
    }
    assert audit["non_strict_rows"]["no_server_config_key"] == [2, 3]
    assert audit["non_strict_rows"]["null_value"] == [4]
    assert audit["non_strict_rows"]["value_mismatch"] == [5]


def test_g34_ledger_check_cli_on_real_ledger():
    """The real 25-row V0 ledger must account as 22 strict + 3 legacy no-key."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo / "04_bench" / "ledger_check.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LEDGER_CHECK: OK" in proc.stdout


# ---------------------------------------------------------------------------
# O2 — /health usage counters + runner measured aggregation


def test_o2_health_reports_usage_and_device(tmp_config):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    app = create_app(tmp_config())
    with TestClient(app) as c:
        body = c.get("/health").json()
    assert body["model_calls"] == 0  # NO_LLM: real zero, not a hardcoded one
    assert body["tokens"] == 0
    assert body["cost_usd"] == 0.0
    assert body["dense_device"] in ("none", "cpu", "mps", "cuda", "off")


def test_o2_runner_aggregates_usage_delta(tmp_config, tmp_path):
    """MODEL_CALLS/TOKENS/COST in the ledger record must equal the measured
    delta of the server's cumulative usage counters — not constants."""
    from aml_adapter import Usage
    from aml_api import create_app
    from aml_api.service import MemoryService
    from benchlib.locomo_loader import Conversation, LocomoQuestion, LocomoTurn
    from benchlib.runner import run_benchmark
    from fastapi.testclient import TestClient

    t = 1683000000000
    conv = Conversation(
        sample_id="mini-1",
        sessions=[
            [
                LocomoTurn(
                    dia_id="D1:1", speaker="alice", text="I adopted a dog named Barkley in May 2023.", ts_ms=t
                ),
                LocomoTurn(dia_id="D1:2", speaker="bob", text="How old is Barkley?", ts_ms=t + 60_000),
                LocomoTurn(dia_id="D1:3", speaker="alice", text="Barkley is four years old.", ts_ms=t + 120_000),
            ]
        ],
        questions=[
            LocomoQuestion(question="What is the name of the dog?", answer="Barkley", evidence=("D1:1",), category=1),
        ],
    )

    cfg = tmp_config()
    svc = MemoryService(cfg)
    app = create_app(cfg, service=svc)

    class _UsageBumpingClient:
        """Proxies the TestClient; after the first successful search it grows
        the server-side adapter usage so the runner's end snapshot must see a
        real delta."""

        def __init__(self, inner, adapter):
            self._inner = inner
            self._adapter = adapter
            self._bumped = False

        def post(self, path, **kw):
            resp = self._inner.post(path, **kw)
            if path == "/search" and resp.status_code == 200 and not self._bumped:
                self._adapter.usage.add(Usage(model_calls=2, tokens_in=11, tokens_out=7, cost_usd=0.02))
                self._bumped = True
            return resp

        def get(self, path, **kw):
            return self._inner.get(path, **kw)

    with TestClient(app) as tc:
        proxy = _UsageBumpingClient(tc, svc.adapter)
        record = run_benchmark(
            proxy,  # type: ignore[arg-type]
            [conv],
            cfg,
            run_id="o2-delta",
            results_root=tmp_path / "results",
            label="o2",
        )
    assert record["MODEL_CALLS"] == 2
    assert record["TOKENS"] == 18
    assert record["COST"] == pytest.approx(0.02)
    svc.close()
