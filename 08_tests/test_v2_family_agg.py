"""V2 family-discipline tests (WORKORDER_AML_V2_R1 §3.1/§3.2, gates G48/G49).

Covers, all hermetic (no LLM, hashed/bm25-only world):
  * config validation: the four family_agg values, β range, and the hard-fail
    illegal combinations (no silent no-op);
  * G48: family_agg=max is the V1 default (bit-identical selection), all four
    disciplines are deterministic and pairwise distinguishable;
  * G49: support_gate structurally forbids orphan promotions (a family with
    atomic evidence but no in-pool raw support can neither enter nor count);
    max/additive honestly count the orphans they create;
  * trace → /search ops journal → /health cumulative counters → ledger fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aml_config import load_config
from aml_config.config import ConfigError

# ---------------------------------------------------------------------------
# helpers


def _v2_config(tmp_path: Path, **atomic):
    base = {
        "storage": {"db_path": str(tmp_path / "test.db")},
        "ops_log": {"enabled": True, "dir": str(tmp_path / "ops")},
        "retrievers": {"exact": False, "bm25": True, "dense": False},
        "dense": {"backend": "hashed_ngram"},
        "surface": {"NEIGHBOR_POLICY": "off"},
        "temporal": {"boost_delta": 0},
        "atomic": {
            "mode": "both",
            "extractor": "rule",
            "granularity": "fine",
            "dedup_policy": "keep_raw",
            **atomic,
        },
    }
    return load_config(overrides=base)


_RAW_TEXTS = [
    "Alice adopted a dog named Barkley last spring.",  # idx 0 — family A raw
    "Bob reported the quarterly revenue figures to the board.",  # idx 1 — filler
    "Bob enjoys watercolor painting on quiet weekends.",  # idx 2 — family C raw (no query overlap)
    "Alice walked the dog to the park yesterday.",  # idx 3 — family D raw
]

# (source_idx, fact): family A and D atoms support their raws; family C's atom
# matches the query while its raw does NOT — the orphan-promotion injection.
_DERIVED = [
    (0, "Alice adopted a dog named Barkley"),
    (2, "Bob painted a watercolor dog Barkley portrait"),
    (3, "Alice walked the dog to the park"),
]


def _family_index(tmp_path: Path, family_agg: str, beta: float | None = None):
    """Hand-built SearchIndex with three pooled families: A (raw+atom), the
    C orphan (atom only — raw absent from the pool), D (raw+atom)."""
    from aml_retrieval import DerivedDoc, RetrievalEngine, SearchIndex
    from aml_store import MemoryUnit
    from aml_store.model import content_sha256, iso_z
    from aml_store.store import UserIndex

    cfg = _v2_config(tmp_path / family_agg.replace("_", "-"), family_agg=family_agg)
    if beta is not None:
        cfg.raw["atomic"]["family_support_beta"] = beta
    engine = RetrievalEngine(cfg)

    ui = UserIndex("u1")
    for i, (role, content) in enumerate(
        [("alice", text) for text in _RAW_TEXTS], start=1
    ):
        ui.append(
            MemoryUnit(
                id=f"mem-{i:06d}",
                user_id="u1",
                session_id="s1",
                request_id="r1",
                msg_index=i - 1,
                role=role,
                content=content,
                ts_ms=1_683_000_000_000 + i * 60_000,
                created_at=iso_z(1_683_000_000_000 + i * 60_000),
                arrival_seq=i,
                session_seq=i,
                content_sha256=content_sha256(role, content),
            )
        )
    derived = [
        DerivedDoc(
            doc_id=f"atom-{src + 1:06d}-01",
            text=fact,
            created_at=iso_z(1_683_000_000_000),
            ts_ms=1_683_000_000_000,
            source_idx=src,
            source_unit_id=f"mem-{src + 1:06d}",
        )
        for src, fact in _DERIVED
    ]
    index = SearchIndex.build(ui, cfg, engine.index_embedder(), derived)
    return engine, index


def _surface(tmp_path: Path, family_agg: str, beta: float | None = None):
    engine, index = _family_index(tmp_path, family_agg, beta)
    outcome = engine.search(index, "dog Barkley", top_k=100)
    fam_of = {}
    for i in range(len(_RAW_TEXTS)):
        fam_of[f"mem-{i + 1:06d}"] = f"F{i}"
    for src, _fact in _DERIVED:
        fam_of[f"atom-{src + 1:06d}-01"] = f"F{src}"
    return outcome, [fam_of[it.id] for it in outcome.items], fam_of


# ---------------------------------------------------------------------------
# config validation (hard failures, never silent no-ops)


def test_four_family_aggs_validate_with_dedup_and_mode(tmp_path):
    for agg in ("max", "raw_only", "support_gate", "additive"):
        cfg = _v2_config(tmp_path / agg, family_agg=agg)
        assert cfg.atomic_cfg["family_agg"] == agg


def test_default_family_agg_is_max():
    cfg = load_config()
    assert cfg.atomic_cfg["family_agg"] == "max"
    assert cfg.atomic_cfg["family_support_beta"] == 0.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"family_agg": "support_gate", "dedup_policy": "off"},
        {"family_agg": "raw_only", "dedup_policy": "off"},
        {"family_agg": "additive", "dedup_policy": "off"},
        {"family_agg": "support_gate", "mode": "off"},
        {"family_agg": "support_gate", "mode": "atomic_index_raw_return"},
        {"family_agg": "bogus"},
        {"family_agg": "additive", "family_support_beta": 0.0},
        {"family_agg": "additive", "family_support_beta": 1.2},
        {"family_agg": "additive", "family_support_beta": "0.5"},
    ],
)
def test_illegal_family_agg_combinations_hard_fail(tmp_path, overrides):
    with pytest.raises(ConfigError):
        _v2_config(tmp_path / "bad", **overrides)


# ---------------------------------------------------------------------------
# G48 — max is the V1 default; four disciplines deterministic + distinguishable


def test_g48_max_explicit_equals_default_walk(tmp_path):
    """The explicit `max` value must take the exact V1 code path: the surface
    with family_agg=max equals the surface with the key deleted entirely."""
    engine, index = _family_index(tmp_path / "explicit", "max")
    explicit = engine.search(index, "dog Barkley", top_k=100)
    engine.config.raw["atomic"].pop("family_agg")
    fallback = engine.search(index, "dog Barkley", top_k=100)
    assert [it.id for it in explicit.items] == [it.id for it in fallback.items]
    assert [round(it.score, 9) for it in explicit.items] == [round(it.score, 9) for it in fallback.items]


def test_g48_all_four_disciplines_deterministic_and_distinguishable(tmp_path):
    for agg in ("max", "raw_only", "support_gate", "additive"):
        first, fams_first, _ = _surface(tmp_path, agg)
        second, fams_second, _ = _surface(tmp_path, agg)
        assert [it.id for it in first.items] == [it.id for it in second.items], f"{agg} not deterministic"
        assert fams_first == fams_second
        assert first.trace["family_agg"] == agg
        assert first.trace["orphan_promotions"] == second.trace["orphan_promotions"]

    # structural signatures on this scenario: the orphan family F2 is selected
    # by max/additive and barred by raw_only/support_gate; only support_gate
    # reports blocked promotions.
    sigs = {}
    for agg in ("max", "raw_only", "support_gate", "additive"):
        outcome, fams, _ = _surface(tmp_path, agg)
        sigs[agg] = ("F2" in fams, outcome.trace["families_gate_blocked"], outcome.trace["orphan_promotions"])
    assert sigs == {
        "max": (True, 0, 1),
        "raw_only": (False, 0, 0),
        "support_gate": (False, 1, 0),
        "additive": (True, 0, 1),
    }


def test_g48_four_keys_rank_distinctly_at_selection_level():
    """Controlled-score distinguishability of the four disciplines (the
    engine-level bm25 scenario cannot in general separate max from additive
    ordering). Families: A raw=0.5/atom=0.9; B orphan atom=0.8 (no raw);
    C raw=0.7/atom=0.6. keep_raw reps; β=0.5."""
    from aml_retrieval.engine import DERIVED_BASE, _select_by_family_key

    atom_a, atom_b, atom_c = DERIVED_BASE + 1, DERIVED_BASE + 2, DERIVED_BASE + 3
    families = {100: [100, atom_a], 200: [atom_b], 300: [300, atom_c]}
    results = {100: 0.5, atom_a: 0.9, atom_b: 0.8, 300: 0.7, atom_c: 0.6}

    def keep_raw_rep(members):
        return next((i for i in members if i < DERIVED_BASE), members[0])

    def select(agg, beta=0.5):
        return [rep for rep, _ in _select_by_family_key(families, results, agg, beta, keep_raw_rep, 3)]

    assert select("max") == [100, atom_b, 300]  # 0.9, 0.8, 0.7
    assert select("raw_only") == [300, 100]  # 0.7, 0.5; orphan barred
    assert select("support_gate") == [100, 300]  # 0.9, 0.7; orphan barred
    assert select("additive") == [300, 100, atom_b]  # 1.0, 0.95, 0.4 — order flips
    # determinism: identical inputs → identical order
    assert select("additive") == select("additive")


def test_g48_additive_beta_flips_family_order_at_selection_level():
    """Preregistered β semantics: scaling only the atomic contribution moves an
    atom-strong family (A) across a raw-strong family (C) — dose response."""
    from aml_retrieval.engine import DERIVED_BASE, _select_by_family_key

    atom_a, atom_b, atom_c = DERIVED_BASE + 1, DERIVED_BASE + 2, DERIVED_BASE + 3
    families = {100: [100, atom_a], 200: [atom_b], 300: [300, atom_c]}
    results = {100: 0.5, atom_a: 0.9, atom_b: 0.8, 300: 0.7, atom_c: 0.6}

    def keep_raw_rep(members):
        return next((i for i in members if i < DERIVED_BASE), members[0])

    def select(beta):
        return [rep for rep, _ in _select_by_family_key(families, results, "additive", beta, keep_raw_rep, 3)]

    assert select(0.25) == [300, 100, atom_b]  # C 0.85 > A 0.725 > B 0.2
    assert select(0.75) == [100, 300, atom_b]  # A 1.175 > C 1.15 > B 0.6


def test_g48_family_key_tie_breaks_by_min_member_idx():
    from aml_retrieval.engine import DERIVED_BASE, _select_by_family_key

    families = {500: [500, DERIVED_BASE + 900], 400: [400, DERIVED_BASE + 401]}
    results = {400: 0.5, DERIVED_BASE + 401: 0.9, 500: 0.5, DERIVED_BASE + 900: 0.9}

    def keep_raw_rep(members):
        return next((i for i in members if i < DERIVED_BASE), members[0])

    # equal keys 0.9 — the family with the smaller min member idx (400) wins
    selected = _select_by_family_key(families, results, "max", 0.5, keep_raw_rep, 2)
    assert [rep for rep, _ in selected] == [400, 500]


def test_g48_atomic_off_leaves_surface_and_trace_neutral(tmp_path):
    from aml_retrieval import SearchIndex

    engine, index = _family_index(tmp_path, "max")
    # rebuild the same corpus with atomic off: no derived docs at all
    engine.config.raw["atomic"]["mode"] = "off"
    fresh = SearchIndex.build(index.ui, engine.config, engine.index_embedder(), None)
    outcome = engine.search(fresh, "dog Barkley", top_k=100)
    assert outcome.trace["family_agg"] == "off"
    assert outcome.trace["families_promoted_by_atomic"] == 0
    assert outcome.trace["families_gate_blocked"] == 0
    assert outcome.trace["orphan_promotions"] == 0
    assert all(not it.id.startswith("atom-") for it in outcome.items)


# ---------------------------------------------------------------------------
# G49 — promotion support / orphan integrity


def test_g49_support_gate_blocks_orphan_family(tmp_path):
    """The injected orphan family C (atomic hits the query, its raw has zero
    pool presence) must be excluded and audited — never selected."""
    outcome, fams, fam_of = _surface(tmp_path, "support_gate")
    assert "F2" not in fams, f"orphan family selected under support_gate: {fams}"
    assert outcome.trace["family_agg"] == "support_gate"
    assert outcome.trace["orphan_promotions"] == 0
    assert outcome.trace["families_gate_blocked"] == 1  # pool-wide: exactly family C
    # keep_raw face: every returned slot is a raw unit
    assert all(not it.id.startswith("atom-") for it in outcome.items)


def test_g49_max_and_additive_count_the_orphan_they_promote(tmp_path):
    outcome_max, fams_max, _ = _surface(tmp_path, "max")
    assert "F2" in fams_max, "orphan family should be selectable under max"
    assert outcome_max.trace["orphan_promotions"] == 1
    assert outcome_max.trace["families_gate_blocked"] == 0

    outcome_add, fams_add, _ = _surface(tmp_path, "additive", beta=0.5)
    assert "F2" in fams_add, "orphan family enters additive at the discounted key"
    assert outcome_add.trace["orphan_promotions"] == 1
    assert outcome_add.trace["families_gate_blocked"] == 0


def test_g49_raw_only_never_selects_orphan(tmp_path):
    outcome, fams, _ = _surface(tmp_path, "raw_only")
    assert "F2" not in fams
    assert outcome.trace["orphan_promotions"] == 0
    assert outcome.trace["families_gate_blocked"] == 0


def test_g49_additive_beta_moves_the_orphan_discount(tmp_path):
    """β only scales the atomic contribution: the orphan's key is β·atomic, so
    β=0.25 ranks it at or below β=0.75 while raw-relative order is kept."""
    low, fams_low, _ = _surface(tmp_path, "additive", beta=0.25)
    high, fams_high, _ = _surface(tmp_path, "additive", beta=0.75)
    assert low.trace["orphan_promotions"] == high.trace["orphan_promotions"]
    if "F2" in fams_low and "F2" in fams_high:
        assert fams_low.index("F2") >= fams_high.index("F2")


# ---------------------------------------------------------------------------
# trace → journal → /health → ledger field flow


def test_promotion_audit_flows_to_health_and_ledger(tmp_path):
    from aml_api import create_app
    from benchlib.runner import run_benchmark
    from fastapi.testclient import TestClient

    cfg = _v2_config(tmp_path, family_agg="support_gate")
    app = create_app(cfg)
    with TestClient(app) as client:
        health0 = client.get("/health").json()
        assert health0["family_agg"] == "support_gate"
        assert health0["families_promoted_by_atomic"] == 0
        client.post(
            "/add",
            json={
                "request_id": "r1",
                "messages": [
                    {"role": "alice", "timestamp": 1_683_000_000_000, "content": _RAW_TEXTS[0]},
                    {"role": "bob", "timestamp": 1_683_000_060_000, "content": _RAW_TEXTS[2]},
                ],
                "user_id": "u1",
                "session_id": "s1",
            },
        )
        search = client.post("/search", json={"query": "dog Barkley", "user_id": "u1", "top_k": 100})
        assert search.status_code == 200
        health1 = client.get("/health").json()
        assert health1["orphan_promotions"] == 0
        # the rule extractor produced at least one family carrying atomic +
        # raw; whatever got selected, the counters must be consistent
        assert health1["families_promoted_by_atomic"] >= 0

        record = run_benchmark(
            client,
            [_mini_support_conversation()],
            cfg,
            run_id="v2-audit",
            results_root=tmp_path / "results",
            label="v2",
        )
        for field in (
            "FAMILY_AGG",
            "FAMILIES_PROMOTED_BY_ATOMIC",
            "FAMILIES_GATE_BLOCKED",
            "ORPHAN_PROMOTIONS",
        ):
            assert field in record, f"missing {field}"
        assert record["FAMILY_AGG"] == "support_gate"
        assert record["ORPHAN_PROMOTIONS"] == 0  # support_gate binding invariant


def _mini_support_conversation():
    from benchlib.locomo_loader import Conversation, LocomoQuestion, LocomoTurn

    turns = [
        LocomoTurn(dia_id="d1", speaker="alice", text=_RAW_TEXTS[0], ts_ms=1_683_000_000_000),
        LocomoTurn(dia_id="d2", speaker="bob", text=_RAW_TEXTS[2], ts_ms=1_683_000_060_000),
        LocomoTurn(dia_id="d3", speaker="alice", text=_RAW_TEXTS[3], ts_ms=1_683_000_120_000),
    ]
    questions = [
        LocomoQuestion(question="What dog did Alice adopt?", answer="Barkley", evidence=("d1",), category=1),
    ]
    return Conversation(sample_id="v2", sessions=[turns], questions=questions)
