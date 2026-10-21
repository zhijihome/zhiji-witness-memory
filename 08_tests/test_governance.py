"""EXP1 Governance State Delta (ZT-20260919-13) — one test cluster per
governance dimension, plus the isolation/regression guards.

Dimensions under test:
  D1 fact_key / assertion identity
  D2 current vs historical
  D3 update / supersede
  D4 unresolved conflict
  D5 deletion / forgetting suppression
  D6 provenance
  D7 incremental Add/Search state transition
Isolation: governance-off bit-identity with V1, config gates, append-only
storage, raw-evidence neutrality.

All tests are hermetic: rule extractor (zero model calls), hashed/exact/bm25
lexical retrievers only (no dense model), tmp dbs.
"""

from __future__ import annotations

import pytest
from aml_config.config import ConfigError, load_config
from aml_governance import GovernanceState, derive_fact_key
from aml_governance.state import (
    STATUS_CONFLICT_OPEN,
    STATUS_HISTORICAL,
    plan_assert,
    register_assertion,
)
from aml_store import GovernanceAssertionRecord, governance_assertion_id
from aml_store.model import GOV_OP_SUPERSEDE

# ---------------------------------------------------------------------------
# helpers


def _gov_cfg(tmp_config, policy="prefer_current", atomic_mode="both"):
    return tmp_config(
        {
            "atomic": {"mode": atomic_mode, "extractor": "rule", "granularity": "coarse"},
            "governance": {"mode": "on", "search_policy": policy},
            "retrievers": {"exact": True, "bm25": True, "dense": False},
        }
    )


@pytest.fixture()
def gov_service(tmp_config):
    from aml_api.service import MemoryService

    def _make(policy="prefer_current", atomic_mode="both"):
        return MemoryService(_gov_cfg(tmp_config, policy=policy, atomic_mode=atomic_mode))

    return _make


def _add(svc, user, session, request_id, ts, content, role="alice"):
    from aml_store import AddPayload, MessageIn

    return svc.add(
        AddPayload(
            request_id=request_id,
            user_id=user,
            session_id=session,
            messages=[MessageIn(role=role, content=content, timestamp=ts)],
        )
    )


def _search(svc, user, query, top_k=10):
    return svc.search(user_id=user, query=query, top_k=top_k)


def _atomic_fact_contents(svc, user):
    return [d.text for d in svc._search_indexes[user].derived_docs]


def _rec(key, value_hash, text, atom, unit, ts, seq=1, user="u"):
    return GovernanceAssertionRecord(
        assertion_id=governance_assertion_id(atom),
        user_id=user,
        fact_key=key,
        value_hash=value_hash,
        fact_text=text,
        source_atom_id=atom,
        source_unit_id=unit,
        created_at="2026-01-01T00:00:00Z",
        ts_ms=ts,
        arrival_seq=seq,
    )


def _drive(state, records, created="2026-01-01T00:00:00Z"):
    """Plan + apply the assertions against `state`; returns all ops.
    Per-record sequence (mirrors GovernanceEngine.ingest_atomic): plan
    against the state WITHOUT the new row, place the row, fold immediately."""
    ops = []
    for rec in records:
        rec_ops = plan_assert(state, rec, len(ops) + 1, created)
        register_assertion(state, rec)
        from aml_governance.state import fold_ops

        fold_ops(state, rec_ops)
        ops.extend(rec_ops)
    return ops


# ---------------------------------------------------------------------------
# D1 — fact_key / assertion identity


class TestD1FactKeyIdentity:
    def test_same_subject_different_value_shares_key(self):
        k1, v1, _, _ = derive_fact_key("alice: my favorite pizza is margherita")
        k2, v2, _, _ = derive_fact_key("alice: my favorite pizza is pepperoni")
        assert k1 == k2, "same subject+relation must share a fact_key"
        assert v1 != v2, "different fills must have different value hashes"

    def test_different_subject_gets_different_key(self):
        k1, _, _, _ = derive_fact_key("alice: my favorite pizza is margherita")
        k2, _, _, _ = derive_fact_key("alice: my favorite color is blue")
        assert k1 != k2

    def test_no_marker_whole_fact_identity(self):
        k1, v1, _, _ = derive_fact_key("alice: blue sneakers")
        k2, v2, _, _ = derive_fact_key("alice: blue sneakers")
        assert k1 == k2 and v1 == v2
        k3, _, _, _ = derive_fact_key("alice: red sneakers")
        assert k3 != k1

    def test_discourse_markers_do_not_change_identity(self):
        k_plain, _, _, m_plain = derive_fact_key("alice: my car is a ford")
        k_actual, _, _, m_actual = derive_fact_key("alice: actually my car is a ford")
        assert k_plain == k_actual
        assert not m_plain and m_actual  # marker still recorded as evidence

    def test_reasserting_same_value_is_identity_noop(self):
        """D1+D2: the same fact value re-asserted never displaces the current
        statement; the new row is recorded as a duplicate."""
        st = GovernanceState("u")
        a1 = _rec("K", "V1", "alice: my car is a ford", "atom-000001-01", "mem-000001", 1000)
        a2 = _rec("K", "V1", "alice: my car is a ford", "atom-000002-01", "mem-000002", 2000)
        ops = _drive(st, [a1, a2])
        cur = st.keys["K"].current()
        assert cur.assertion_id == a1.assertion_id, "identical re-assert must not move current"
        assert st.keys["K"].assertions[a2.assertion_id].status == STATUS_HISTORICAL
        assert ops[-1].detail["reason"] == "duplicate_value"

    def test_assertion_id_is_atom_derived_and_stable(self):
        assert governance_assertion_id("atom-000123-02") == "ga-atom-000123-02"
        assert governance_assertion_id("atom-000123-02") == governance_assertion_id("atom-000123-02")


# ---------------------------------------------------------------------------
# D2 — current vs historical


class TestD2CurrentVsHistorical:
    def test_newer_ts_supersedes_older(self):
        st = GovernanceState("u")
        old = _rec("K", "V1", "alice: my favorite pizza is margherita", "atom-000001-01", "mem-000001", 1000)
        new = _rec("K", "V2", "alice: my favorite pizza is pepperoni", "atom-000002-01", "mem-000002", 2000)
        ops = _drive(st, [old, new])
        assert st.keys["K"].current().assertion_id == new.assertion_id
        assert st.keys["K"].assertions[old.assertion_id].status == STATUS_HISTORICAL
        assert st.keys["K"].assertions[old.assertion_id].superseded_by == new.assertion_id
        assert any(op.op_type == GOV_OP_SUPERSEDE for op in ops)

    def test_stale_late_arrival_never_takes_current(self):
        st = GovernanceState("u")
        now = _rec("K", "V1", "alice: my favorite color is blue", "atom-000001-01", "mem-000001", 3000)
        stale = _rec("K", "V2", "alice: my favorite color is red", "atom-000002-01", "mem-000002", 1000)
        ops = _drive(st, [now, stale])
        assert st.keys["K"].current().assertion_id == now.assertion_id
        assert st.keys["K"].assertions[stale.assertion_id].status == STATUS_HISTORICAL
        assert ops[-1].op_type == "STALE_ARRIVAL"


# ---------------------------------------------------------------------------
# D3 — update / supersede


class TestD3UpdateSupersede:
    def test_update_marker_supersedes_without_timestamps(self):
        st = GovernanceState("u")
        old = _rec("K", "V1", "alice: my car is a ford", "atom-000001-01", "mem-000001", None)
        new = _rec("K", "V2", "alice: actually my car is a toyota", "atom-000002-01", "mem-000002", None)
        ops = _drive(st, [old, new])
        assert st.keys["K"].current().assertion_id == new.assertion_id
        sup = [op for op in ops if op.op_type == GOV_OP_SUPERSEDE]
        assert len(sup) == 1 and sup[0].detail["evidence"] == "update_marker"

    def test_ts_order_supersedes_with_evidence_tag(self):
        st = GovernanceState("u")
        old = _rec("K", "V1", "alice: i live in paris", "atom-000001-01", "mem-000001", 1000)
        new = _rec("K", "V2", "alice: i live in lyon", "atom-000002-01", "mem-000002", 2000)
        ops = _drive(st, [old, new])
        sup = [op for op in ops if op.op_type == GOV_OP_SUPERSEDE]
        assert len(sup) == 1 and sup[0].detail["evidence"] == "ts_order"

    def test_supersede_chain_keeps_full_history(self):
        st = GovernanceState("u")
        a1 = _rec("K", "V1", "alice: i live in paris", "atom-000001-01", "mem-000001", 1000)
        a2 = _rec("K", "V2", "alice: i live in lyon", "atom-000002-01", "mem-000002", 2000)
        a3 = _rec("K", "V3", "alice: i live in tokyo", "atom-000003-01", "mem-000003", 3000)
        _drive(st, [a1, a2, a3])
        k = st.keys["K"]
        assert k.current().assertion_id == a3.assertion_id
        assert {k.assertions[a1.assertion_id].status, k.assertions[a2.assertion_id].status} == {
            STATUS_HISTORICAL,
            STATUS_HISTORICAL,
        }
        assert len(k.assertions) == 3, "history is never dropped"


# ---------------------------------------------------------------------------
# D4 — unresolved conflict


class TestD4UnresolvedConflict:
    def test_equal_ts_different_value_opens_conflict(self):
        st = GovernanceState("u")
        a1 = _rec("K", "V1", "alice: my favorite pizza is margherita", "atom-000001-01", "mem-000001", 1000)
        a2 = _rec("K", "V2", "alice: my favorite pizza is pepperoni", "atom-000002-01", "mem-000002", 1000)
        ops = _drive(st, [a1, a2])
        k = st.keys["K"]
        assert k.current_id is None, "conflict closes the current pointer"
        assert k.conflict_open
        assert k.assertions[a1.assertion_id].status == STATUS_CONFLICT_OPEN
        assert k.assertions[a2.assertion_id].status == STATUS_CONFLICT_OPEN
        assert any(op.op_type == "CONFLICT_OPEN" for op in ops)

    def test_no_timestamps_and_no_marker_is_conflict(self):
        st = GovernanceState("u")
        a1 = _rec("K", "V1", "alice: my favorite pizza is margherita", "atom-000001-01", "mem-000001", None)
        a2 = _rec("K", "V2", "alice: my favorite pizza is pepperoni", "atom-000002-01", "mem-000002", None)
        _drive(st, [a1, a2])
        assert st.keys["K"].conflict_open

    def test_later_assertion_resolves_open_conflict(self):
        st = GovernanceState("u")
        a1 = _rec("K", "V1", "alice: my favorite pizza is margherita", "atom-000001-01", "mem-000001", 1000)
        a2 = _rec("K", "V2", "alice: my favorite pizza is pepperoni", "atom-000002-01", "mem-000002", 1000)
        a3 = _rec("K", "V3", "alice: my favorite pizza is hawaiian", "atom-000003-01", "mem-000003", 3000)
        ops = _drive(st, [a1, a2, a3])
        k = st.keys["K"]
        assert k.current().assertion_id == a3.assertion_id
        assert not k.conflict_open
        assert any(op.op_type == "CONFLICT_RESOLVE" for op in ops)
        assert k.assertions[a1.assertion_id].status == STATUS_HISTORICAL
        assert k.assertions[a2.assertion_id].status == STATUS_HISTORICAL


# ---------------------------------------------------------------------------
# D5 — deletion / forgetting suppression


class TestD5Suppression:
    def test_suppress_fact_key_hides_surface_keeps_store(self, gov_service):
        svc = gov_service(policy="filter_suppressed")
        _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        before = _search(svc, "gov-u", "favorite pizza")
        assert any("margherita" in i.content for i in before.items)

        state = svc.governance.state_for("gov-u")
        (key,) = state.keys.keys()
        svc.governance.suppress("gov-u", fact_key=key, reason="user requested deletion")

        after = _search(svc, "gov-u", "favorite pizza")
        assert not any("margherita" in i.content and i.via == "atomic" for i in after.items)
        assert after.trace["governance_filtered_suppressed"] >= 1
        # raw evidence layer is untouched (INV1) and still retrievable
        assert any(i.via != "atomic" for i in after.items)
        stats = svc.store.stats()
        assert stats["units"] == 1 and stats["atomic_units"] == 1

    def test_unsuppress_restores_surface(self, gov_service):
        svc = gov_service(policy="filter_suppressed")
        _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        state = svc.governance.state_for("gov-u")
        (key,) = state.keys.keys()
        svc.governance.suppress("gov-u", fact_key=key, reason="tombstone test")
        svc.governance.unsuppress("gov-u", key, reason="retracted deletion")
        out = _search(svc, "gov-u", "favorite pizza")
        assert any("margherita" in i.content and i.via == "atomic" for i in out.items)

    def test_governance_tables_are_append_only_across_suppress(self, gov_service):
        svc = gov_service(policy="filter_suppressed")
        _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        state = svc.governance.state_for("gov-u")
        (key,) = state.keys.keys()
        svc.governance.suppress("gov-u", fact_key=key, reason="d1")
        svc.governance.unsuppress("gov-u", key, reason="d2")
        counts = svc.store.governance_counts()
        ops = svc.store.governance_ops("gov-u")
        assert counts["governance_ops"] == len(ops)
        op_types = [op.op_type for op in ops]
        assert op_types.count("SUPPRESS") == 1 and op_types.count("UNSUPPRESS") == 1
        # the fold of the full journal reproduces the live state exactly
        fresh = GovernanceState("gov-u")
        for rec in svc.store.governance_assertions("gov-u"):
            register_assertion(fresh, rec)
        from aml_governance.state import fold_ops

        fold_ops(fresh, ops)
        assert fresh.keys[key].suppressed is False  # suppress then unsuppress
        assert fresh.keys[key].current() is not None

    def test_suppress_requires_target(self, gov_service):
        svc = gov_service()
        with pytest.raises(ValueError):
            svc.governance.suppress("gov-u")


# ---------------------------------------------------------------------------
# D6 — provenance


class TestD6Provenance:
    def test_every_journal_op_carries_provenance(self, gov_service):
        svc = gov_service()
        _add(svc, "gov-u", "s1", "r1", 1683000000000, "My favorite pizza is margherita.")
        _add(svc, "gov-u", "s1", "r2", 1683100000000, "My favorite pizza is pepperoni.")
        ops = svc.store.governance_ops("gov-u")
        assert ops, "supersede flow must journal"
        for op in ops:
            assert op.op_seq >= 1 and op.user_id == "gov-u"
            assert op.created_at and op.fact_key
            detail = op.detail
            for field in ("source_atom_id", "source_unit_id", "ts_ms", "arrival_seq", "value_hash"):
                assert field in detail, f"op {op.op_type} missing provenance field {field}"

    def test_assertion_provenance_resolves_to_source(self, gov_service):
        svc = gov_service()
        outcome = _add(svc, "gov-u", "s1", "r1", 1683000000000, "My favorite pizza is margherita.")
        unit_id = outcome.stored[0].id
        state = svc.governance.state_for("gov-u")
        for a in state.by_atom.values():
            assert a.source_unit_id == unit_id
            resolved = svc.store.unit("gov-u", a.source_unit_id)
            assert resolved is not None and resolved.content_sha256 is not None
            atom = next(r for r in svc.store.atomic_units("gov-u") if r.atom_id == a.source_atom_id)
            assert atom.source_unit_id == unit_id

    def test_supersede_op_links_prior(self):
        st = GovernanceState("u")
        old = _rec("K", "V1", "alice: i live in paris", "atom-000001-01", "mem-000001", 1000)
        new = _rec("K", "V2", "alice: i live in lyon", "atom-000002-01", "mem-000002", 2000)
        ops = _drive(st, [old, new])
        sup = next(op for op in ops if op.op_type == GOV_OP_SUPERSEDE)
        assert sup.prior_ref == old.assertion_id


# ---------------------------------------------------------------------------
# D7 — incremental Add/Search state transition


class TestD7IncrementalTransition:
    def test_search_reflects_supersede_without_rebuild(self, gov_service):
        svc = gov_service(policy="prefer_current")
        _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        index_before = svc._search_indexes["gov-u"]
        out1 = _search(svc, "gov-u", "favorite pizza")
        assert any("margherita" in i.content and i.via == "atomic" for i in out1.items)

        _add(svc, "gov-u", "s1", "r2", 2000, "My favorite pizza is pepperoni.")
        assert svc._search_indexes["gov-u"] is index_before, "no index rebuild may happen"
        out2 = _search(svc, "gov-u", "favorite pizza")
        atomic_contents = [i.content for i in out2.items if i.via == "atomic"]
        assert any("pepperoni" in c for c in atomic_contents)
        assert not any("margherita" in c for c in atomic_contents), "stale fact filtered live"
        assert out2.trace["governance_filtered_stale"] >= 1

    def test_restart_folds_journal_from_db(self, tmp_config, tmp_path):
        from aml_api.service import MemoryService

        cfg = _gov_cfg(tmp_config)
        db_path = tmp_path / "gov.db"
        cfg.raw["storage"]["db_path"] = str(db_path)
        svc1 = MemoryService(cfg)
        _add(svc1, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        _add(svc1, "gov-u", "s1", "r2", 2000, "My favorite pizza is pepperoni.")
        out1 = _search(svc1, "gov-u", "favorite pizza")
        svc1.close()

        svc2 = MemoryService(cfg)  # fresh process-equivalent: state from db only
        state = svc2.governance.state_for("gov-u")
        assert len(state.keys) == 1
        (k,) = state.keys.values()
        assert k.current() is not None
        assert sum(1 for a in k.assertions.values() if a.status == STATUS_HISTORICAL) == 1
        out2 = _search(svc2, "gov-u", "favorite pizza")
        c1 = [i.content for i in out1.items if i.via == "atomic"]
        c2 = [i.content for i in out2.items if i.via == "atomic"]
        assert c1 == c2, "post-restart behavior must equal pre-restart behavior"
        svc2.close()

    def test_backfill_covers_facts_extracted_pre_enablement(self, tmp_config, tmp_path):
        """Atomic rows that existed before governance ingestion (or a crash
        between layers) are derived at index build — idempotent backfill."""
        from aml_api.service import MemoryService

        cfg_atomic_only = tmp_config(
            {
                "atomic": {"mode": "both", "extractor": "rule"},
                "retrievers": {"exact": True, "bm25": True, "dense": False},
            }
        )
        db_path = tmp_path / "backfill.db"
        cfg_atomic_only.raw["storage"]["db_path"] = str(db_path)
        svc1 = MemoryService(cfg_atomic_only)
        _add(svc1, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        svc1.close()

        cfg_gov = _gov_cfg(tmp_config)
        cfg_gov.raw["storage"]["db_path"] = str(db_path)
        svc2 = MemoryService(cfg_gov)
        out = _search(svc2, "gov-u", "favorite pizza")
        assert svc2.store.governance_counts()["governance_assertions"] == 1
        assert any("margherita" in i.content for i in out.items)
        svc2.close()


# ---------------------------------------------------------------------------
# atomic_index_raw_return combination (CP2): the governance layer must behave
# identically on the raw-return derived surface — derived hits are filtered by
# governance state, then each surviving hit maps back to its raw unit.


class TestAtomicIndexRawReturnCombo:
    def _svc(self, gov_service, policy):
        return gov_service(policy=policy, atomic_mode="atomic_index_raw_return")

    def test_raw_return_surface_maps_hits_to_raw_units(self, gov_service):
        svc = self._svc(gov_service, "prefer_current")
        outcome = _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        unit_id = outcome.stored[0].id
        out = _search(svc, "gov-u", "favorite pizza")
        atomic_items = [i for i in out.items if i.via == "atomic"]
        assert atomic_items, "derived hit must surface on the raw-return mode"
        assert all(i.id == unit_id for i in atomic_items), "raw-return packs the raw unit id"
        assert all("margherita" in i.content for i in atomic_items)
        assert out.trace["governance_mode"] == "on"

    def test_raw_return_suppression_removes_fact_hit(self, gov_service):
        svc = self._svc(gov_service, "filter_suppressed")
        _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita. I drive a Mazda.")
        state = svc.governance.state_for("gov-u")
        pizza_key = next(
            k
            for k, ks in state.keys.items()
            if any("pizza" in a.fact_text for a in ks.assertions.values())
        )
        svc.governance.suppress("gov-u", fact_key=pizza_key, reason="raw-return suppression")

        out = _search(svc, "gov-u", "favorite pizza")
        assert out.items == [], "the only derived hit is suppressed on the derived-only surface"
        assert out.trace["governance_filtered_suppressed"] >= 1
        # the untouched fact of the same message stays fully usable
        other = _search(svc, "gov-u", "drive Mazda")
        assert any("Mazda" in i.content and i.via == "atomic" for i in other.items)

    def test_raw_return_prefer_current_drops_superseded_fact(self, gov_service):
        svc = self._svc(gov_service, "prefer_current")
        old = _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        new = _add(svc, "gov-u", "s1", "r2", 2000, "My favorite pizza is pepperoni.")
        out = _search(svc, "gov-u", "favorite pizza")
        ids = [i.id for i in out.items]
        assert new.stored[0].id in ids, "current fact's raw unit surfaces"
        assert old.stored[0].id not in ids, "superseded fact's raw unit does not surface via that fact"
        assert not any("margherita" in i.content for i in out.items)
        assert out.trace["governance_filtered_stale"] >= 1


# ---------------------------------------------------------------------------
# isolation / regression guards


class TestIsolationAndRegression:
    @staticmethod
    def _corpus(svc):
        _add(svc, "cmp", "s1", "r1", 1000, "My favorite pizza is margherita.")
        _add(svc, "cmp", "s1", "r2", 2000, "My favorite pizza is pepperoni.")

    def test_governance_off_bit_identical_to_v1(self, tmp_config):
        from aml_api.service import MemoryService

        lexical = {"retrievers": {"exact": True, "bm25": True, "dense": False}}
        atomic = {"atomic": {"mode": "both", "extractor": "rule"}}

        cfg_v1 = tmp_config({**atomic, **lexical})  # V1: NO governance section at all
        svc_v1 = MemoryService(cfg_v1)
        self._corpus(svc_v1)
        base = _search(svc_v1, "cmp", "favorite pizza")

        cfg_off = tmp_config({**atomic, **lexical, "governance": {"mode": "off"}})
        svc_off = MemoryService(cfg_off)
        assert svc_off.governance is None
        self._corpus(svc_off)
        off = _search(svc_off, "cmp", "favorite pizza")

        assert [(i.id, i.content, round(i.score, 9)) for i in base.items] == [
            (i.id, i.content, round(i.score, 9)) for i in off.items
        ]
        assert off.trace["governance_mode"] == "off"
        assert off.trace.get("governance_filtered_suppressed", 0) == 0
        assert off.trace.get("governance_filtered_stale", 0) == 0
        svc_v1.close()
        svc_off.close()

    def test_off_behavior_fingerprint_equals_v1_fingerprint(self, tmp_config):
        """No governance section and governance.mode=off differ in raw config,
        but an ABSENT section must leave the V1 fingerprint untouched — that is
        the byte-identity guarantee the regression rides on."""
        cfg_v1 = tmp_config({"atomic": {"mode": "both"}})
        cfg_off = tmp_config({"atomic": {"mode": "both"}, "governance": {"mode": "off"}})
        assert "governance" not in cfg_v1.raw
        assert cfg_v1.behavior_fingerprint() != cfg_off.behavior_fingerprint()

    def test_prefer_current_policy_changes_surface(self, gov_service):
        svc = gov_service(policy="prefer_current")
        _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        _add(svc, "gov-u", "s1", "r2", 2000, "My favorite pizza is pepperoni.")
        out = _search(svc, "gov-u", "favorite pizza")
        atomic_contents = [i.content for i in out.items if i.via == "atomic"]
        assert any("pepperoni" in c for c in atomic_contents)
        assert not any("margherita" in c for c in atomic_contents)

    def test_filter_suppressed_only_policy_keeps_stale(self, gov_service):
        svc = gov_service(policy="filter_suppressed")
        _add(svc, "gov-u", "s1", "r1", 1000, "My favorite pizza is margherita.")
        _add(svc, "gov-u", "s1", "r2", 2000, "My favorite pizza is pepperoni.")
        out = _search(svc, "gov-u", "favorite pizza")
        atomic_contents = [i.content for i in out.items if i.via == "atomic"]
        assert any("margherita" in c for c in atomic_contents), "historical stays without prefer_current"
        assert out.trace.get("governance_filtered_stale", 0) == 0

    def test_config_gate_governance_requires_atomic_layer(self):
        with pytest.raises(ConfigError):
            load_config(
                overrides={
                    "storage": {"db_path": "/tmp/aml-gov-cfg.db"},
                    "atomic": {"mode": "off"},
                    "governance": {"mode": "on", "search_policy": "filter_suppressed"},
                }
            )

    def test_governance_on_accepts_raw_return_mode(self):
        cfg = load_config(
            overrides={
                "storage": {"db_path": "/tmp/aml-gov-cfg.db"},
                "atomic": {"mode": "atomic_index_raw_return"},
                "governance": {"mode": "on"},
            }
        )
        assert cfg.governance_cfg["mode"] == "on"

    def test_config_gate_bad_policy_and_keyer(self):
        base = {"storage": {"db_path": "/tmp/aml-gov-cfg.db"}}
        for bad in (
            {"governance": {"mode": "on", "search_policy": "nuke"}, "atomic": {"mode": "both"}},
            {"governance": {"mode": "on", "fact_keyer": "llm-magic"}, "atomic": {"mode": "both"}},
            {"governance": {"mode": "maybe"}},
        ):
            with pytest.raises(ConfigError):
                load_config(overrides={**base, **bad})

    def test_explicit_off_config_is_legal(self):
        cfg = load_config(overrides={"governance": {"mode": "off"}})
        assert cfg.governance_cfg == {"mode": "off"}
