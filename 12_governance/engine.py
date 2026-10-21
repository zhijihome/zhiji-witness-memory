"""GovernanceEngine (EXP1) — derives assertions from atomic facts, plans
journal ops, persists them insert-only, and folds the per-user state
incrementally.

Flow per Add (after the raw commit and the atomic extraction):

  atomic records → per fact: derive_fact_key (rule-gk1) → plan_assert against
  the current folded state → persist assertion rows + ops (store, insert-only)
  → apply the same ops to the in-memory state → the per-user view attached to
  the live SearchIndex sees the new state immediately (dimension 7: incremental
  Add/Search state transition, no index rebuild).

玄神经 invariants, same family as 11_atomic (G37): this engine only READS
atomic fact text and only WRITES the two governance tables. No code path here
touches `memories` or `atomic_units`. Zero model calls by construction — the
keyer is rule-based text processing (12_governance.factkey).
"""

from __future__ import annotations

from datetime import UTC, datetime

from aml_config.config import AMLConfig
from aml_store import (
    CompetitionMemoryStore,
    GovernanceAssertionRecord,
    GovernanceOpRecord,
    governance_assertion_id,
)

from .factkey import derive_fact_key
from .state import (
    GovernanceState,
    fold_ops,
    plan_assert,
    plan_suppress,
    register_assertion,
)


class GovernanceEngine:
    def __init__(self, config: AMLConfig, store: CompetitionMemoryStore) -> None:
        self.config = config
        self.store = store
        gov = config.governance_cfg
        self.mode = str(gov.get("mode", "off"))
        self.search_policy = str(gov.get("search_policy", "filter_suppressed"))
        self.fact_keyer = str(gov.get("fact_keyer", "rule-gk1"))
        if self.fact_keyer != "rule-gk1":
            raise ValueError(f"unknown governance.fact_keyer: {self.fact_keyer!r}")
        self._states: dict[str, GovernanceState] = {}

    # -- state views -----------------------------------------------------------

    def state_for(self, user_id: str) -> GovernanceState:
        state = self._states.get(user_id)
        if state is None:
            state = self._load_state(user_id)
            self._states[user_id] = state
        return state

    def view_for(self, user_id: str) -> GovernanceView:
        """The read-side view attached to a SearchIndex (dimension 7: the live
        index sees state changes without any rebuild)."""
        return GovernanceView(self.state_for(user_id), self.search_policy)

    def _load_state(self, user_id: str) -> GovernanceState:
        """Rebuild state from persisted rows: assertions first (row placement),
        then the journal fold. Pure db-derived — restart-safe (event sourcing)."""
        state = GovernanceState(user_id)
        for record in self.store.governance_assertions(user_id):
            register_assertion(state, record)
        fold_ops(state, self.store.governance_ops(user_id))
        return state

    # -- ingest ------------------------------------------------------------------

    def ingest_atomic(self, user_id: str, records: list) -> list[GovernanceOpRecord]:
        """Derive governance assertions from new atomic records (factless
        marker rows are skipped), plan + persist ops, fold them in. Returns
        the appended ops for callers that want to assert on them."""
        facts = [r for r in records if r.fact.strip()]
        if not facts:
            return []
        state = self.state_for(user_id)
        op_seq = self.store.governance_next_op_seq(user_id)
        now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        assertion_rows: list[GovernanceAssertionRecord] = []
        ops: list[GovernanceOpRecord] = []
        for rec in facts:
            fact_key, value_hash, _norm, _marker = derive_fact_key(rec.fact)
            k_state = state.keys.get(fact_key)
            row = GovernanceAssertionRecord(
                assertion_id=governance_assertion_id(rec.atom_id),
                user_id=user_id,
                fact_key=fact_key,
                value_hash=value_hash,
                fact_text=rec.fact,
                source_atom_id=rec.atom_id,
                source_unit_id=rec.source_unit_id,
                created_at=rec.created_at,
                ts_ms=rec.ts_ms,
                arrival_seq=self._arrival_seq(user_id, rec),
            )
            if k_state is not None and row.assertion_id in k_state.assertions:
                continue  # idempotent re-ingest (crash between store write and fold)
            # per-record sequence: PLAN against the state WITHOUT this row,
            # place the row, then apply the ops immediately — so the next
            # record's planning sees the transitions this record produced.
            rec_ops = plan_assert(state, row, op_seq + len(ops), now)
            register_assertion(state, row)
            fold_ops(state, rec_ops)
            ops.extend(rec_ops)
            assertion_rows.append(row)

        if not ops:
            return []
        self.store.insert_governance_assertions(assertion_rows)
        self.store.append_governance_ops(ops)
        return ops

    def _arrival_seq(self, user_id: str, rec) -> int:
        """Governance ordering borrows the raw layer's arrival order. The atom
        id embeds the source arrival seq (`atom-<seq>-<k>`), so this is
        recoverable from the id alone — no read of the memories table."""
        try:
            return int(rec.atom_id.split("-")[1])
        except (IndexError, ValueError):  # pragma: no cover - defensive
            return 0

    # -- suppression (deletion / forgetting dimension) -------------------------

    def suppress(
        self, user_id: str, fact_key: str = "", assertion_id: str = "", reason: str = ""
    ) -> list[GovernanceOpRecord]:
        if not fact_key and not assertion_id:
            raise ValueError("suppress requires fact_key or assertion_id")
        if assertion_id and not fact_key:
            state = self.state_for(user_id)
            found = next(
                (a for k in state.keys.values() for a in k.assertions.values() if a.assertion_id == assertion_id),
                None,
            )
            if found is None:
                raise ValueError(f"unknown governance assertion: {assertion_id!r}")
            fact_key = found.fact_key
        return self._apply_suppression(user_id, fact_key, assertion_id, reason, unsuppress=False)

    def unsuppress(self, user_id: str, fact_key: str, reason: str = "") -> list[GovernanceOpRecord]:
        return self._apply_suppression(user_id, fact_key, "", reason, unsuppress=True)

    def _apply_suppression(
        self, user_id: str, fact_key: str, assertion_id: str, reason: str, *, unsuppress: bool
    ) -> list[GovernanceOpRecord]:
        state = self.state_for(user_id)
        op_seq = self.store.governance_next_op_seq(user_id)
        now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        ops = plan_suppress(state, user_id, fact_key, assertion_id, reason, op_seq, now, unsuppress=unsuppress)
        self.store.append_governance_ops(ops)
        fold_ops(state, ops)
        return ops


class GovernanceView:
    """Read-only projection handed to the retrieval layer. Lives on the
    SearchIndex (`index.governance`); because it wraps the live GovernanceState
    object, add-time folds are visible to search immediately."""

    def __init__(self, state: GovernanceState, search_policy: str) -> None:
        self.state = state
        self.search_policy = search_policy

    def is_suppressed_atom(self, source_atom_id: str) -> bool:
        a = self.state.assertion_by_atom(source_atom_id)
        if a is None:
            return False
        k = self.state.keys.get(a.fact_key)
        return a.suppressed or (k is not None and k.suppressed)

    def is_stale_atom(self, source_atom_id: str) -> bool:
        """Superseded/conflicted assertion whose key HAS a live current
        representative — the prefer_current drop class."""
        a = self.state.assertion_by_atom(source_atom_id)
        if a is None or a.suppressed:
            return False
        return a.status != "current" and self.state.key_has_live_current(a.fact_key)

    def stats(self) -> dict[str, int]:
        keys = self.state.keys.values()
        return {
            "fact_keys": len(self.state.keys),
            "assertions": sum(len(k.assertions) for k in keys),
            "conflicts_open": sum(1 for k in keys if k.conflict_open),
            "keys_suppressed": sum(1 for k in keys if k.suppressed),
        }
