"""Governance fold state (EXP1) — event-sourced, deterministic.

The per-user governance state is DEFINED as the fold of the append-only
governance journal (`governance_journal` table). `apply_op` is the single
transition function: the live engine computes ops against the in-memory state,
persists them, then applies the very same function — and a restart rebuilds
the state by folding the journal rows from the db. State and record can never
drift because there is exactly one transition implementation.

Statuses (current vs historical dimension):
  current        — the standing assertion for a fact_key (at most one)
  historical     — superseded, duplicate, or stale-late-arriving assertions
  conflict_open  — an assertion engaged in an unresolved same-key conflict

Deletion / forgetting suppression is a READ-SURFACE overlay: suppressed
assertions/keys stay in every table (append-only discipline) and are excluded
from the search surface by the filter (12_governance.filter). Nothing in the
canonical evidence layers is ever touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aml_store.model import (
    GOV_OP_ASSERT,
    GOV_OP_CONFLICT_OPEN,
    GOV_OP_CONFLICT_RESOLVE,
    GOV_OP_STALE_ARRIVAL,
    GOV_OP_SUPERSEDE,
    GOV_OP_SUPPRESS,
    GOV_OP_UNSUPPRESS,
    GovernanceAssertionRecord,
    GovernanceOpRecord,
)

from .factkey import has_update_marker

STATUS_CURRENT = "current"
STATUS_HISTORICAL = "historical"
STATUS_CONFLICT_OPEN = "conflict_open"

LIVE_STATUSES = (STATUS_CURRENT, STATUS_CONFLICT_OPEN)


@dataclass
class AssertionState:
    assertion_id: str
    fact_key: str
    value_hash: str
    fact_text: str
    source_atom_id: str
    source_unit_id: str
    ts_ms: int | None
    arrival_seq: int
    status: str = STATUS_HISTORICAL
    suppressed: bool = False
    superseded_by: str = ""  # assertion_id of the successor, when superseded


@dataclass
class FactKeyState:
    fact_key: str
    assertions: dict[str, AssertionState] = field(default_factory=dict)
    current_id: str | None = None
    conflict_open: bool = False
    suppressed: bool = False

    def current(self) -> AssertionState | None:
        if self.current_id is None:
            return None
        return self.assertions.get(self.current_id)

    def conflict_anchor(self) -> AssertionState:
        """Deterministic anchor for ordering against an open conflict: the
        max-ts member, ties broken by max arrival_seq."""
        return max(
            self.assertions.values(),
            key=lambda a: (a.ts_ms if a.ts_ms is not None else -1, a.arrival_seq),
        )


class GovernanceState:
    """Folded governance state for ONE user."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.keys: dict[str, FactKeyState] = {}
        self.by_atom: dict[str, AssertionState] = {}  # source_atom_id -> assertion

    # -- lookups ------------------------------------------------------------

    def fact_key_state(self, fact_key: str) -> FactKeyState | None:
        return self.keys.get(fact_key)

    def assertion_by_atom(self, source_atom_id: str) -> AssertionState | None:
        return self.by_atom.get(source_atom_id)

    def key_has_live_current(self, fact_key: str) -> bool:
        k = self.keys.get(fact_key)
        if k is None or k.suppressed:
            return False
        cur = k.current()
        return cur is not None and not cur.suppressed


def _op(
    op_seq: int,
    user_id: str,
    op_type: str,
    fact_key: str,
    assertion_id: str,
    prior_ref: str,
    detail: dict,
    created_at: str,
) -> GovernanceOpRecord:
    return GovernanceOpRecord(
        op_seq=op_seq,
        user_id=user_id,
        op_type=op_type,
        fact_key=fact_key,
        assertion_id=assertion_id,
        prior_ref=prior_ref,
        detail=detail,
        created_at=created_at,
    )


def plan_assert(
    state: GovernanceState, record: GovernanceAssertionRecord, op_seq: int, created_at: str
) -> list[GovernanceOpRecord]:
    """Compute the journal ops for asserting `record` against `state`.

    Pure function of (state, record): reads state, mutates nothing. Caller
    sequence per record (see GovernanceEngine.ingest_atomic): plan FIRST
    (state must not yet contain the new row), then register_assertion, then
    fold the returned ops before planning the next record. Deterministic
    transition table (documented in the EXP1 report):

      no key state            → ASSERT (becomes current)
      same value as anchor    → ASSERT (duplicate identity; anchor stays current)
      ts(anchor) < ts(new)    → ASSERT + SUPERSEDE (+ CONFLICT_RESOLVE if open)
      ts(anchor) > ts(new)    → ASSERT + STALE_ARRIVAL (new stays historical)
      no ordering evidence    → ASSERT + CONFLICT_OPEN (both open, current=None)
      update vocabulary, no
      contradicting ts        → ASSERT + SUPERSEDE (evidence=update_marker)
    """
    user = record.user_id
    ops: list[GovernanceOpRecord] = []
    provenance = {
        "source_atom_id": record.source_atom_id,
        "source_unit_id": record.source_unit_id,
        "ts_ms": record.ts_ms,
        "arrival_seq": record.arrival_seq,
        "value_hash": record.value_hash,
    }

    def _assert_op(became: str) -> GovernanceOpRecord:
        return _op(
            op_seq + len(ops), user, GOV_OP_ASSERT, record.fact_key, record.assertion_id,
            "", {**provenance, "became": became}, created_at,
        )

    k = state.keys.get(record.fact_key)
    if k is None or not k.assertions:
        ops.append(_assert_op(STATUS_CURRENT))
        return ops

    anchor = k.current() if k.current_id is not None else k.conflict_anchor()
    if anchor is None:  # defensive: every member suppressed -> fresh current
        ops.append(_assert_op(STATUS_CURRENT))
        return ops

    if anchor.value_hash == record.value_hash:
        # identity: the same fact value re-asserted never displaces the
        # current statement; the new row is recorded as a duplicate.
        ops.append(
            _op(
                op_seq + len(ops), user, GOV_OP_ASSERT, record.fact_key, record.assertion_id,
                anchor.assertion_id, {**provenance, "became": STATUS_HISTORICAL,
                                      "reason": "duplicate_value", "duplicate_of": anchor.assertion_id},
                created_at,
            )
        )
        return ops

    a_ts, anchor_ts = record.ts_ms, anchor.ts_ms
    ordered_newer = a_ts is not None and anchor_ts is not None and a_ts > anchor_ts
    ordered_older = a_ts is not None and anchor_ts is not None and a_ts < anchor_ts
    marker = has_update_marker(record.fact_text)

    if ordered_newer or (marker and not ordered_older):
        evidence = "ts_order" if ordered_newer else "update_marker"
        ops.append(_assert_op(STATUS_CURRENT))
        ops.append(
            _op(
                op_seq + len(ops), user, GOV_OP_SUPERSEDE, record.fact_key, record.assertion_id,
                anchor.assertion_id, {**provenance, "evidence": evidence,
                                      "superseded": anchor.assertion_id},
                created_at,
            )
        )
        if k.conflict_open:
            ops.append(
                _op(
                    op_seq + len(ops), user, GOV_OP_CONFLICT_RESOLVE, record.fact_key,
                    record.assertion_id, anchor.assertion_id,
                    {**provenance, "evidence": evidence, "resolved": sorted(k.assertions)},
                    created_at,
                )
            )
        return ops

    if ordered_older:
        ops.append(
            _op(
                op_seq + len(ops), user, GOV_OP_STALE_ARRIVAL, record.fact_key, record.assertion_id,
                anchor.assertion_id, {**provenance, "became": STATUS_HISTORICAL,
                                      "reason": "stale_late_arrival"}, created_at,
            )
        )
        return ops

    # no ordering evidence → unresolved conflict: both stay open, current=None
    ops.append(_assert_op(STATUS_CONFLICT_OPEN))
    ops.append(
        _op(
            op_seq + len(ops), user, GOV_OP_CONFLICT_OPEN, record.fact_key, record.assertion_id,
            anchor.assertion_id, {**provenance, "opened_with": anchor.assertion_id}, created_at,
        )
    )
    return ops


def plan_suppress(
    state: GovernanceState,
    user_id: str,
    fact_key: str,
    assertion_id: str,
    reason: str,
    op_seq: int,
    created_at: str,
    unsuppress: bool = False,
) -> list[GovernanceOpRecord]:
    """Key-level or assertion-level (un)suspend ops. Suppression never touches
    any canonical table — it only flips read-surface flags in the fold."""
    op_type = GOV_OP_UNSUPPRESS if unsuppress else GOV_OP_SUPPRESS
    return [
        GovernanceOpRecord(
            op_seq=op_seq,
            user_id=user_id,
            op_type=op_type,
            fact_key=fact_key,
            assertion_id=assertion_id,
            prior_ref="",
            detail={"reason": reason, "scope": "assertion" if assertion_id else "fact_key"},
            created_at=created_at,
        )
    ]


def apply_op(state: GovernanceState, op: GovernanceOpRecord) -> None:
    """The single transition function. Fold = apply_op over journal order."""
    if op.op_type == GOV_OP_ASSERT:
        # ASSERT carries the target via assertion_id; find it in any key bucket.
        a = None
        for k in state.keys.values():
            a = k.assertions.get(op.assertion_id)
            if a is not None:
                break
        if a is None:
            raise RuntimeError(f"governance fold: ASSERT for unknown assertion {op.assertion_id}")
        a.status = op.detail.get("became", STATUS_CURRENT)
        k = state.keys[a.fact_key]
        if a.status == STATUS_CURRENT:
            k.current_id = a.assertion_id
        return

    if op.op_type == GOV_OP_SUPERSEDE:
        k = state.keys[op.fact_key]
        old = k.assertions.get(op.prior_ref)
        new = k.assertions.get(op.assertion_id)
        if old is not None:
            old.status = STATUS_HISTORICAL
            old.superseded_by = new.assertion_id if new is not None else ""
        if new is not None:
            new.status = STATUS_CURRENT
        k.current_id = new.assertion_id if new is not None else None
        return

    if op.op_type == GOV_OP_STALE_ARRIVAL:
        k = state.keys[op.fact_key]
        a = k.assertions.get(op.assertion_id)
        if a is not None:
            a.status = STATUS_HISTORICAL
        return

    if op.op_type == GOV_OP_CONFLICT_OPEN:
        k = state.keys[op.fact_key]
        a = k.assertions.get(op.assertion_id)
        if a is not None:
            a.status = STATUS_CONFLICT_OPEN
        anchor = k.assertions.get(op.prior_ref)
        if anchor is not None:
            anchor.status = STATUS_CONFLICT_OPEN
        k.current_id = None
        k.conflict_open = True
        return

    if op.op_type == GOV_OP_CONFLICT_RESOLVE:
        k = state.keys[op.fact_key]
        k.conflict_open = False
        # every remaining open member is resolved by the new current: it
        # becomes historical (superseded), so exactly one current remains.
        for a in k.assertions.values():
            if a.status == STATUS_CONFLICT_OPEN and a.assertion_id != k.current_id:
                a.status = STATUS_HISTORICAL
                a.superseded_by = k.current_id or ""
        return

    if op.op_type in (GOV_OP_SUPPRESS, GOV_OP_UNSUPPRESS):
        suppress = op.op_type == GOV_OP_SUPPRESS
        k = state.keys.get(op.fact_key)
        if k is None:
            return
        if op.assertion_id:
            a = k.assertions.get(op.assertion_id)
            if a is not None:
                a.suppressed = suppress
        else:
            k.suppressed = suppress
            for a in k.assertions.values():
                a.suppressed = suppress
        return

    raise RuntimeError(f"governance fold: unknown op type {op.op_type!r}")


def register_assertion(state: GovernanceState, record: GovernanceAssertionRecord) -> AssertionState:
    """Insert a new assertion row into the folded state (pre-ASSERT placement
    so apply_op can position it). Mirrors one governance_assertions row."""
    k = state.keys.setdefault(record.fact_key, FactKeyState(fact_key=record.fact_key))
    a = AssertionState(
        assertion_id=record.assertion_id,
        fact_key=record.fact_key,
        value_hash=record.value_hash,
        fact_text=record.fact_text,
        source_atom_id=record.source_atom_id,
        source_unit_id=record.source_unit_id,
        ts_ms=record.ts_ms,
        arrival_seq=record.arrival_seq,
    )
    k.assertions[a.assertion_id] = a
    state.by_atom[record.source_atom_id] = a
    return a


def fold_ops(state: GovernanceState, ops: list[GovernanceOpRecord]) -> None:
    for op in ops:
        apply_op(state, op)
