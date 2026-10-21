"""Memory unit model and add-payload types for the CompetitionMemoryStore.

Semantic provenance: raw-evidence-as-canonical, provenance-on-write and
user-scoped isolation follow the zhiji RC design (commit 5fe9b1d8), rewritten
clean-room for AML per ARCHITECTURE B3. No code was copied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _canonical_content(role: str, content: str) -> str:
    return "\x1f".join((role, content))


def content_sha256(role: str, content: str) -> str:
    return hashlib.sha256(_canonical_content(role, content).encode("utf-8")).hexdigest()


def dedupe_key(user_id: str, session_id: str, request_id: str, msg_index: int, role: str, content: str) -> str:
    return hashlib.sha256(
        "\x1f".join((user_id, session_id, request_id, str(msg_index), _canonical_content(role, content))).encode(
            "utf-8"
        )
    ).hexdigest()


GENERIC_CHAT_ROLES = frozenset(
    {"user", "assistant", "system", "tool", "function", "developer"}
)


def speaker_label(role: str | None) -> str | None:
    """Speaker label carried by a raw role, or None for generic chat roles.

    The official contract fixes role to "user"/"assistant" — a generic value is
    not a participant identity, so every surface (display prefix, derived fact
    text, index text) treats it as "no speaker" rather than emitting the role
    itself as a label."""
    if not role:
        return None
    label = role.strip()
    return None if label.lower() in GENERIC_CHAT_ROLES else label


def iso_z(ts_ms: int | None, fallback_now: datetime | None = None) -> str:
    """UTC ISO-8601 (second precision, 'Z' suffix) — fixed width, sortable."""
    if ts_ms is not None:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    else:
        dt = fallback_now if fallback_now is not None else datetime.now(tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class MemoryUnit:
    """One stored evidence unit. Raw evidence is canonical (INV1): this row is
    the source of truth; every derived structure (indexes, scores) is rebuildable
    and never written back (INV2)."""

    id: str
    user_id: str
    session_id: str
    request_id: str
    msg_index: int
    role: str
    content: str
    ts_ms: int | None
    created_at: str
    arrival_seq: int
    session_seq: int
    content_sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            "unit_id": self.id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "msg_index": self.msg_index,
            "role": self.role,
            "content_sha256": self.content_sha256,
            "ts_ms": self.ts_ms,
            "arrival_seq": self.arrival_seq,
            "session_seq": self.session_seq,
        }

    def provenance_json(self) -> str:
        return json.dumps(self.provenance(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class MessageIn:
    role: str
    content: str
    timestamp: int | None = None


@dataclass(frozen=True)
class AddPayload:
    request_id: str
    user_id: str
    session_id: str
    messages: list[MessageIn]


@dataclass(frozen=True)
class AddOutcome:
    request_id: str
    user_id: str
    session_id: str
    stored: list[MemoryUnit] = field(default_factory=list)
    skipped_duplicate: list[str] = field(default_factory=list)  # ids of pre-existing units


# ---------------------------------------------------------------------------
# Derived layer (V1 Atomic). Rows here are DERIVED, never canonical: the raw
# `memories` table stays the only source of truth (INV1), every derived row
# carries provenance back to its source raw unit, and nothing is ever written
# back into `memories` (INV2). The table lives in the same SQLite db but is a
# separate namespace exactly so the layering is structural, not conventional.


def atomic_unit_id(source_arrival_seq: int, fact_index: int) -> str:
    """Stable derived-doc id: `atom-<source arrival seq>-<fact index>`.

    The source seq is embedded in the id itself so any consumer (including the
    bench harness) can resolve provenance from the id alone, without a
    side channel. fact_index is 1-based within the source message.
    """
    return f"atom-{source_arrival_seq:06d}-{fact_index:02d}"


@dataclass(frozen=True)
class AtomicUnitRecord:
    """One derived atomic fact extracted from a single raw unit at Add time."""

    atom_id: str
    user_id: str
    source_unit_id: str
    source_content_sha256: str
    session_id: str
    msg_index: int
    fact_index: int
    fact: str
    extractor: str  # e.g. "rule-v1" / "llm-v1-fine@<model id>" (id recorded, never referenced)
    granularity: str
    created_at: str  # inherited from the source raw unit
    ts_ms: int | None  # inherited from the source raw unit

    def fact_sha256(self) -> str:
        return hashlib.sha256(self.fact.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Governance layer (EXP1, ZT-20260919-13). Rows here are DERIVED governance
# state, never canonical: the raw `memories` table stays the only source of
# truth (INV1), every governance row carries provenance back to its source
# atomic fact and through it to the raw unit, and both tables are INSERT-ONLY
# (no UPDATE/DELETE path exists — governance suppression is read-surface
# semantics recorded in an append-only journal, never data destruction).


def governance_assertion_id(source_atom_id: str) -> str:
    """Stable assertion identity: one atomic fact yields exactly one assertion,
    so the id is a namespace-renamed atom id (provenance is structural)."""
    return f"ga-{source_atom_id}"


@dataclass(frozen=True)
class GovernanceAssertionRecord:
    """One governance assertion derived from a single atomic fact."""

    assertion_id: str
    user_id: str
    fact_key: str  # hex sha256 of the normalized subject+relation skeleton
    value_hash: str  # hex sha256 of the normalized value remainder
    fact_text: str  # the speaker-bound fact text, verbatim from the atom
    source_atom_id: str
    source_unit_id: str
    created_at: str  # inherited from the source raw unit
    ts_ms: int | None  # inherited from the source raw unit
    arrival_seq: int  # inherited from the source raw unit


# Governance journal op types (the fold transitions, see 12_governance/state.py)
GOV_OP_ASSERT = "ASSERT"
GOV_OP_SUPERSEDE = "SUPERSEDE"
GOV_OP_STALE_ARRIVAL = "STALE_ARRIVAL"
GOV_OP_CONFLICT_OPEN = "CONFLICT_OPEN"
GOV_OP_CONFLICT_RESOLVE = "CONFLICT_RESOLVE"
GOV_OP_SUPPRESS = "SUPPRESS"
GOV_OP_UNSUPPRESS = "UNSUPPRESS"

GOV_OP_TYPES = (
    GOV_OP_ASSERT,
    GOV_OP_SUPERSEDE,
    GOV_OP_STALE_ARRIVAL,
    GOV_OP_CONFLICT_OPEN,
    GOV_OP_CONFLICT_RESOLVE,
    GOV_OP_SUPPRESS,
    GOV_OP_UNSUPPRESS,
)


@dataclass(frozen=True)
class GovernanceOpRecord:
    """One append-only governance journal entry (event-sourced state)."""

    op_seq: int  # per-user monotonic, 1-based
    user_id: str
    op_type: str  # one of GOV_OP_TYPES
    fact_key: str
    assertion_id: str  # "": key-scoped ops (SUPPRESS by key) carry no assertion
    prior_ref: str  # assertion/op this op supersedes or resolves (provenance link)
    detail: dict[str, Any]  # full provenance payload (json-encoded on write)
    created_at: str

    def detail_json(self) -> str:
        return json.dumps(self.detail, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
