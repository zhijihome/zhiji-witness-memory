"""CompetitionMemoryStore — the V0 evidence store (architecture layer 01).

Guarantees implemented here:
  * raw evidence is canonical (INV1): units are stored verbatim with provenance;
  * user_id is the only isolation boundary (INV3): every read path is scoped by
    an exact user_id match; no cross-user structure ever exists;
  * synchronous persistence (INV5): `add` returns only after the SQLite
    transaction has committed, so a subsequent search must see the data.

Duplicate-add semantics (G3), documented once and enforced here:
  * dedupe key = sha256(user_id | session_id | request_id | msg_index | role | content);
  * re-delivering an identical message (e.g. a platform retry of the same add
    request) is an idempotent no-op — the unit keeps its original id and the
    response is still success;
  * a *different* message under an already-used request_id is appended as a new
    unit (adds never mutate or overwrite stored units).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any
from unicodedata import normalize as unicodedata_normalize

from .model import (
    AddOutcome,
    AddPayload,
    AtomicUnitRecord,
    GovernanceAssertionRecord,
    GovernanceOpRecord,
    MemoryUnit,
    content_sha256,
    dedupe_key,
    iso_z,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id             TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    request_id     TEXT NOT NULL,
    msg_index      INTEGER NOT NULL,
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    ts_ms          INTEGER,
    created_at     TEXT NOT NULL,
    arrival_seq    INTEGER NOT NULL,
    session_seq    INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    dedupe_key     TEXT NOT NULL,
    provenance     TEXT NOT NULL,
    PRIMARY KEY (user_id, id),
    UNIQUE (user_id, dedupe_key)
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id, arrival_seq);
CREATE INDEX IF NOT EXISTS idx_memories_user_session ON memories (user_id, session_id, session_seq);
CREATE TABLE IF NOT EXISTS user_counters (
    user_id  TEXT PRIMARY KEY,
    next_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS atomic_units (
    atom_id               TEXT NOT NULL,
    user_id               TEXT NOT NULL,
    source_unit_id        TEXT NOT NULL,
    source_content_sha256 TEXT NOT NULL,
    session_id            TEXT NOT NULL,
    msg_index             INTEGER NOT NULL,
    fact_index            INTEGER NOT NULL,
    fact                  TEXT NOT NULL,
    fact_sha256           TEXT NOT NULL,
    extractor             TEXT NOT NULL,
    granularity           TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    ts_ms                 INTEGER,
    PRIMARY KEY (user_id, atom_id)
);
CREATE INDEX IF NOT EXISTS idx_atomic_user_source ON atomic_units (user_id, source_unit_id);
CREATE TABLE IF NOT EXISTS governance_assertions (
    assertion_id    TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    fact_key        TEXT NOT NULL,
    value_hash      TEXT NOT NULL,
    fact_text       TEXT NOT NULL,
    source_atom_id  TEXT NOT NULL,
    source_unit_id  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    ts_ms           INTEGER,
    arrival_seq     INTEGER NOT NULL,
    PRIMARY KEY (user_id, assertion_id)
);
CREATE INDEX IF NOT EXISTS idx_gov_assert_user_key ON governance_assertions (user_id, fact_key);
CREATE TABLE IF NOT EXISTS governance_journal (
    op_seq       INTEGER NOT NULL,
    user_id      TEXT NOT NULL,
    op_type      TEXT NOT NULL,
    fact_key     TEXT NOT NULL,
    assertion_id TEXT NOT NULL,
    prior_ref    TEXT,
    detail       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (user_id, op_seq)
);
CREATE INDEX IF NOT EXISTS idx_gov_journal_user_key ON governance_journal (user_id, fact_key);
"""

_COLS = (
    "id, user_id, session_id, request_id, msg_index, role, content, ts_ms, created_at, "
    "arrival_seq, session_seq, content_sha256"
)


def normalize_text(text: str) -> str:
    """NFKC + casefold + whitespace collapse — shared exact-match normalization."""
    return " ".join(unicodedata_normalize("NFKC", text).casefold().split())


class UserIndex:
    """Arrival-ordered, append-only view of one user's units (the isolation scope)."""

    __slots__ = ("user_id", "units", "by_id", "sessions", "norm_contents", "version")

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.units: list[MemoryUnit] = []
        self.by_id: dict[str, int] = {}
        self.sessions: dict[str, list[int]] = {}
        self.norm_contents: list[str] = []
        self.version = 0

    def append(self, unit: MemoryUnit) -> None:
        if unit.user_id != self.user_id:
            raise ValueError("cross-user append blocked")  # INV3 belt-and-braces
        if unit.id in self.by_id:  # idempotent: store cache and SearchIndex share this object
            return
        idx = len(self.units)
        self.units.append(unit)
        self.by_id[unit.id] = idx
        self.sessions.setdefault(unit.session_id, []).append(idx)
        self.norm_contents.append(normalize_text(unit.content))
        self.version += 1

    def neighbors(self, idx: int, window: int) -> list[tuple[int, int]]:
        """(unit_idx, distance) of session-adjacent units within `window`, both sides."""
        unit = self.units[idx]
        order = self.sessions[unit.session_id]
        try:
            pos = order.index(idx)
        except ValueError:
            return []
        out: list[tuple[int, int]] = []
        for d in range(1, window + 1):
            for pos_n in (pos - d, pos + d):
                if 0 <= pos_n < len(order):
                    out.append((order[pos_n], d))
        return out


def _row_to_unit(row: sqlite3.Row) -> MemoryUnit:
    return MemoryUnit(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        request_id=row["request_id"],
        msg_index=row["msg_index"],
        role=row["role"],
        content=row["content"],
        ts_ms=row["ts_ms"],
        created_at=row["created_at"],
        arrival_seq=row["arrival_seq"],
        session_seq=row["session_seq"],
        content_sha256=row["content_sha256"],
    )


class CompetitionMemoryStore:
    """SQLite-backed store with per-user in-memory index views.

    V0 concurrency model: a single reentrant lock serializes writes and index
    reads; the service must run with one uvicorn worker (README).
    """

    def __init__(self, db_path: str | Path, journal_mode: str = "WAL", synchronous: str = "FULL") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute(f"PRAGMA journal_mode={journal_mode}")
        except sqlite3.OperationalError:
            self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute(f"PRAGMA synchronous={synchronous}")
        with self._conn:
            self._conn.executescript(_SCHEMA)
        self.journal_mode: str = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self._user_index_cache: OrderedDict[str, UserIndex] = OrderedDict()
        self._index_cache_limit = 64

    # -- write path ---------------------------------------------------------

    def add(self, payload: AddPayload) -> AddOutcome:
        """Persist an add request; returns only after the commit is durable (INV5)."""
        with self._lock:
            now_iso = iso_z(None)
            stored: list[MemoryUnit] = []
            skipped: list[str] = []
            with self._conn:
                counter_row = self._conn.execute(
                    "SELECT next_seq FROM user_counters WHERE user_id = ?", (payload.user_id,)
                ).fetchone()
                next_seq = counter_row["next_seq"] if counter_row else 1
                for msg_index, msg in enumerate(payload.messages):
                    key = dedupe_key(
                        payload.user_id, payload.session_id, payload.request_id, msg_index, msg.role, msg.content
                    )
                    existing = self._conn.execute(
                        "SELECT id FROM memories WHERE user_id = ? AND dedupe_key = ?",
                        (payload.user_id, key),
                    ).fetchone()
                    if existing is not None:
                        skipped.append(existing["id"])
                        continue
                    sess_max = self._conn.execute(
                        "SELECT COALESCE(MAX(session_seq), 0) AS m FROM memories "
                        "WHERE user_id = ? AND session_id = ?",
                        (payload.user_id, payload.session_id),
                    ).fetchone()["m"]
                    unit = MemoryUnit(
                        id=f"mem-{next_seq:06d}",
                        user_id=payload.user_id,
                        session_id=payload.session_id,
                        request_id=payload.request_id,
                        msg_index=msg_index,
                        role=msg.role,
                        content=msg.content,
                        ts_ms=msg.timestamp,
                        created_at=iso_z(msg.timestamp) if msg.timestamp is not None else now_iso,
                        arrival_seq=next_seq,
                        session_seq=sess_max + 1,
                        content_sha256=content_sha256(msg.role, msg.content),
                    )
                    self._conn.execute(
                        "INSERT INTO memories (id, user_id, session_id, request_id, msg_index, role, content,"
                        " ts_ms, created_at, arrival_seq, session_seq, content_sha256, dedupe_key, provenance)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            unit.id,
                            unit.user_id,
                            unit.session_id,
                            unit.request_id,
                            unit.msg_index,
                            unit.role,
                            unit.content,
                            unit.ts_ms,
                            unit.created_at,
                            unit.arrival_seq,
                            unit.session_seq,
                            unit.content_sha256,
                            key,
                            unit.provenance_json(),
                        ),
                    )
                    stored.append(unit)
                    next_seq += 1
                self._conn.execute(
                    "INSERT INTO user_counters (user_id, next_seq) VALUES (?, ?)"
                    " ON CONFLICT(user_id) DO UPDATE SET next_seq = excluded.next_seq",
                    (payload.user_id, next_seq),
                )
            # committed — now refresh the cached in-memory view for this user
            cached = self._user_index_cache.get(payload.user_id)
            if cached is not None:
                for unit in stored:
                    cached.append(unit)
            return AddOutcome(
                request_id=payload.request_id,
                user_id=payload.user_id,
                session_id=payload.session_id,
                stored=stored,
                skipped_duplicate=skipped,
            )

    # -- derived layer (V1 Atomic) -------------------------------------------
    # The `memories` table is never touched here: derived rows live in their
    # own table, always carry provenance to their source raw unit, and are
    # insert-only (no update/delete path exists — G37 no-writeback).

    def insert_atomic_units(self, records: list[AtomicUnitRecord]) -> int:
        """Insert derived facts; idempotent on (user_id, atom_id). Returns the
        number of newly stored rows."""
        if not records:
            return 0
        with self._lock, self._conn:
            inserted = 0
            for r in records:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO atomic_units (atom_id, user_id, source_unit_id,"
                    " source_content_sha256, session_id, msg_index, fact_index, fact, fact_sha256,"
                    " extractor, granularity, created_at, ts_ms)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        r.atom_id,
                        r.user_id,
                        r.source_unit_id,
                        r.source_content_sha256,
                        r.session_id,
                        r.msg_index,
                        r.fact_index,
                        r.fact,
                        r.fact_sha256(),
                        r.extractor,
                        r.granularity,
                        r.created_at,
                        r.ts_ms,
                    ),
                )
                inserted += cur.rowcount if cur.rowcount > 0 else 0
            return inserted

    def _atomic_row_to_record(self, row: sqlite3.Row) -> AtomicUnitRecord:
        return AtomicUnitRecord(
            atom_id=row["atom_id"],
            user_id=row["user_id"],
            source_unit_id=row["source_unit_id"],
            source_content_sha256=row["source_content_sha256"],
            session_id=row["session_id"],
            msg_index=row["msg_index"],
            fact_index=row["fact_index"],
            fact=row["fact"],
            extractor=row["extractor"],
            granularity=row["granularity"],
            created_at=row["created_at"],
            ts_ms=row["ts_ms"],
        )

    _ATOMIC_COLS = (
        "atom_id, user_id, source_unit_id, source_content_sha256, session_id, msg_index,"
        " fact_index, fact, fact_sha256, extractor, granularity, created_at, ts_ms"
    )

    def atomic_units(self, user_id: str) -> list[AtomicUnitRecord]:
        """All derived facts for a user, in (source arrival, fact index) order."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._ATOMIC_COLS} FROM atomic_units WHERE user_id = ?"
                " ORDER BY atom_id",
                (user_id,),
            ).fetchall()
            return [self._atomic_row_to_record(row) for row in rows]

    def atomic_source_ids(self, user_id: str) -> set[str]:
        """Ids of raw units that already have at least one derived fact."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT source_unit_id FROM atomic_units WHERE user_id = ?", (user_id,)
            ).fetchall()
            return {row["source_unit_id"] for row in rows}

    def atomic_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) AS c FROM atomic_units").fetchone()["c"]

    # -- governance layer (EXP1, ZT-20260919-13) ------------------------------
    # Same derived-layer discipline as `atomic_units` (G37): these methods only
    # READ raw/atomic content and only WRITE the two governance tables. Both
    # are INSERT-ONLY — there is no UPDATE or DELETE path anywhere in this
    # class; governance state changes are new journal rows folded at read time.

    def insert_governance_assertions(self, records: list[GovernanceAssertionRecord]) -> int:
        """Insert assertion rows; idempotent on (user_id, assertion_id). Returns
        the number of newly stored rows."""
        if not records:
            return 0
        with self._lock, self._conn:
            inserted = 0
            for r in records:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO governance_assertions (assertion_id, user_id, fact_key,"
                    " value_hash, fact_text, source_atom_id, source_unit_id, created_at, ts_ms, arrival_seq)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        r.assertion_id,
                        r.user_id,
                        r.fact_key,
                        r.value_hash,
                        r.fact_text,
                        r.source_atom_id,
                        r.source_unit_id,
                        r.created_at,
                        r.ts_ms,
                        r.arrival_seq,
                    ),
                )
                inserted += cur.rowcount if cur.rowcount > 0 else 0
            return inserted

    def governance_assertion_ids(self, user_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT assertion_id FROM governance_assertions WHERE user_id = ?", (user_id,)
            ).fetchall()
            return {row["assertion_id"] for row in rows}

    def governance_assertions(self, user_id: str) -> list[GovernanceAssertionRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT assertion_id, user_id, fact_key, value_hash, fact_text, source_atom_id,"
                " source_unit_id, created_at, ts_ms, arrival_seq FROM governance_assertions"
                " WHERE user_id = ? ORDER BY arrival_seq, assertion_id",
                (user_id,),
            ).fetchall()
            return [
                GovernanceAssertionRecord(
                    assertion_id=row["assertion_id"],
                    user_id=row["user_id"],
                    fact_key=row["fact_key"],
                    value_hash=row["value_hash"],
                    fact_text=row["fact_text"],
                    source_atom_id=row["source_atom_id"],
                    source_unit_id=row["source_unit_id"],
                    created_at=row["created_at"],
                    ts_ms=row["ts_ms"],
                    arrival_seq=row["arrival_seq"],
                )
                for row in rows
            ]

    def append_governance_ops(self, ops: list[GovernanceOpRecord]) -> int:
        """Append journal rows under the store lock. op_seq is caller-assigned
        and per-user monotonic; INSERT (not OR IGNORE) so a sequence collision
        is a loud error, never a silently dropped governance event."""
        if not ops:
            return 0
        with self._lock, self._conn:
            for op in ops:
                self._conn.execute(
                    "INSERT INTO governance_journal (op_seq, user_id, op_type, fact_key,"
                    " assertion_id, prior_ref, detail, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        op.op_seq,
                        op.user_id,
                        op.op_type,
                        op.fact_key,
                        op.assertion_id,
                        op.prior_ref,
                        op.detail_json(),
                        op.created_at,
                    ),
                )
            return len(ops)

    def governance_ops(self, user_id: str) -> list[GovernanceOpRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT op_seq, user_id, op_type, fact_key, assertion_id, prior_ref, detail, created_at"
                " FROM governance_journal WHERE user_id = ? ORDER BY op_seq",
                (user_id,),
            ).fetchall()
            return [
                GovernanceOpRecord(
                    op_seq=row["op_seq"],
                    user_id=row["user_id"],
                    op_type=row["op_type"],
                    fact_key=row["fact_key"],
                    assertion_id=row["assertion_id"],
                    prior_ref=row["prior_ref"] or "",
                    detail=json.loads(row["detail"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def governance_next_op_seq(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(op_seq), 0) AS m FROM governance_journal WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return int(row["m"]) + 1

    def governance_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "governance_assertions": self._conn.execute(
                    "SELECT COUNT(*) AS c FROM governance_assertions"
                ).fetchone()["c"],
                "governance_ops": self._conn.execute(
                    "SELECT COUNT(*) AS c FROM governance_journal"
                ).fetchone()["c"],
            }

    # -- read path ----------------------------------------------------------

    def get_user_index(self, user_id: str) -> UserIndex:
        with self._lock:
            cached = self._user_index_cache.get(user_id)
            if cached is not None:
                self._user_index_cache.move_to_end(user_id)
                return cached
            index = UserIndex(user_id)
            for row in self._conn.execute(
                f"SELECT {_COLS} FROM memories WHERE user_id = ? ORDER BY arrival_seq", (user_id,)
            ).fetchall():
                index.append(_row_to_unit(row))
            self._user_index_cache[user_id] = index
            self._user_index_cache.move_to_end(user_id)
            while len(self._user_index_cache) > self._index_cache_limit:
                self._user_index_cache.popitem(last=False)
            return index

    def unit(self, user_id: str, unit_id: str) -> MemoryUnit | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLS} FROM memories WHERE user_id = ? AND id = ?", (user_id, unit_id)
            ).fetchone()
            return _row_to_unit(row) if row else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            users = self._conn.execute("SELECT COUNT(DISTINCT user_id) AS c FROM memories").fetchone()["c"]
            units = self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
            return {
                "users": users,
                "units": units,
                "atomic_units": self.atomic_count(),
                "journal_mode": self.journal_mode,
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
