"""Extraction cache (workorder A2 / G36) — the determinism anchor for Add-side
LLM extraction.

Cache key = sha256(prompt_version | model_id | granularity | raw content hash
| endpoint_identity). Endpoint identity (ADDENDUM-02 §3.1 / G45) is the
normalized hostname of the LLM endpoint — never a URL with credentials, never
an API key. Identical inputs on the same endpoint therefore ALWAYS hit the
same stored extraction, and the same content through a DIFFERENT endpoint
lands on a different key (relay products can never silently mix with official
ones), so:

  * two extraction passes over the same input produce bit-identical products
    (the second pass is a cache hit);
  * once the cache exists, bench reruns make ZERO LLM calls and reproduce
    scores bit-for-bit;
  * the cache content can be frozen to a canonical JSONL artifact whose
    sha256 (EXTRACTION_CACHE_FP) goes into the experiment ledger, letting the
    Judge verify "same extraction products" across machines;
  * every entry is self-describing about which endpoint produced it (v2
    schema column `endpoint_identity`).

Schema v2 (this round, minimal version bump per ADDENDUM-02 §3.1): legacy v1
entries (keys without endpoint identity) are never deleted — their old keys
simply cannot be hit by v2 lookups (different hash material), so they stay
inert audit history in whatever db holds them.

The cache never enters git (size + license hygiene); only its fingerprint
does. Storage location is config infra (`extraction_cache.dir`), excluded from
the behavior fingerprint like db paths — the *content* is pinned via the
fingerprint instead.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 2
_SCHEMA = """
CREATE TABLE IF NOT EXISTS extraction_cache (
    key              TEXT PRIMARY KEY,
    facts_json       TEXT NOT NULL,
    model_id         TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    granularity      TEXT NOT NULL,
    endpoint_identity TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL
);
"""


def cache_key(
    prompt_version: str,
    model_id: str,
    granularity: str,
    source_content_sha256: str,
    endpoint_identity: str,
) -> str:
    """Workorder A2 + ADDENDUM-02 §3.1 (G45): sha256(raw unit content + model
    id + prompt version + granularity + endpoint identity). The raw content
    enters via its sha256 (role+content); the endpoint enters as the
    normalized hostname only — never keys, never URLs with parameters."""
    material = "\x1f".join(
        (prompt_version, model_id, granularity, source_content_sha256, endpoint_identity)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def canonical_export_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Byte-canonical freeze artifact: rows sorted by key, canonical JSON per
    line. The fingerprint is defined over exactly these bytes."""
    lines = []
    for row in sorted(rows, key=lambda r: r["key"]):
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def cache_fingerprint(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_export_bytes(rows)).hexdigest()


def _has_endpoint_column(conn: sqlite3.Connection) -> bool:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(extraction_cache)")}
    return "endpoint_identity" in cols


def fingerprint_of_db(db_path: str | Path, granularity: str | None = None) -> str:
    """Fingerprint an existing cache db WITHOUT creating it (read-only). The
    bench runner uses this so the client side never materializes cache files.
    `granularity` restricts the fingerprint to that granularity's entries —
    the per-run semantics used by the ledger from the endpoint-identity round
    on (a coarse run must not be labeled with a fine+coarse mixed fingerprint).
    Legacy dbs without the endpoint_identity column keep their exact
    pre-v2 fingerprint bytes (the field simply is not part of their rows)."""
    path = Path(db_path)
    if not path.exists():
        return "missing"
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        with_endpoint = _has_endpoint_column(conn)
        select = (
            "SELECT key, facts_json, model_id, prompt_version, granularity, created_at"
            + (", endpoint_identity" if with_endpoint else "")
            + " FROM extraction_cache"
        )
        params: tuple = ()
        if granularity is not None:
            select += " WHERE granularity = ?"
            params = (granularity,)
        rows = []
        for row in conn.execute(select, params):
            entry = {
                "key": row["key"],
                "facts": json.loads(row["facts_json"]),
                "model_id": row["model_id"],
                "prompt_version": row["prompt_version"],
                "granularity": row["granularity"],
                "created_at": row["created_at"],
            }
            if with_endpoint:
                entry["endpoint_identity"] = row["endpoint_identity"]
            rows.append(entry)
    except sqlite3.OperationalError:
        return "missing"  # no table → empty/foreign db
    finally:
        conn.close()
    return cache_fingerprint(rows)


class ExtractionCache:
    """SQLite-backed content-hash cache. Thread-safe (single shared
    connection guarded by a lock — the pipeline fans extraction out over a
    small thread pool)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            if not _has_endpoint_column(self._conn):
                # Legacy v1 db: keep every entry (ADDENDUM-02 §3.1 — never
                # delete), add the column with '' = legacy/unattributed. Old
                # keys are v1-scheme and can never be hit by v2 lookups.
                self._conn.execute(
                    "ALTER TABLE extraction_cache ADD COLUMN endpoint_identity TEXT NOT NULL DEFAULT ''"
                )

    def get(self, key: str) -> list[str] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT facts_json FROM extraction_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            facts = json.loads(row["facts_json"])
            if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
                raise ValueError("not a list of strings")
        except (ValueError, TypeError):
            with self._lock, self._conn:  # corrupt entry: self-heal, behave as a miss
                self._conn.execute("DELETE FROM extraction_cache WHERE key = ?", (key,))
            return None
        return facts

    def put(
        self,
        key: str,
        facts: list[str],
        *,
        model_id: str,
        prompt_version: str,
        granularity: str,
        endpoint_identity: str,
        created_at: str,
    ) -> bool:
        """Store one entry; returns True when a new row was added."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO extraction_cache"
                " (key, facts_json, model_id, prompt_version, granularity, endpoint_identity, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    key,
                    json.dumps(facts, ensure_ascii=False),
                    model_id,
                    prompt_version,
                    granularity,
                    endpoint_identity,
                    created_at,
                ),
            )
            return cur.rowcount > 0

    def rows(self, granularity: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            sql = (
                "SELECT key, facts_json, model_id, prompt_version, granularity,"
                " endpoint_identity, created_at FROM extraction_cache"
            )
            params: tuple = ()
            if granularity is not None:
                sql += " WHERE granularity = ?"
                params = (granularity,)
            out = []
            for row in self._conn.execute(sql, params).fetchall():
                out.append(
                    {
                        "key": row["key"],
                        "facts": json.loads(row["facts_json"]),
                        "model_id": row["model_id"],
                        "prompt_version": row["prompt_version"],
                        "granularity": row["granularity"],
                        "endpoint_identity": row["endpoint_identity"],
                        "created_at": row["created_at"],
                    }
                )
            return out

    def fingerprint(self, granularity: str | None = None) -> str:
        return cache_fingerprint(self.rows(granularity))

    def export_jsonl(self, path: str | Path, granularity: str | None = None) -> str:
        """Freeze: write the canonical artifact; returns its fingerprint."""
        payload = canonical_export_bytes(self.rows(granularity))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def import_jsonl(self, path: str | Path) -> int:
        """Thaw: load a frozen artifact into a (typically fresh) cache db.
        Returns the number of entries actually added (OR IGNORE keeps existing
        entries intact, so thaw is idempotent). Rows without an
        `endpoint_identity` field (legacy v1 artifacts) import with '' —
        thaw NEVER invents endpoint attribution; use
        `migrate-endpoint` to attribute and re-key legacy products."""
        added = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if self.put(
                row["key"],
                row["facts"],
                model_id=row["model_id"],
                prompt_version=row["prompt_version"],
                granularity=row["granularity"],
                endpoint_identity=row.get("endpoint_identity", ""),
                created_at=row["created_at"],
            ):
                added += 1
        return added

    def close(self) -> None:
        with self._lock:
            self._conn.close()
