#!/usr/bin/env python3
"""aml_atomic CLI — cache artifact freeze/thaw/fingerprint + provenance audit.

    python -m aml_atomic.cli fingerprint [--cache-dir DIR]
    python -m aml_atomic.cli freeze --out FILE [--cache-dir DIR]
    python -m aml_atomic.cli thaw FILE [--cache-dir DIR]
    python -m aml_atomic.cli audit --db DB          # G37 provenance check

`fingerprint` prints the EXTRACTION_CACHE_FP of the current cache content.
`freeze` writes the canonical JSONL artifact and prints its sha256.
`thaw` loads a frozen artifact into a cache db (idempotent; zero LLM calls).
`audit` verifies, for every derived row in a service db, that the provenance
points at an existing raw unit and that no raw unit content was mutated
(raw-canonical / no-writeback evidence for G37).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aml_store import CompetitionMemoryStore  # noqa: E402

import _bootstrap  # noqa: E402,F401

from .cache import ExtractionCache  # noqa: E402


def _cache(cache_dir: str) -> ExtractionCache:
    return ExtractionCache(Path(cache_dir) / "extraction_cache.sqlite3")


def main() -> int:
    parser = argparse.ArgumentParser(description="aml_atomic cache + provenance tooling")
    parser.add_argument("--cache-dir", default=str(REPO / "06_results" / "atomic_cache"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fingerprint")
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--out", required=True)
    for subp in (sub.choices["fingerprint"], p_freeze):
        subp.add_argument("--granularity", default=None, choices=("coarse", "fine"))
    p_thaw = sub.add_parser("thaw")
    p_thaw.add_argument("file")
    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--db", required=True)

    args = parser.parse_args()

    if args.cmd == "fingerprint":
        cache = _cache(args.cache_dir)
        print(cache.fingerprint(args.granularity))
        cache.close()
        return 0

    if args.cmd == "freeze":
        cache = _cache(args.cache_dir)
        fp = cache.export_jsonl(args.out, args.granularity)
        n = len(cache.rows(args.granularity))
        cache.close()
        print(f"frozen {n} entries -> {args.out}")
        print(f"EXTRACTION_CACHE_FP {fp}")
        return 0

    if args.cmd == "thaw":
        cache = _cache(args.cache_dir)
        added = cache.import_jsonl(args.file)
        fp = cache.fingerprint()
        cache.close()
        print(f"thawed {added} new entries from {args.file}")
        print(f"EXTRACTION_CACHE_FP {fp}")
        return 0

    if args.cmd == "audit":
        db = Path(args.db)
        if not db.exists():
            print(f"db not found: {args.db}")
            return 2
        store = CompetitionMemoryStore(args.db)
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        derived = conn.execute("SELECT COUNT(*) AS c FROM atomic_units").fetchone()["c"]
        markers = conn.execute("SELECT COUNT(*) AS c FROM atomic_units WHERE fact = ''").fetchone()["c"]
        orphans = conn.execute(
            "SELECT COUNT(*) AS c FROM atomic_units a"
            " LEFT JOIN memories m ON m.user_id = a.user_id AND m.id = a.source_unit_id"
            " WHERE m.id IS NULL"
        ).fetchone()["c"]
        tampered = conn.execute(
            "SELECT COUNT(*) AS c FROM atomic_units a"
            " JOIN memories m ON m.user_id = a.user_id AND m.id = a.source_unit_id"
            " WHERE m.content_sha256 <> a.source_content_sha256"
        ).fetchone()["c"]
        conn.close()
        store.close()
        print(f"derived rows         {derived} (markers {markers})")
        print(f"orphan provenance    {orphans}")
        print(f"source-mutated rows  {tampered}")
        ok = orphans == 0 and tampered == 0
        print("AUDIT", "OK" if ok else "VIOLATION")
        return 0 if ok else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
