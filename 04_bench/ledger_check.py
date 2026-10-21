#!/usr/bin/env python3
"""Ledger SERVER_CONFIG accounting (G34 / workorder H7).

Classifies every row's SERVER_CONFIG vs CONFIG relation — explicitly
distinguishing "key missing" (legacy rows 1–3, pre-R1.1 fingerprint scheme)
from "null value" and from a strict match. Never folds the missing key into a
None comparison via `.get()`.

Usage:
    python 04_bench/ledger_check.py [path/to/ledger.jsonl]

Exit 0 when every row is either a strict match or a documented legacy
missing-key row; exit 1 when any row has a null or mismatched SERVER_CONFIG
value (integrity violation — investigate, never auto-promote).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_bench"))

from benchlib.ledger import (  # noqa: E402
    REL_NO_KEY,
    REL_NULL_VALUE,
    REL_STRICT_MATCH,
    REL_VALUE_MISMATCH,
    audit_ledger_llm_provenance,
    audit_ledger_server_config,
    server_config_relation,
)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "06_results" / "ledger" / "ledger.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"ledger   : {path} ({len(records)} rows)")
    print(f"{'row':>4}  {'relation':22}  {'provenance':26}  RUN_ID / LABEL")
    prov = audit_ledger_llm_provenance(records)
    for i, rec in enumerate(records, start=1):
        rel = server_config_relation(rec)
        prel = next((k for k, rows in prov["non_ok_rows"].items() if i in rows), "provenance_ok")
        print(f"{i:>4}  {rel:22}  {prel:26}  {rec.get('RUN_ID', '?')} / {rec.get('LABEL', '')}")
    audit = audit_ledger_server_config(records)
    print("\nsummary (SERVER_CONFIG):")
    for rel, n in audit["counts"].items():
        print(f"  {rel:22} {n}")
    print(f"  non-strict rows by class: {audit['non_strict_rows'] or 'none'}")
    print("summary (G46 LLM provenance):")
    for rel, n in prov["counts"].items():
        print(f"  {rel:26} {n}")
    print(f"  non-ok rows by class: {prov['non_ok_rows'] or 'none'}")
    bad = audit["counts"][REL_NULL_VALUE] + audit["counts"][REL_VALUE_MISMATCH]
    bad += prov["counts"]["provenance_null_value"]
    bad += prov["counts"]["provenance_malformed"]
    bad += prov["counts"]["provenance_key_material"]
    print(
        "\nLEDGER_CHECK: "
        + (
            f"FAIL — SERVER_CONFIG null/mismatch: {audit['counts'][REL_NULL_VALUE] + audit['counts'][REL_VALUE_MISMATCH]}, "
            f"provenance violations: {prov['counts']['provenance_null_value'] + prov['counts']['provenance_malformed'] + prov['counts']['provenance_key_material']}"
            if bad
            else f"OK — {audit['counts'][REL_STRICT_MATCH]} strict + {audit['counts'][REL_NO_KEY]} documented legacy no-key rows; "
            f"{prov['counts']['provenance_ok']} provenance-ok + {prov['counts']['provenance_legacy_missing']} documented legacy provenance-missing rows"
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
