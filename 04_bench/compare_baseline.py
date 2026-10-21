#!/usr/bin/env python3
"""Compare the newest ledger record against the R1 Mac baseline (B0 gate).

Baseline = ledger.jsonl line 1 (RUN_ID local-20260905-131741-229622, label
local-baseline, commit 224ce55 lineage). Compares the 13 numeric score fields,
the 4 honestly-null AML dimensions, and the run counters. Latency fields are
reported but excluded from the verdict (wall-clock is not reproducible).
Exit 0 = all compared fields match; exit 1 = any mismatch (REFORGE-level
signal per workorder §3.1 — stop the matrix and report).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCORE_FIELDS = [
    "OVERALL",
    "FACT",
    "TEMPORAL",
    "MULTI_HOP",
    "EM_OVERALL",
    "MEMORY_ONLY_OVERALL",
    "EVIDENCE_RECALL_AT_10",
    "EVIDENCE_RECALL_AT_100",
    "LOCOMO_CAT1_F1",
    "LOCOMO_CAT2_F1",
    "LOCOMO_CAT3_F1",
    "LOCOMO_CAT4_F1",
    "LOCOMO_CAT5_F1",
]
NULL_FIELDS = ["GOVERNANCE", "PERSONALIZATION", "RULE", "SAFETY"]
COUNTER_FIELDS = [
    "N_CONVS",
    "N_QUESTIONS",
    "N_ADDS",
    "N_SEARCHES",
    "MODEL_CALLS",
    "TOKENS",
    "COST",
    "FAILURE_CASES",
]
REPORT_ONLY_FIELDS = ["LATENCY_P50", "LATENCY_P95", "ADD_LATENCY_P50", "ADD_LATENCY_P95", "CONFIG", "SERVER_CONFIG"]


def _same(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def main() -> int:
    ledger = Path(sys.argv[1] if len(sys.argv) > 1 else "06_results/ledger/ledger.jsonl")
    lines = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    baseline = lines[0]
    candidate = lines[-1]
    print(f"baseline : {baseline.get('RUN_ID')} ({baseline.get('LABEL')}, CONFIG {str(baseline.get('CONFIG'))[:12]}…)")
    print(f"candidate: {candidate.get('RUN_ID')} ({candidate.get('LABEL')}, CONFIG {str(candidate.get('CONFIG'))[:12]}…)")
    print(f"candidate COMMIT {candidate.get('COMMIT')}\n")

    mismatch: list[str] = []
    for field in SCORE_FIELDS:
        bv, cv = baseline.get(field), candidate.get(field)
        ok = bv is not None and cv is not None and _same(bv, cv)
        print(f"{field:26} base={bv!r:24} cand={cv!r:24} {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            mismatch.append(field)
    for field in NULL_FIELDS:
        ok = baseline.get(field) is None and candidate.get(field) is None
        print(f"{field:26} base={baseline.get(field)} cand={candidate.get(field)} {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            mismatch.append(field)
    for field in COUNTER_FIELDS:
        ok = _same(baseline.get(field), candidate.get(field))
        print(f"{field:26} base={baseline.get(field)!r:8} cand={candidate.get(field)!r:8} {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            mismatch.append(field)
    print()
    for field in REPORT_ONLY_FIELDS:
        print(f"{field:26} base={baseline.get(field)} cand={candidate.get(field)}  (report-only, not gated)")
    print(f"\nfingerprints equal on candidate: {candidate.get('CONFIG') == candidate.get('SERVER_CONFIG')}")
    verdict = "ALL_MATCH" if not mismatch else "MISMATCH:" + ",".join(mismatch)
    print(f"B0_RESULT: {verdict}")
    return 0 if not mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
