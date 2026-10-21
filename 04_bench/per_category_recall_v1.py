#!/usr/bin/env python3
"""Per-category evidence recall for V1 arms (workorder §2 requirement).

The ledger carries per-category F1 but only OVERALL evidence recall; the
workorder asks for cat1~cat5 recall breakdown. This recomputes it from each
arm's questions.jsonl detail (which stores per-question recall + category).

Usage: .venv/bin/python 04_bench/per_category_recall_v1.py [RUN_ID ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def cat_recall(run_id: str, results_root: Path = REPO / "06_results") -> dict | None:
    path = results_root / "runs" / run_id / "questions.jsonl"
    if not path.exists():
        return None
    buckets: dict[int, list[tuple[float, float]]] = {}
    for line in path.read_text().splitlines():
        q = json.loads(line)
        buckets.setdefault(q["category"], []).append((q["evidence_recall_at_10"], q["evidence_recall_at_100"]))
    out = {}
    for cat in sorted(buckets):
        vals = buckets[cat]
        out[cat] = {
            "n": len(vals),
            "r10": round(sum(v[0] for v in vals) / len(vals), 6),
            "r100": round(sum(v[1] for v in vals) / len(vals), 6),
        }
    return out


def main() -> int:
    ledger = [json.loads(line) for line in (REPO / "06_results" / "ledger" / "ledger.jsonl").read_text().splitlines()]
    run_ids = sys.argv[1:]
    if not run_ids:
        run_ids = [r["RUN_ID"] for r in ledger if r.get("LABEL", "").startswith("v1-")]
    for run_id in run_ids:
        rec = next((r for r in ledger if r["RUN_ID"] == run_id), None)
        label = rec["LABEL"] if rec else "?"
        stats = cat_recall(run_id)
        if stats is None:
            print(f"{run_id} ({label}): no detail file")
            continue
        cells = " | ".join(
            f"cat{c}: @{10} {s['r10']:.3f} @{100} {s['r100']:.3f} (n={s['n']})" for c, s in stats.items()
        )
        print(f"{run_id} ({label})")
        print(f"  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
