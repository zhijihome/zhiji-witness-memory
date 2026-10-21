#!/usr/bin/env python3
"""Top-10 family turnover between two runs (WORKORDER_AML_V2_R1 §3.3).

FROZEN DEFINITION (set before any V2 arm ran; must not change afterwards):
the top-10 gold-evidence (dia) SET of each question, Jaccard distance
1 - |A∩B| / |A∪B| per question, averaged over all questions where BOTH runs
returned a non-empty top-10. This is `turnover_mean` below.

Secondary diagnostic (same tool, clearly labeled): `turnover_pooled` pools
all intersections/unions first (Σ|A∩B| / Σ|A∪B|). The Planner's reference
value 0.648 for a1-kr vs a0 corresponds to this pooled variant; under the
frozen per-question mean the same pair measures 0.629787 (both values are
reported so the historical reference stays reconcilable — the frozen
definition is authoritative for V2 comparisons).

Usage:
    .venv/bin/python 04_bench/top10_turnover.py RUN_ID BASELINE_RUN_ID [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _top10_dia_sets(results_root: Path, run_id: str) -> list[set[str]]:
    path = results_root / "runs" / run_id / "questions.jsonl"
    if not path.exists():
        raise SystemExit(f"detail file not found: {path}")
    out: list[set[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        out.append({t["dia"] for t in q.get("top_returned", []) if t.get("dia")})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="top-10 dia-set turnover (frozen definition)")
    parser.add_argument("run_id")
    parser.add_argument("baseline_run_id")
    parser.add_argument("--results-root", default=str(REPO / "06_results"))
    parser.add_argument("--json", action="store_true", help="machine-readable single-line JSON output")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    run_sets = _top10_dia_sets(results_root, args.run_id)
    base_sets = _top10_dia_sets(results_root, args.baseline_run_id)
    if len(run_sets) != len(base_sets):
        raise SystemExit(
            f"question count mismatch: {args.run_id} has {len(run_sets)}, "
            f"{args.baseline_run_id} has {len(base_sets)} — runs are not comparable"
        )

    distances: list[float] = []
    inter_sum = 0
    union_sum = 0
    for a, b in zip(run_sets, base_sets, strict=True):
        if not a or not b:
            continue  # frozen eligibility: both top-10 non-empty
        distances.append(1.0 - len(a & b) / len(a | b))
        inter_sum += len(a & b)
        union_sum += len(a | b)

    turnover_mean = sum(distances) / len(distances) if distances else None
    turnover_pooled = (1.0 - inter_sum / union_sum) if union_sum else None
    result = {
        "run_id": args.run_id,
        "baseline_run_id": args.baseline_run_id,
        "definition": "per-question top-10 dia-set Jaccard distance, mean over questions with both top-10 non-empty (WORKORDER_AML_V2_R1 §3.3, frozen)",
        "questions_total": len(run_sets),
        "questions_eligible": len(distances),
        "turnover_mean": round(turnover_mean, 6) if turnover_mean is not None else None,
        "turnover_pooled_reference_variant": round(turnover_pooled, 6) if turnover_pooled is not None else None,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(
            f"turnover {args.run_id} vs {args.baseline_run_id}: "
            f"mean={result['turnover_mean']} pooled={result['turnover_pooled_reference_variant']} "
            f"(eligible {len(distances)}/{len(run_sets)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
