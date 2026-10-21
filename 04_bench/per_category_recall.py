#!/usr/bin/env python3
"""Per-category recall breakdown from a run's questions.jsonl (report aid)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CAT_NAMES = {1: "cat1 single-hop", 2: "cat2 multi-hop", 3: "cat3 temporal", 4: "cat4 open-domain", 5: "cat5 adversarial"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", required=True)
    args = parser.parse_args()
    buckets: dict[int, list[dict]] = defaultdict(list)
    for line in Path(args.details).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            buckets[rec["category"]].append(rec)
    print("| cat | n | recall@10 | recall@100 |")
    print("|---|---|---|---|")
    for cat in sorted(buckets):
        rows = buckets[cat]
        n = len(rows)
        r10 = sum(r["evidence_recall_at_10"] for r in rows) / n
        r100 = sum(r["evidence_recall_at_100"] for r in rows) / n
        print(f"| {CAT_NAMES.get(cat, cat)} | {n} | {r10:.3f} | {r100:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
