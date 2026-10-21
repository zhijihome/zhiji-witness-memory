"""Cycle-2 eval-data purge (rules: "delete within 30 days after the run").

Everything the evaluation sends us lands under 06_results/:

  * 06_results/aml.db                     — raw units, atomic derived docs,
                                            governance journal (all eval content)
  * 06_results/embedding_cache/dashscope.sqlite
                                          — content-keyed embedding cache
  * 06_results/ops/ops.jsonl              — request journal: eval-supplied
                                            request/user/session ids + digests
  * 06_results/spend/spend_state.json     — call counters over that traffic

Plus WAL/SHM/journal sidecars next to each SQLite file. The deployment mounts
a named volume at /app/06_results so redeploys never drop eval data; retiring
the instance deletes it wholesale with `docker rm zvem && docker volume rm
aml-eval-data`. This script is the in-place, evidenced equivalent for when
the service must stay up.

Usage:
    python competition/purge_eval_data.py [--root /app] [--dry-run]
    # in the Docker image: python /app/competition/purge_eval_data.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

EVAL_DATA_FILES = (
    "06_results/aml.db",
    "06_results/embedding_cache/dashscope.sqlite",
    "06_results/ops/ops.jsonl",
    "06_results/spend/spend_state.json",
)
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def iter_eval_paths(root: Path) -> list[Path]:
    out: list[Path] = []
    for rel in EVAL_DATA_FILES:
        p = root / rel
        out.append(p)
        out.extend(p.with_name(p.name + sfx) for sfx in _SIDECAR_SUFFIXES)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Purge AML Cycle-2 eval data")
    ap.add_argument("--root", default=os.environ.get("AML_ROOT", "."))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    found = [p for p in iter_eval_paths(root) if p.exists()]
    if not found:
        print("purge: nothing found under", root)
        return 0
    for p in found:
        print(("would remove" if args.dry_run else "removing"), p)
        if not args.dry_run:
            p.unlink()
    if not args.dry_run:
        print(f"purge: removed {len(found)} file(s); eval data is gone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
