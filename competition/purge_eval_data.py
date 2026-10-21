"""Cycle-2 eval-data purge (rules: "delete within 30 days after the run").

Every byte the evaluation sends us lives in two files under the app root:

  * 06_results/aml.db                     — raw units, atomic derived docs,
                                            governance journal (all eval content)
  * 06_results/embedding_cache/dashscope.sqlite
                                          — content-keyed embedding cache

Plus WAL/SHM sidecars next to each. The container mounts no volumes, so
`docker rm` already destroys everything; this script is the in-place,
in-container (or checkout-local) equivalent for when the service must stay up
or the deletion must be evidenced without killing the deployment.

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
