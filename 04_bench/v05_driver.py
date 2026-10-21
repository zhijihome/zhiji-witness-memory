#!/usr/bin/env python3
"""V0.5 retrieval-hardening arm driver (workorder AML-V0.5 §1-H5).

Reuses the V0 ablation driver's per-arm mechanics (fresh temp db per arm, the
same behavior-relevant --set overrides passed to BOTH serve.py and
run_bench.py, G24/G25 asserted on every new ledger row). LOCAL coordinate
only; no official quotas are touched.

Usage:
    .venv/bin/python 04_bench/v05_driver.py                     # full C-suite
    .venv/bin/python 04_bench/v05_driver.py --only c0           # single arm
    .venv/bin/python 04_bench/v05_driver.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_bench"))

from ablation_driver import KEY_FIELDS, run_arm  # noqa: E402

import _bootstrap  # noqa: E402,F401


def build_arms() -> list[dict[str, Any]]:
    """V0 defaults restored explicitly on every arm so each ledger row is
    self-describing regardless of what defaults.json says at run time."""

    def base(dense: str, boost: float, extra: list[str] | None = None) -> list[str]:
        sets = [f"dense.backend={dense}", f"temporal.boost_delta={boost}"] + (extra or [])
        return sets

    arms: list[dict[str, Any]] = []
    # C0 — V0 default (对照, = ledger B0): hashed dense, boost 0.002, rrf,
    #      no neighbor, exact on. G29/G33 evidence.
    arms.append({"id": "c0", "label": "v05-c0-v0-default", "sets": base("hashed_ngram", 0.002)})
    # b3-fixed — ORDERING_POLICY=temporal_asc on the C0 config. After the H1
    #      select/order decoupling its r@100 must equal C0's (G30).
    arms.append(
        {
            "id": "b3-fixed",
            "label": "v05-b3-temporal-asc-fixed",
            "sets": base("hashed_ngram", 0.002) + ["surface.ORDERING_POLICY=temporal_asc"],
        }
    )
    # C1 — H2 alone: semantic dense, everything else V0.
    arms.append({"id": "c1", "label": "v05-c1-st-dense", "sets": base("sentence_transformers", 0.002)})
    # C2 — H2 + H4: semantic dense + boost 0.01.
    arms.append({"id": "c2", "label": "v05-c2-st-boost01", "sets": base("sentence_transformers", 0.01)})
    # C3 — C2 + neighbor window:1.
    arms.append(
        {
            "id": "c3",
            "label": "v05-c3-st-boost01-window1",
            "sets": base("sentence_transformers", 0.01, ["surface.NEIGHBOR_POLICY=window:1"]),
        }
    )
    # C4 — C3 + H3 fusion (parameter-free CombMNZ).
    arms.append(
        {
            "id": "c4",
            "label": "v05-c4-normsum",
            "sets": base(
                "sentence_transformers", 0.01, ["surface.NEIGHBOR_POLICY=window:1", "fusion.method=norm_sum"]
            ),
        }
    )
    # C5 — C4 with exact off (exact keep/drop ruling arm).
    arms.append(
        {
            "id": "c5",
            "label": "v05-c5-normsum-no-exact",
            "sets": base(
                "sentence_transformers",
                0.01,
                ["surface.NEIGHBOR_POLICY=window:1", "fusion.method=norm_sum", "retrievers.exact=false"],
            ),
        }
    )
    # G31 single-channel controls at the C4 config (same fusion/backends).
    for solo, others in (
        ("bm25", ["retrievers.exact=false", "retrievers.dense=false"]),
        ("dense", ["retrievers.exact=false", "retrievers.bm25=false"]),
        ("exact", ["retrievers.bm25=false", "retrievers.dense=false"]),
    ):
        arms.append(
            {
                "id": f"solo-{solo}@c4",
                "label": f"v05-solo-{solo}-at-c4",
                "sets": base(
                    "sentence_transformers",
                    0.01,
                    ["surface.NEIGHBOR_POLICY=window:1", "fusion.method=norm_sum"] + others,
                ),
            }
        )
    # H4 sensitivity — boost delta neighborhood at the C4 config (0.01 is C4).
    for delta in (0.005, 0.02):
        arms.append(
            {
                "id": f"sens-boost-{delta}",
                "label": f"v05-sens-boost-{delta}",
                "sets": base(
                    "sentence_transformers",
                    delta,
                    ["surface.NEIGHBOR_POLICY=window:1", "fusion.method=norm_sum"],
                ),
            }
        )
    # H3 sensitivity — minmax normalization variant of C4 (max-norm is C4).
    arms.append(
        {
            "id": "sens-norm-minmax",
            "label": "v05-sens-norm-minmax",
            "sets": base(
                "sentence_transformers",
                0.01,
                ["surface.NEIGHBOR_POLICY=window:1", "fusion.method=norm_sum", "fusion.norm=minmax"],
            ),
        }
    )
    # H3 attribution — rrf with exact off: separates the exact-drop effect from
    # the fusion-method effect (C5 = norm_sum + no exact vs C6 = rrf + no exact).
    arms.append(
        {
            "id": "c6",
            "label": "v05-c6-rrf-no-exact",
            "sets": base(
                "sentence_transformers",
                0.01,
                ["surface.NEIGHBOR_POLICY=window:1", "retrievers.exact=false"],
            ),
        }
    )
    # H3 2x2 completion — minmax with exact off (candidate final default).
    arms.append(
        {
            "id": "c7",
            "label": "v05-c7-minmax-no-exact",
            "sets": base(
                "sentence_transformers",
                0.01,
                ["surface.NEIGHBOR_POLICY=window:1", "fusion.method=norm_sum", "fusion.norm=minmax", "retrievers.exact=false"],
            ),
        }
    )
    # H4 boundary probe — is the additive boost alive at all under the
    # norm_sum score scale (~1.0) when raised 10x above the default?
    arms.append(
        {
            "id": "sens-boost-0.1",
            "label": "v05-sens-boost-0.1",
            "sets": base(
                "sentence_transformers",
                0.1,
                ["surface.NEIGHBOR_POLICY=window:1", "fusion.method=norm_sum", "fusion.norm=minmax", "retrievers.exact=false"],
            ),
        }
    )
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description="V0.5 retrieval-hardening arm driver (LOCAL only)")
    parser.add_argument("--results-root", default=str(REPO / "06_results"))
    parser.add_argument("--data", default=str(REPO / "04_bench" / "data" / "locomo10.json"))
    parser.add_argument("--only", default="", help="comma-separated arm ids to run (default: all)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit-convs", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    args = parser.parse_args()

    arms = build_arms()
    if args.list:
        for arm in arms:
            print(f"{arm['id']:22} {arm['label']:34} {arm['sets']}")
        return 0
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        arms = [a for a in arms if a["id"] in wanted]
        if not arms:
            print(f"no arms matched --only {args.only}", file=sys.stderr)
            return 2

    if not Path(args.data).exists():
        print(f"dataset not found: {args.data} — run 04_bench/fetch_locomo.py first", file=sys.stderr)
        return 2
    results_root = Path(args.results_root)
    summary_path = results_root / "experiments" / "v05_summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"arms to run: {[a['id'] for a in arms]}", flush=True)
    for arm in arms:
        started = time.time()
        print(f"\n=== ARM {arm['id']} (label={arm['label']}, sets={arm['sets']}) ===", flush=True)
        record = run_arm(
            arm,
            results_root=results_root,
            data=Path(args.data),
            top_k=args.top_k,
            limit_convs=args.limit_convs,
            limit_questions=args.limit_questions,
        )
        elapsed = time.time() - started
        for field in KEY_FIELDS:
            print(f"{field:24} {record.get(field)}", flush=True)
        print(
            f"DENSE_DEVICE             {record.get('DENSE_DEVICE')}  MODEL_CALLS {record.get('MODEL_CALLS')}",
            flush=True,
        )
        print(f"arm {arm['id']} done in {elapsed:.1f}s — RUN_ID {record['RUN_ID']}", flush=True)
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "arm": arm["id"],
                        "label": arm["label"],
                        "sets": arm["sets"],
                        "top_k": args.top_k,
                        "limit_convs": args.limit_convs,
                        "limit_questions": args.limit_questions,
                        "wall_seconds": round(elapsed, 1),
                        "record": record,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("\nV0.5 SUITE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
