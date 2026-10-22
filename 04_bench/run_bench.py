#!/usr/bin/env python3
"""One-command LOCAL benchmark runner (G7).

Typical use (service already running on :8093):

    python 04_bench/run_bench.py --base-url http://127.0.0.1:8093 --label mylabel

Or via make: `make bench` boots the service first, then runs this.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "04_bench"))

import _bootstrap  # noqa: E402,F401 — must precede any aml_* import

import httpx  # noqa: E402
from aml_config import load_config, merge_set_overrides  # noqa: E402
from benchlib.locomo_loader import load_locomo  # noqa: E402
from benchlib.runner import ConfigMismatchError, run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="zhiji-aml LOCAL benchmark runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8093")
    parser.add_argument("--config", default=str(REPO_ROOT / "04_bench" / "configs" / "bench_default.json"))
    parser.add_argument("--data", default=str(REPO_ROOT / "04_bench" / "data" / "locomo10.json"))
    parser.add_argument("--results-root", default=str(REPO_ROOT / "06_results"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--limit-convs", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--generic-roles", action="store_true",
                        help="send role=user/assistant like the official contract instead of speaker names")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--set", dest="sets", action="append", default=[], help="dotted.path=value override")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(
            f"dataset not found: {args.data}\n"
            "fetch it first:  python 04_bench/fetch_locomo.py\n"
            "(LOCOMO, CC BY-NC 4.0 — we do not redistribute it; see README)"
        )
        return 2

    overrides = merge_set_overrides(args.sets)
    config = load_config(path=args.config if Path(args.config).exists() else None, overrides=overrides or None)

    conversations = load_locomo(args.data)
    if args.limit_convs:
        conversations = conversations[: args.limit_convs]

    run_id = args.run_id or datetime.now(tz=UTC).strftime("local-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    import os as _os
    with httpx.Client(base_url=args.base_url, timeout=float(_os.environ.get("AML_BENCH_TIMEOUT", "120"))) as client:
        try:
            record = run_benchmark(
                client,
                conversations,
                config,
                run_id=run_id,
                results_root=args.results_root,
                label=args.label,
                top_k=args.top_k,
                limit_questions=args.limit_questions,
                generic_roles=args.generic_roles,
            )
        except ConfigMismatchError as exc:
            print(str(exc), file=sys.stderr)
            return 3

    print("=== LOCAL benchmark record ===")
    for key, value in record.items():
        print(f"{key:26} {value}")
    print(
        "\nNOTE: this is a LOCAL coordinate result on a public proxy dataset with a "
        "fixed surrogate answerer.\nIt is NOT an AML score (Contract §5, Architecture C3)."
    )
    return 0 if record["FAILURE_CASES"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
