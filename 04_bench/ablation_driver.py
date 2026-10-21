#!/usr/bin/env python3
"""Serial ablation driver for the V0 matrix (workorder R1.1 §3).

Boots one throwaway service per arm (fresh temp db), waits for /health, then
runs `run_bench.py` with the arm's behavior-relevant --set overrides passed to
BOTH sides (server + client). The G24 fingerprint gate must pass on every arm —
the driver asserts SERVER_CONFIG == CONFIG on each new ledger record and never
modifies core code. LOCAL coordinate only; no official quotas are touched.

Usage:
    .venv/bin/python 04_bench/ablation_driver.py                     # full matrix
    .venv/bin/python 04_bench/ablation_driver.py --only b0-default
    .venv/bin/python 04_bench/ablation_driver.py --limit-convs 2 --limit-questions 5 --results-root /tmp/probe
"""

from __future__ import annotations

import argparse
import json
import socket
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_bench"))

import _bootstrap  # noqa: E402,F401


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_healthy(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001 — refused until up
            last = exc
        time.sleep(0.3)
    raise RuntimeError(f"service did not become healthy: {last}")


def build_arms() -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []

    # B0 — default config, straight rerun (baseline reproduction check)
    arms.append({"id": "b0-default", "label": "b0-mac-repro", "sets": []})

    # B1 — retriever switches: all 7 non-empty combinations
    combos: dict[str, list[str]] = {
        "b1-exact": ["retrievers.bm25=false", "retrievers.dense=false"],
        "b1-bm25": ["retrievers.exact=false", "retrievers.dense=false"],
        "b1-dense": ["retrievers.exact=false", "retrievers.bm25=false"],
        "b1-exact-bm25": ["retrievers.dense=false"],
        "b1-exact-dense": ["retrievers.bm25=false"],
        "b1-bm25-dense": ["retrievers.exact=false"],
        "b1-all-on": [],
    }
    for combo_id, combo_sets in combos.items():
        arms.append({"id": combo_id, "label": f"b1-{combo_id}", "sets": combo_sets})

    # B2 — RETURN_SIZE (100 explicit; with bench top_k=100 it equals "top_k")
    for size in (5, 10, 20, 40, 100):
        arms.append(
            {"id": f"b2-return-{size}", "label": f"b2-return-size-{size}", "sets": [f"surface.RETURN_SIZE={size}"]}
        )

    # B3 — ORDERING_POLICY (relevance_desc is the B0 default)
    arms.append(
        {"id": "b3-temporal-asc", "label": "b3-ordering-temporal-asc", "sets": ["surface.ORDERING_POLICY=temporal_asc"]}
    )

    # B4 — NEIGHBOR_POLICY (off is the B0 default)
    for policy in ("window:1", "window:2"):
        arms.append(
            {
                "id": f"b4-{policy.replace(':', '')}",
                "label": f"b4-neighbor-{policy}",
                "sets": [f"surface.NEIGHBOR_POLICY={policy}"],
            }
        )

    # B5 — dense backend (hashed_ngram is the B0 default)
    arms.append(
        {
            "id": "b5-st",
            "label": "b5-dense-sentence-transformers",
            "sets": ["dense.backend=sentence_transformers"],
        }
    )

    # B6 — fusion.rrf_k (60 is the B0 default; optional per workorder §3.1)
    for k in (20, 120):
        arms.append({"id": f"b6-rrf{k}", "label": f"b6-rrf-k-{k}", "sets": [f"fusion.rrf_k={k}"]})

    # B7 — temporal.boost_delta (0.002 is the B0 default; optional)
    for delta in (0, 0.01):
        arms.append(
            {
                "id": f"b7-delta{str(delta).replace('.', '')}",
                "label": f"b7-boost-delta-{delta}",
                "sets": [f"temporal.boost_delta={delta}"],
            }
        )
    return arms


def _flagged(sets: list[str]) -> list[str]:
    """['a=1', 'b=2'] → ['--set', 'a=1', '--set', 'b=2'] (bare strings would
    land in argparse's positional slot and die as unrecognized arguments)."""
    out: list[str] = []
    for item in sets:
        out += ["--set", item]
    return out


def run_arm(
    arm: dict[str, Any],
    *,
    results_root: Path,
    data: Path,
    top_k: int,
    limit_convs: int | None,
    limit_questions: int | None,
) -> dict[str, Any]:
    arm_id, label, sets = arm["id"], arm["label"], arm["sets"]
    db = Path(tempfile.mkdtemp(prefix=f"aml_{arm_id}_")) / "arm.db"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    run_id = f"{arm_id}-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"

    server_log = db.parent / "server.log"
    server_stderr = open(server_log, "w")
    server = subprocess.Popen(
        [sys.executable, str(REPO / "serve.py"), "--port", str(port), "--set", f"storage.db_path={db}", *_flagged(sets)],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=server_stderr,
    )
    try:
        try:
            wait_healthy(port, 60.0)
        except RuntimeError:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
            err_tail = ""
            if server.stderr is not None:
                try:
                    err_tail = server.stderr.read().decode(errors="replace")[-2000:]
                except Exception:  # noqa: BLE001 — diagnostics only
                    err_tail = "<stderr unavailable>"
            raise RuntimeError(
                f"arm {arm_id}: service never became healthy on :{port}; server stderr tail:\n{err_tail}"
            ) from None
        bench_cmd = [
            sys.executable,
            str(REPO / "04_bench" / "run_bench.py"),
            "--base-url",
            base_url,
            "--results-root",
            str(results_root),
            "--run-id",
            run_id,
            "--label",
            label,
            "--top-k",
            str(top_k),
            *_flagged(sets),
        ]
        if limit_convs:
            bench_cmd += ["--limit-convs", str(limit_convs)]
        if limit_questions:
            bench_cmd += ["--limit-questions", str(limit_questions)]
        proc = subprocess.run(bench_cmd, cwd=REPO, capture_output=True, text=True, timeout=int(os.environ.get("AML_ARM_TIMEOUT", "7200")))
        if proc.returncode != 0:
            server_stderr.flush()
            log_tail = ""
            try:
                log_tail = server_log.read_text()[-1500:]
            except OSError:
                pass
            raise RuntimeError(
                f"arm {arm_id} run_bench rc={proc.returncode}\n--- stdout tail ---\n{proc.stdout[-1500:]}"
                f"\n--- stderr tail ---\n{proc.stderr[-1500:]}"
                f"\n--- server log tail ({server_log}) ---\n{log_tail}"
            )
        ledger_path = results_root / "ledger" / "ledger.jsonl"
        record = json.loads(ledger_path.read_text().splitlines()[-1])
        if record.get("RUN_ID") != run_id:
            raise RuntimeError(f"arm {arm_id}: ledger tail is {record.get('RUN_ID')}, expected {run_id}")
        if record.get("SERVER_CONFIG") != record.get("CONFIG"):
            raise RuntimeError(f"arm {arm_id}: SERVER_CONFIG != CONFIG in new ledger record (G25 broken?)")
        return record
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


KEY_FIELDS = [
    "RETRIEVERS_ON",
    "RETURN_SIZE",
    "ORDERING_POLICY",
    "NEIGHBOR_POLICY",
    "EVIDENCE_RECALL_AT_10",
    "EVIDENCE_RECALL_AT_100",
    "OVERALL",
    "LATENCY_P50",
    "LATENCY_P95",
    "N_QUESTIONS",
    "FAILURE_CASES",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="V0 ablation matrix driver (LOCAL coordinate only)")
    parser.add_argument("--results-root", default=str(REPO / "06_results"))
    parser.add_argument("--data", default=str(REPO / "04_bench" / "data" / "locomo10.json"))
    parser.add_argument("--only", default="", help="comma-separated arm ids to run (default: all)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit-convs", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"dataset not found: {args.data} — run 04_bench/fetch_locomo.py first", file=sys.stderr)
        return 2

    arms = build_arms()
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        arms = [a for a in arms if a["id"] in wanted]

    results_root = Path(args.results_root)
    summary_path = results_root / "experiments" / "ablation_summary.jsonl"
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
    print("\nMATRIX DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
