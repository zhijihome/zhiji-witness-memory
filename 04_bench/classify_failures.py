#!/usr/bin/env python3
"""Failure-sample classification for the V0 matrix (workorder R1.1 §3.3).

Classifies per-question retrieval misses from a run's questions.jsonl detail.
Taxonomy (sanctioned by the workorder):

  RANKING_ONLY      gold evidence IS inside the returned face but ranked
                    below the cut (recall@10 == 0 while recall@face > 0)
  TEMPORAL_WINDOW   query carries explicit time anchors and the gold evidence
                    text matches those anchors, yet the unit was not surfaced
  MULTI_EVIDENCE    question needs several evidence units; some were reached
                    deeper but not all made the face
  LEXICAL_MISMATCH  zero content-token overlap between query and every gold
                    evidence text (pure vocabulary miss)
  SURROGATE_CEILING evidence ranked well (recall@10 > 0) but the fixed
                    answerer still produced F1 == 0 — counted separately
  OTHER             anything else (annotated when the gold unit exists beyond
                    the official return face)

All labels below the raw facts are heuristics produced by this script — the
experiment report must present them as Experimenter inference, not fact.

Usage:
    .venv/bin/python 04_bench/classify_failures.py \
        --details 06_results/runs/<run_id>/questions.jsonl \
        [--base-url http://127.0.0.1:8093]        # optional deep probe @1000
        [--out-csv 06_results/experiments/failure_cases.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_bench"))

from aml_retrieval.temporal import query_anchors  # noqa: E402  (read-only reuse of engine logic)
from aml_retrieval.tokenize import content_tokens  # noqa: E402
from benchlib.locomo_loader import Conversation, load_locomo  # noqa: E402

import _bootstrap  # noqa: E402,F401

CLASSES = [
    "RANKING_ONLY",
    "TEMPORAL_WINDOW",
    "MULTI_EVIDENCE",
    "LEXICAL_MISMATCH",
    "SURROGATE_CEILING",
    "OTHER",
]


def dia_map(conv: Conversation) -> dict[str, str]:
    """mem-id → dia_id, replicating the runner's chunked add order (chunk 20)."""
    out: dict[str, str] = {}
    n = 1
    for session in conv.sessions:
        for start in range(0, len(session), 20):
            for turn in session[start : start + 20]:
                out[f"mem-{n:06d}"] = turn.dia_id
                n += 1
    return out


def build_context(data_path: str | Path) -> dict[str, tuple[Conversation, dict[str, str]]]:
    ctx: dict[str, tuple[Conversation, dict[str, str]]] = {}
    for conv in load_locomo(data_path):
        ctx[conv.sample_id] = (conv, dia_map(conv))
    return ctx


def _token_overlap(a: str, b: str) -> int:
    ta = set(content_tokens(a))
    tb = set(content_tokens(b))
    return len(ta & tb)


def classify(question: str, evidence_ids: tuple[str, ...], units: dict[str, tuple[str, int | None]],
             r10: float, r100: float, f1: float, deep_hits: dict[str, bool] | None) -> tuple[str, str]:
    evidence_texts = [units.get(d, ("", None))[0] for d in evidence_ids]
    anchors = query_anchors(question)
    overlap = sum(_token_overlap(question, t) for t in evidence_texts)

    if r100 > 0:
        if r10 > 0:
            return ("SURROGATE_CEILING", "evidence in top-10 but F1=0") if f1 == 0 else ("OTHER", "no miss")
        return ("RANKING_ONLY", "evidence inside the face but ranked below 10")
    # r100 == 0: true recall miss against the official face
    if deep_hits is not None:
        n_deep = sum(1 for d in evidence_ids if deep_hits.get(d))
        if n_deep == len(evidence_ids) and len(evidence_ids) >= 1:
            return ("OTHER", "all gold units exist beyond the official face (found @1000 probe)")
        if 0 < n_deep < len(evidence_ids):
            return ("MULTI_EVIDENCE", f"{n_deep}/{len(evidence_ids)} gold units reached @1000, none in face")
    if anchors and any(
        any(qa.overlaps(_anchor_set(t, ts)) for qa in anchors) for t, ts in ((units.get(d, ("", None))) for d in evidence_ids)
    ):
        return ("TEMPORAL_WINDOW", "query anchored in time, gold text matches the anchor, still not surfaced")
    if overlap == 0:
        return ("LEXICAL_MISMATCH", "no content-token overlap between query and gold evidence")
    return ("OTHER", "recall miss without an obvious lexical/temporal signature")


def _anchor_set(text: str, ts_ms: int | None):
    from aml_retrieval.temporal import extract_anchors, unit_anchor_from_timestamp

    anchors = extract_anchors(text, ts_ms)
    if ts_ms is not None:
        anchors.append(unit_anchor_from_timestamp(ts_ms))
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser(description="classify retrieval misses (Experimenter heuristics)")
    parser.add_argument("--details", required=True, help="questions.jsonl of the run to analyze")
    parser.add_argument("--data", default=str(REPO / "04_bench" / "data" / "locomo10.json"))
    parser.add_argument("--base-url", default=None, help="optional: live B0-config service for @1000 probe")
    parser.add_argument("--categories", default="3,1", help="LOCOMO categories to include (default: 3,1)")
    parser.add_argument("--top-k-probe", type=int, default=1000)
    parser.add_argument("--out-csv", default=str(REPO / "06_results" / "experiments" / "failure_cases.csv"))
    args = parser.parse_args()

    wanted_cats = {int(x) for x in args.categories.split(",") if x.strip()}
    ctx = build_context(args.data)

    # the probe must search the SAME per-conversation users the run wrote to,
    # otherwise it sees an empty store; run_id comes from the details path
    run_id = Path(args.details).resolve().parent.name

    client = httpx.Client(base_url=args.base_url, timeout=60.0) if args.base_url else None

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {c: 0 for c in CLASSES}
    surrogate_ceiling = 0
    for line in Path(args.details).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["category"] not in wanted_cats:
            continue
        conv, mem2dia = ctx[rec["sample_id"]]
        units = {t.dia_id: (t.text, t.ts_ms) for session in conv.sessions for t in session}
        q = next(q for q in conv.questions if q.question == rec["question"])
        evidence_ids = q.evidence

        deep_hits: dict[str, bool] | None = None
        if client is not None:
            resp = client.post(
                "/search",
                json={
                    "query": rec["question"],
                    "user_id": f"bench:{run_id}:c{rec['conv']}",
                    "top_k": args.top_k_probe,
                },
            )
            resp.raise_for_status()
            returned_dias = [mem2dia.get(it["id"], "") for it in resp.json().get("data", [])]
            deep_hits = {d: (d in returned_dias) for d in evidence_ids}

        label, note = classify(
            rec["question"], evidence_ids, units, rec["evidence_recall_at_10"],
            rec["evidence_recall_at_100"], rec["f1"], deep_hits,
        )
        if label == "SURROGATE_CEILING":
            surrogate_ceiling += 1
            continue
        if label == "OTHER" and note == "no miss":
            continue
        counts[label] += 1
        rows.append(
            {
                "sample_id": rec["sample_id"],
                "category": rec["category"],
                "question": rec["question"],
                "gold_answer": rec.get("gold", ""),
                "evidence_ids": ",".join(evidence_ids),
                "recall_at_10": rec["evidence_recall_at_10"],
                "recall_at_100": rec["evidence_recall_at_100"],
                "f1": rec["f1"],
                "class": label,
                "note": note,
            }
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["sample_id", "class"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"analyzed categories: {sorted(wanted_cats)}")
    print(f"miss rows written: {len(rows)} -> {out}")
    print("SURROGATE_CEILING (evidence ranked fine, answerer failed):", surrogate_ceiling)
    for c in CLASSES:
        if c != "SURROGATE_CEILING":
            print(f"{c:18} {counts[c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
