"""Public-endpoint load probe — measures what the official harness will see.

Runs against $AML_BASE with AML_API_KEY auth. Sends Add
batches shaped like the official contract (role=user/assistant, optional
timestamp), then Search latency probes. Synthetic content only — this is a
perf probe, not an eval.

Usage: python competition/loadtest_public.py [--add-batches 6] [--searches 60]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("AML_BASE")  # required — the eval endpoint is not shipped in-repo

_LOREM = (
    "hiking trip to the mountains last weekend was exhausting but the view from "
    "the summit was worth it we took photos of the sunrise and Caroline packed "
    "sandwiches for everyone the trail was steeper than expected and my knees "
    "still hurt thinking about it next time we should start earlier to avoid "
    "the afternoon heat"
).split()


def _post(path: str, body: dict, key: str) -> tuple[int, float, bytes]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, (time.monotonic() - t0) * 1000, r.read()
    except urllib.error.HTTPError as e:
        return e.code, (time.monotonic() - t0) * 1000, e.read()


def _msg(rng: random.Random, i: int) -> dict:
    n = rng.randint(15, 45)
    return {
        "role": rng.choice(["user", "assistant"]),
        "content": " ".join(rng.choices(_LOREM, k=n)),
        "timestamp": 1704067200000 + i * 60000,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-batches", type=int, default=6)
    ap.add_argument("--msgs-per-batch", type=int, default=40)
    ap.add_argument("--searches", type=int, default=60)
    args = ap.parse_args()

    key = os.environ.get("AML_API_KEY")
    if not key:
        for cand in (
            Path(__file__).resolve().parents[2] / ".env",
            Path.cwd() / ".env",
        ):
            if cand.exists():
                kv = dict(l.split("=", 1) for l in cand.read_text().splitlines() if "=" in l)
                key = kv.get("AML_API_KEY")
                if key:
                    break
    if not key:
        sys.exit("AML_API_KEY not set and no .env found")
    if not BASE:
        sys.exit("AML_BASE not set — pass the target endpoint explicitly")
    rng = random.Random(7)
    uid = f"loadtest-{int(time.time())}"

    add_lat, total_msgs = [], 0
    for b in range(args.add_batches):
        msgs = [_msg(rng, i) for i in range(args.msgs_per_batch)]
        st, ms, body = _post(
            "/add",
            {"request_id": f"{uid}-b{b}", "messages": msgs, "user_id": uid, "session_id": uid},
            key,
        )
        if st != 200:
            print(f"ADD FAIL {st}: {body[:200]}", file=sys.stderr)
            return 1
        add_lat.append(ms)
        total_msgs += len(msgs)
    print(
        f"ADD  n={args.add_batches} batches ({total_msgs} msgs) "
        f"mean={statistics.mean(add_lat):.0f}ms p95={statistics.quantiles(add_lat, n=20)[-1]:.0f}ms"
    )

    queries = [
        "what did the user do last weekend",
        "who packed sandwiches",
        "when was the hiking trip",
        "how steep was the trail",
        "did anyone take photos of the sunrise",
    ]
    lat = []
    for i in range(args.searches):
        st, ms, body = _post(
            "/search",
            {"query": queries[i % len(queries)], "user_id": uid, "top_k": 100},
            key,
        )
        if st != 200:
            print(f"SEARCH FAIL {st}: {body[:200]}", file=sys.stderr)
            return 1
        lat.append(ms)
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
    print(
        f"SEARCH n={len(lat)} p50={p(.50):.0f}ms p95={p(.95):.0f}ms "
        f"p99={p(.99):.0f}ms max={lat[-1]:.0f}ms mean={statistics.mean(lat):.0f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
