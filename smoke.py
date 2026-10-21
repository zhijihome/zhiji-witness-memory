#!/usr/bin/env python3
"""Zero-dependency end-to-end smoke test against a running service (stdlib only).

Verifies the contract core: health 2xx, add→immediate search hit, user
isolation, duplicate-add idempotency, and the exact response shapes. Usable
against the dev server, docker compose, or the official-style remote endpoint.

Usage:
    python smoke.py --base-url http://127.0.0.1:8093 [--wait 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def _request(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict | list | None, str]:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None), raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw), raw
        except json.JSONDecodeError:
            return exc.code, None, raw


def _wait_healthy(base: str, wait_s: float) -> bool:
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            status, body, _ = _request(base, "GET", "/health")
            if 200 <= status < 300 and isinstance(body, dict) and body.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8093")
    parser.add_argument("--wait", type=float, default=30.0)
    args = parser.parse_args()

    if not _wait_healthy(args.base_url, args.wait):
        print("SMOKE FAIL: service did not become healthy")
        return 1

    run = uuid.uuid4().hex[:8]
    user = f"smoke-user-{run}"
    ok = True

    payload = {
        "request_id": f"smoke-{run}-1",
        "messages": [
            {
                "role": "tester",
                "timestamp": 1700000000000,
                "content": f"smoke {run} the viola teacher moved to Lisbon.",
            },
            {
                "role": "tester",
                "timestamp": 1700000600000,
                "content": f"smoke {run} her cat is called Miso.",
            },
        ],
        "user_id": user,
        "session_id": f"smoke-{run}-s1",
    }
    status, body, raw = _request(args.base_url, "POST", "/add", payload)
    if status != 200 or body != {
        "success": True,
        "request_id": payload["request_id"],
        "user_id": user,
        "session_id": payload["session_id"],
    }:
        print(f"SMOKE FAIL add: {status} {raw[:200]}")
        ok = False

    status, body, _ = _request(args.base_url, "POST", "/add", payload)  # duplicate delivery
    if status != 200 or not (isinstance(body, dict) and body.get("success") is True):
        print(f"SMOKE FAIL duplicate add: {status}")
        ok = False

    status, body, _ = _request(
        args.base_url, "POST", "/search", {"query": f"viola teacher {run} Lisbon", "user_id": user, "top_k": 10}
    )
    data = (body or {}).get("data") if isinstance(body, dict) else None
    if status != 200 or not isinstance(data, list) or not any("viola" in i.get("content", "") for i in data):
        print(f"SMOKE FAIL search: {status} {str(body)[:200]}")
        ok = False
    elif any(("answer" in i) for i in data):
        print("SMOKE FAIL search: answer field present")
        ok = False

    status, body, _ = _request(
        args.base_url, "POST", "/search", {"query": f"viola teacher {run}", "user_id": f"other-{run}", "top_k": 10}
    )
    data = (body or {}).get("data") if isinstance(body, dict) else None
    if status != 200 or not isinstance(data, list) and data != []:
        print(f"SMOKE FAIL isolation: {status} {str(body)[:200]}")
        ok = False
    else:
        if any("viola" in i.get("content", "") for i in (data or [])):
            print("SMOKE FAIL isolation: cross-user leak")
            ok = False

    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
