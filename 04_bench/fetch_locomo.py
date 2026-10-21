#!/usr/bin/env python3
"""Fetch the public LOCOMO dataset (locomo10.json) into 04_bench/data/.

Source: https://github.com/snap-research/locomo (data/locomo10.json)
License: CC BY-NC 4.0 — non-commercial; we do not redistribute it, and it never
enters the repository (04_bench/data/ is gitignored).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
MIN_BYTES = 1_000_000  # sanity: real file is ~2.8 MB


def main() -> int:
    target = Path(__file__).resolve().parent / "data" / "locomo10.json"
    if target.exists() and target.stat().st_size >= MIN_BYTES:
        print(f"already present: {target} ({target.stat().st_size} bytes)")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp, open(target, "wb") as fh:
        fh.write(resp.read())
    size = target.stat().st_size
    if size < MIN_BYTES:
        print(f"downloaded file suspiciously small ({size} bytes); deleting", file=sys.stderr)
        target.unlink()
        return 1
    print(f"saved {target} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
