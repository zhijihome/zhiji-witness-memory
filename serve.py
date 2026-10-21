#!/usr/bin/env python3
"""Dev entrypoint: run the service from a source checkout (no install needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401

from aml_api.server import main  # noqa: E402

if __name__ == "__main__":
    main()
