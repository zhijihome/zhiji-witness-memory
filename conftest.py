"""Root conftest: make layer packages importable from a bare source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401
