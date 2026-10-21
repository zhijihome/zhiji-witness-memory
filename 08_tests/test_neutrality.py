"""G21 — BENCH-ID-NEUTRALITY self-audit.

Greps the competition logic (store / retrieval / adapter / api / config) for
benchmark-identifying literals. The sanctioned exception zone is 04_bench
(dataset loading + harness) and the frozen docs (00_contract, 07_workorders),
which contain the literals as documentation, never as logic.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LOGIC_DIRS = ["01_store", "02_retrieval", "03_adapter", "09_api", "10_config", "11_atomic"]

BANNED_LITERALS = [
    "locomo",
    "locomo_refined",
    "locomorefined",
    "conv-",
    "eval:",
    "bench:",
    "sample:",
]


def _py_files(folder: Path):
    for py in folder.rglob("*.py"):
        if py.name.startswith("._") or "__pycache__" in py.parts:
            continue  # macOS AppleDouble junk on exFAT volumes
        yield py


def test_no_benchmark_literals_in_logic_code():
    violations: list[str] = []
    for folder in LOGIC_DIRS:
        for py in _py_files(REPO / folder):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
                low = line.lower()
                for lit in BANNED_LITERALS:
                    if lit in low:
                        violations.append(f"{py.relative_to(REPO)}:{i}: {lit} -> {line.strip()}")
    assert not violations, "BENCH-ID-NEUTRALITY violations:\n" + "\n".join(violations)


def test_ids_treated_as_opaque_no_branching():
    """Structural check: no code path parses or pattern-matches the three IDs."""
    import re

    id_parsing = re.compile(r"(request_id|user_id|session_id)\s*(\.startswith|\.endswith|\.split|in\s*\[|==\s*[\"'])")
    for folder in LOGIC_DIRS:
        for py in _py_files(REPO / folder):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
                if id_parsing.search(line) and "pytest" not in line:
                    raise AssertionError(f"suspicious ID branching at {py.relative_to(REPO)}:{i}: {line.strip()}")
