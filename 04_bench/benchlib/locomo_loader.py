"""LOCOMO (locomo10.json) loader — the local proxy dataset (Contract §10-U5).

This file is one of the sanctioned exceptions where dataset literals may live
(workorder §6 / G21). The public dataset is LOCOMO from snap-research/locomo,
license CC BY-NC 4.0 — see README for acquisition; we do not redistribute it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_DATE_RE = re.compile(
    r"(?:\d{1,2}:\d{2}\s*(?:am|pm)\s+)?on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.IGNORECASE
)

# LOCOMO question categories (dataset paper): 1 single-hop, 2 multi-hop,
# 3 temporal, 4 open-domain knowledge, 5 adversarial.
CATEGORY_TO_AML_DIM = {
    1: "FACT",
    2: "MULTI_HOP",
    3: "TEMPORAL",
    4: None,  # world knowledge — no AML memory-dim equivalent
    5: None,  # adversarial — no AML memory-dim equivalent
}


@dataclass(frozen=True)
class LocomoTurn:
    dia_id: str
    speaker: str
    text: str
    ts_ms: int | None


@dataclass(frozen=True)
class LocomoQuestion:
    question: str
    answer: str
    evidence: tuple[str, ...]
    category: int


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    sessions: list[list[LocomoTurn]] = field(default_factory=list)
    questions: list[LocomoQuestion] = field(default_factory=list)


def _parse_session_date(raw: str) -> int | None:
    m = _DATE_RE.search(raw or "")
    if not m:
        return None
    day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
    month = _MONTHS.get(month_name)
    if not month:
        return None
    try:
        return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)
    except ValueError:
        return None


def load_locomo(path: str | Path) -> list[Conversation]:
    raw: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Conversation] = []
    for conv in raw:
        sample_id = str(conv.get("sample_id", f"sample{len(out)}"))
        sessions: list[list[LocomoTurn]] = []
        session_keys = sorted(
            (k for k in conv["conversation"] if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        for skey in session_keys:
            date_raw = conv["conversation"].get(f"{skey}_date_time", "")
            ts_ms = _parse_session_date(date_raw)
            turns = [
                LocomoTurn(dia_id=t["dia_id"], speaker=t.get("speaker", "user"), text=t.get("text", ""), ts_ms=ts_ms)
                for t in conv["conversation"][skey]
                if t.get("text")
            ]
            if turns:
                sessions.append(turns)
        questions = [
            LocomoQuestion(
                question=str(q.get("question", "")).strip(),
                answer=str(q.get("answer", "")).strip(),
                evidence=tuple(str(e) for e in q.get("evidence", []) if str(e).strip()),
                category=int(q.get("category", 0)),
            )
            for q in conv.get("qa", [])
            if str(q.get("question", "")).strip()
        ]
        out.append(Conversation(sample_id=sample_id, sessions=sessions, questions=questions))
    return out
