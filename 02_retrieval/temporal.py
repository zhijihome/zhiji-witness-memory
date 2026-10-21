"""Basic temporal anchors (V0 scope, workorder deliverable 11).

Builds time anchors from `messages[].timestamp` (the reference point) plus
textual time expressions in the content. Anchors are absolute millisecond
intervals; relative expressions ("yesterday", "3 weeks ago") resolve against
the unit's own timestamp when present. No standing/supersede semantics here —
V0 only anchors and lightly boosts temporally-matching evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_DAY_MS = 86_400_000

_RE_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_RE_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\.?(?:,)?\s+(\d{4})\b", re.IGNORECASE
)
_RE_MONTH_DAY_YEAR = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b", re.IGNORECASE
)
_RE_MONTH_YEAR = re.compile(r"\b(" + "|".join(_MONTHS) + r")\.?(?:,)?\s+(\d{4})\b", re.IGNORECASE)
_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_RE_REL_DAYS_AGO = re.compile(r"\b(\d{1,3})\s+(day|week|month|year)s?\s+ago\b", re.IGNORECASE)
_RE_REL_IN_N = re.compile(r"\bin\s+(\d{1,3})\s+(day|week|month|year)s?\b", re.IGNORECASE)
_RE_REL_LAST_NEXT = re.compile(r"\b(last|next|past)\s+(week|month|year)\b", re.IGNORECASE)
_RE_YESTERDAY = re.compile(r"\b(yesterday|today|tomorrow)\b", re.IGNORECASE)

_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


@dataclass(frozen=True)
class TimeAnchor:
    start_ms: int
    end_ms: int
    kind: str  # day | month | year | relative
    raw: str

    def overlaps(self, other: TimeAnchor) -> bool:
        return self.start_ms <= other.end_ms and other.start_ms <= self.end_ms


def _utc_ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


def _month_end(year: int, month: int) -> int:
    if month == 12:
        nxt = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        nxt = datetime(year, month + 1, 1, tzinfo=UTC)
    return int(nxt.timestamp() * 1000) - 1


def _shift(ref_ms: int, days: int) -> tuple[int, int]:
    base = datetime.fromtimestamp(ref_ms / 1000.0, tz=UTC)
    start = base + timedelta(days=days)
    return int(start.timestamp() * 1000), int(start.timestamp() * 1000) + _DAY_MS - 1


def extract_anchors(text: str, ref_ts_ms: int | None) -> list[TimeAnchor]:
    """Extract absolute time intervals from `text`.

    Relative expressions require a reference timestamp; without one they are
    skipped (we never fabricate a reference).
    """
    anchors: list[TimeAnchor] = []
    seen_spans: list[tuple[int, int]] = []

    def _taken(span: tuple[int, int]) -> bool:
        return any(not (span[1] <= s[0] or s[1] <= span[0]) for s in seen_spans)

    def _add(m: re.Match, anchor: TimeAnchor) -> None:
        span = m.span()
        if not _taken(span):
            seen_spans.append(span)
            anchors.append(anchor)

    for m in _RE_ISO_DATE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            s = _utc_ms(y, mo, d)
        except ValueError:
            continue
        _add(m, TimeAnchor(s, s + _DAY_MS - 1, "day", m.group(0)))

    for m in _RE_DAY_MONTH_YEAR.finditer(text):
        d, mon, y = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
        try:
            s = _utc_ms(y, mon, d)
        except ValueError:
            continue
        _add(m, TimeAnchor(s, s + _DAY_MS - 1, "day", m.group(0)))

    for m in _RE_MONTH_DAY_YEAR.finditer(text):
        mon, d, y = _MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
        try:
            s = _utc_ms(y, mon, d)
        except ValueError:
            continue
        _add(m, TimeAnchor(s, s + _DAY_MS - 1, "day", m.group(0)))

    for m in _RE_MONTH_YEAR.finditer(text):
        mon, y = _MONTHS[m.group(1).lower()], int(m.group(2))
        try:
            _add(m, TimeAnchor(_utc_ms(y, mon, 1), _month_end(y, mon), "month", m.group(0)))
        except ValueError:
            continue

    if ref_ts_ms is not None:
        for m in _RE_REL_DAYS_AGO.finditer(text):
            n, unit = int(m.group(1)), m.group(2).lower()
            s, e = _shift(ref_ts_ms, -n * _UNIT_DAYS[unit])
            _add(m, TimeAnchor(s, e, "relative", m.group(0)))
        for m in _RE_REL_IN_N.finditer(text):
            n, unit = int(m.group(1)), m.group(2).lower()
            s, e = _shift(ref_ts_ms, n * _UNIT_DAYS[unit])
            _add(m, TimeAnchor(s, e, "relative", m.group(0)))
        for m in _RE_REL_LAST_NEXT.finditer(text):
            direction, unit = m.group(1).lower(), m.group(2).lower()
            span_days = _UNIT_DAYS[unit]
            base = datetime.fromtimestamp(ref_ts_ms / 1000.0, tz=UTC)
            if direction in ("last", "past"):
                start = int((base - timedelta(days=span_days)).timestamp() * 1000)
                end = int(base.timestamp() * 1000)
            else:
                start = int(base.timestamp() * 1000)
                end = int((base + timedelta(days=span_days)).timestamp() * 1000)
            _add(m, TimeAnchor(start, end, "relative", m.group(0)))
        for m in _RE_YESTERDAY.finditer(text):
            word = m.group(1).lower()
            delta = {"yesterday": -1, "today": 0, "tomorrow": 1}[word]
            s, e = _shift(ref_ts_ms, delta)
            _add(m, TimeAnchor(s, e, "relative", m.group(0)))

    for m in _RE_YEAR.finditer(text):
        y = int(m.group(1))
        try:
            _add(m, TimeAnchor(_utc_ms(y, 1, 1), _month_end(y, 12), "year", m.group(0)))
        except ValueError:
            continue

    return anchors


def query_anchors(query: str) -> list[TimeAnchor]:
    """Anchors usable at query time: absolute expressions only (no reference)."""
    return extract_anchors(query, ref_ts_ms=None)


def unit_anchor_from_timestamp(ts_ms: int) -> TimeAnchor:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    s = int(datetime(dt.year, dt.month, dt.day, tzinfo=UTC).timestamp() * 1000)
    return TimeAnchor(s, s + _DAY_MS - 1, "day", "unit_timestamp")
