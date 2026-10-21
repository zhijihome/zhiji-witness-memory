"""Temporal anchors (V0 scope): extraction, query-side overlap, temporal_asc
ordering, and the boost — no standing/supersede anywhere."""

from __future__ import annotations

from datetime import UTC

from aml_retrieval.temporal import extract_anchors, query_anchors, unit_anchor_from_timestamp

REF = 1684000000000  # 2023-05-13


def ts_for(year: int, month: int = 1, day: int = 1) -> int:
    from datetime import datetime

    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


def test_absolute_expressions():
    anchors = extract_anchors("We met on 8 May 2023 and again on 2023-06-01.", None)
    kinds = sorted(a.kind for a in anchors)
    assert kinds == ["day", "day"]
    starts = sorted(a.start_ms for a in anchors)
    assert starts[0] == ts_for(2023, 5, 8)
    assert starts[1] == ts_for(2023, 6, 1)


def test_year_and_month_year():
    anchors = extract_anchors("Back in 2019, and during May 2023.", None)
    by_raw = {a.raw: a for a in anchors}
    assert by_raw["2019"].kind == "year"
    assert by_raw["2019"].start_ms == ts_for(2019)
    assert by_raw["May 2023"].kind == "month"
    assert by_raw["May 2023"].start_ms == ts_for(2023, 5, 1)


def test_relative_needs_reference():
    assert extract_anchors("yesterday", None) == []  # never fabricate a reference
    anchors = extract_anchors("I moved here 2 weeks ago", REF)
    assert len(anchors) == 1 and anchors[0].kind == "relative"
    # "2 weeks ago" denotes a day ~14 days before the reference, not a 14-day range
    span_days = (anchors[0].end_ms - anchors[0].start_ms) / 86_400_000
    assert span_days <= 1.0
    delta_days = (REF - anchors[0].start_ms) / 86_400_000
    assert 13 <= delta_days <= 15


def test_query_side_absolute_anchors():
    q = query_anchors("What did she do in 2023?")
    assert any(a.kind == "year" and a.start_ms == ts_for(2023) for a in q)


def test_overlap_semantics():
    a = extract_anchors("May 2023", None)[0]
    b = extract_anchors("8 May 2023", None)[0]
    c = extract_anchors("2019", None)[0]
    assert a.overlaps(b)
    assert not a.overlaps(c)


def test_unit_timestamp_anchor():
    anchor = unit_anchor_from_timestamp(ts_for(2023, 5, 8) + 3_600_000)
    assert anchor.start_ms == ts_for(2023, 5, 8)


def test_temporal_boost_pulls_matching_year_up(tmp_path, tmp_config):
    from aml_retrieval import RetrievalEngine
    from aml_store import AddPayload, CompetitionMemoryStore, MessageIn

    store = CompetitionMemoryStore(tmp_path / "t.db")
    store.add(
        AddPayload(
            request_id="a",
            user_id="u",
            session_id="s",
            messages=[MessageIn(role="u", content="We visited the lighthouse in 2019.", timestamp=ts_for(2019, 7, 1))],
        )
    )
    store.add(
        AddPayload(
            request_id="b",
            user_id="u",
            session_id="s",
            messages=[MessageIn(role="u", content="We visited the lighthouse in 2023.", timestamp=ts_for(2023, 7, 1))],
        )
    )
    cfg = tmp_config()
    engine = RetrievalEngine(cfg)
    index = engine.build_index(store.get_user_index("u"))
    out = engine.search(index, "Which lighthouse visit happened in 2023?", 10)
    assert out.trace["temporal_boosted"] >= 1
    assert "2023" in out.items[0].content  # year anchor pulls the right unit up
    store.close()
