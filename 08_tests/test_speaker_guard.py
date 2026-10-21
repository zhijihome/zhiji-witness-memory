"""Speaker-label guard + display-prefix gates (candidate A1-DS headline feature).

The official contract fixes message.role to generic values (user/assistant) —
a generic role is a protocol position, not a participant identity, and must
never surface as a speaker label on ANY path (display, index, derived facts).
Real names (bench harness sends role="Caroline") still label.
"""

from __future__ import annotations

from aml_store.model import MemoryUnit, speaker_label
from aml_retrieval.engine import _index_text


def test_speaker_label_unit_semantics():
    # real participant names keep their label
    assert speaker_label("Caroline") == "Caroline"
    assert speaker_label("Jon") == "Jon"
    assert speaker_label("  Alice  ") == "Alice"  # whitespace tolerated
    # every generic chat role is suppressed (case-insensitive)
    for role in ("user", "assistant", "system", "tool", "function", "developer",
                 "User", "ASSISTANT", " System "):
        assert speaker_label(role) is None, role
    # absent roles carry no label
    assert speaker_label(None) is None
    assert speaker_label("") is None


def _unit(role: str | None, content: str) -> MemoryUnit:
    return MemoryUnit(
        id="u1", user_id="u", session_id="s", request_id="r", msg_index=0,
        role=role or "", content=content, ts_ms=None, created_at="",
        arrival_seq=0, session_seq=0, content_sha256="",
    )


def test_index_text_matches_display_semantics():
    # index text and display prefix must agree: name binds, generic drops
    assert _index_text(_unit("Caroline", "went hiking")) == "Caroline: went hiking"
    assert _index_text(_unit("user", "went hiking")) == "went hiking"
    assert _index_text(_unit("assistant", "sure thing")) == "sure thing"
    assert _index_text(_unit(None, "no role")) == "no role"


def test_derived_fact_text_guard():
    from aml_atomic.pipeline import _fact_text

    assert _fact_text("Caroline", "I adopted a dog") == "Caroline: I adopted a dog"
    assert _fact_text("user", "I adopted a dog") == "I adopted a dog"


def test_search_display_prefix_end_to_end(client_factory):
    cfg = {"search": {"date_prefix": False, "speaker_prefix": True}}
    with client_factory(cfg) as c:
        c.post("/add", json={
            "request_id": "r1", "user_id": "u", "session_id": "s",
            "messages": [
                {"role": "Caroline", "content": "I climbed the ridge yesterday"},
                {"role": "user", "content": "the ridge was icy"},
            ],
        })
        data = c.post("/search", json={"query": "ridge", "user_id": "u", "top_k": 10}).json()["data"]
    texts = [i["content"] for i in data]
    assert any(t.startswith("Caroline: ") for t in texts), texts
    assert all(not t.startswith("user: ") and not t.startswith("assistant: ") for t in texts)
    assert any("the ridge was icy" in t for t in texts)


def test_display_prefix_off_when_disabled(client_factory):
    cfg = {"search": {"date_prefix": False, "speaker_prefix": False}}
    with client_factory(cfg) as c:
        c.post("/add", json={
            "request_id": "r2", "user_id": "u2", "session_id": "s",
            "messages": [{"role": "Caroline", "content": "plain fact about pianos"}],
        })
        data = c.post("/search", json={"query": "pianos", "user_id": "u2", "top_k": 10}).json()["data"]
    assert any(i["content"] == "plain fact about pianos" for i in data)
