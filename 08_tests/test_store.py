"""Store-level gates: G3 (full), persistence, provenance, isolation internals."""

from __future__ import annotations

import json

from aml_store import AddPayload, CompetitionMemoryStore, MessageIn


def _payload(**kw):
    base = {
        "request_id": "r",
        "user_id": "u",
        "session_id": "s",
        "messages": [MessageIn(role="user", content="hello world")],
    }
    base.update(kw)
    return AddPayload(**base)


def test_g3_dedupe_key_semantics(tmp_path):
    store = CompetitionMemoryStore(tmp_path / "s.db")
    first = store.add(_payload(user_id="u", session_id="s", request_id="r"))
    assert len(first.stored) == 1
    same = store.add(_payload(user_id="u", session_id="s", request_id="r"))
    assert same.stored == [] and len(same.skipped_duplicate) == 1  # idempotent no-op
    other_user = store.add(_payload(user_id="v", session_id="s", request_id="r"))
    assert len(other_user.stored) == 1  # different user: independent scope
    other_msg = store.add(
        _payload(user_id="u", session_id="s", request_id="r", messages=[MessageIn(role="user", content="x")])
    )
    assert len(other_msg.stored) == 1  # different content: appended
    store.close()


def test_ids_are_per_user_sequential_and_stable(tmp_path):
    store = CompetitionMemoryStore(tmp_path / "s.db")
    a1 = store.add(_payload(user_id="a", messages=[MessageIn(role="u", content="1")]))
    b1 = store.add(_payload(user_id="b", messages=[MessageIn(role="u", content="2")]))
    a2 = store.add(_payload(user_id="a", request_id="r2", messages=[MessageIn(role="u", content="3")]))
    assert [u.id for u in a1.stored] == ["mem-000001"]
    assert [u.id for u in b1.stored] == ["mem-000001"]  # per-user namespace
    assert [u.id for u in a2.stored] == ["mem-000002"]
    idx = store.get_user_index("a")
    assert [u.id for u in idx.units] == ["mem-000001", "mem-000002"]
    store.close()


def test_provenance_recorded(tmp_path):
    store = CompetitionMemoryStore(tmp_path / "s.db")
    out = store.add(
        _payload(
            messages=[MessageIn(role="speaker-x", content="provenance probe", timestamp=1700000000000)]
        )
    )
    unit = out.stored[0]
    prov = json.loads(unit.provenance_json())
    assert prov["request_id"] == "r"
    assert prov["session_id"] == "s"
    assert prov["msg_index"] == 0
    assert prov["role"] == "speaker-x"
    assert prov["arrival_seq"] == 1
    assert len(prov["content_sha256"]) == 64
    assert unit.created_at == "2023-11-14T22:13:20Z"
    # round-trip from DB
    got = store.unit("u", unit.id)
    assert got is not None and got.content_sha256 == unit.content_sha256
    store.close()


def test_session_seq_orders_arrival_within_session(tmp_path):
    store = CompetitionMemoryStore(tmp_path / "s.db")
    msgs = [MessageIn(role="u", content=f"m{i}") for i in range(5)]
    out = store.add(_payload(messages=msgs))
    seqs = [u.session_seq for u in out.stored]
    assert seqs == [1, 2, 3, 4, 5]
    out2 = store.add(_payload(request_id="r2", messages=[MessageIn(role="u", content="m9")]))
    assert out2.stored[0].session_seq == 6  # continues the same (user, session)
    store.close()


def test_isolation_no_cross_user_reads(tmp_path):
    store = CompetitionMemoryStore(tmp_path / "s.db")
    store.add(_payload(user_id="alice", messages=[MessageIn(role="u", content="secret alice")]))
    store.add(_payload(user_id="bob", messages=[MessageIn(role="u", content="secret bob")]))
    alice_index = store.get_user_index("alice")
    assert all("alice" in u.content for u in alice_index.units)
    bob_index = store.get_user_index("bob")
    assert all("bob" in u.content for u in bob_index.units)
    # ids are per-user scoped: the same id exists in both namespaces but reads
    # never cross the user boundary
    same_id = alice_index.units[0].id
    assert bob_index.units[0].id == same_id
    got = store.unit("bob", same_id)
    assert got is not None and got.user_id == "bob" and "bob" in got.content
    assert store.unit("carol", same_id) is None
    store.close()
