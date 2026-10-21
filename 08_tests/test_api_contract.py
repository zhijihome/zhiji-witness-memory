"""Contract gates exercised over the live HTTP surface.

Covers G1, G2, G13, G14, G15, G16, G17, G18, G19, G20 (G3 partly here, fully
in test_store.py; G4 partly here).
"""

from __future__ import annotations


def test_g1_add_then_immediate_search_hit(seeded_client):
    resp = seeded_client.post(
        "/add",
        json={
            "request_id": "req-g1",
            "messages": [{"role": "alice", "content": "Unique marker: zebra quantum flute."}],
            "user_id": "alice",
            "session_id": "sess-3",
        },
    )
    assert resp.status_code == 200
    out = seeded_client.post("/search", json={"query": "zebra quantum flute", "user_id": "alice", "top_k": 10})
    assert out.status_code == 200
    data = out.json()["data"]
    assert any("zebra quantum flute" in item["content"] for item in data)


def test_g1_immediacy_across_service_restart(tmp_config, tmp_path):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    cfg = tmp_config()
    app1 = create_app(cfg)
    with TestClient(app1) as c1:
        r = c1.post(
            "/add",
            json={
                "request_id": "rr",
                "messages": [{"role": "u", "content": "Persisted durable fact about kayaks."}],
                "user_id": "u-persist",
                "session_id": "s",
            },
        )
        assert r.status_code == 200
    # fresh process over the same database file
    app2 = create_app(tmp_config())
    with TestClient(app2) as c2:
        out = c2.post("/search", json={"query": "kayaks", "user_id": "u-persist", "top_k": 10})
        assert out.status_code == 200
        assert any("kayaks" in i["content"] for i in out.json()["data"])


def test_g2_user_isolation(seeded_client):
    out = seeded_client.post("/search", json={"query": "dog", "user_id": "bob", "top_k": 100})
    data = out.json()["data"]
    assert all("Barkley" not in i["content"] for i in data)
    assert any("Rex" in i["content"] for i in data)
    # and the reverse: alice never sees bob's data
    out = seeded_client.post("/search", json={"query": "dog named", "user_id": "alice", "top_k": 100})
    assert all("Rex" not in i["content"] for i in out.json()["data"])
    # unknown user: empty, not error, not null
    out = seeded_client.post("/search", json={"query": "dog", "user_id": "nobody", "top_k": 100})
    assert out.status_code == 200
    assert out.json()["data"] == []


def test_g3_duplicate_add_idempotent_and_documented(seeded_client):
    payload = {
        "request_id": "req-dup",
        "messages": [{"role": "alice", "content": "Duplicate delivery probe wobble-clock."}],
        "user_id": "alice",
        "session_id": "sess-dup",
    }
    first = seeded_client.post("/add", json=payload)
    assert first.status_code == 200 and first.json()["success"] is True
    second = seeded_client.post("/add", json=payload)  # identical re-delivery
    assert second.status_code == 200 and second.json()["success"] is True
    out = seeded_client.post(
        "/search", json={"query": "wobble-clock", "user_id": "alice", "top_k": 100}
    )
    contents = [i["content"] for i in out.json()["data"]]
    assert contents.count("Duplicate delivery probe wobble-clock.") == 1  # no double store

    # different content under the same request_id → appended (documented semantics)
    payload2 = dict(payload, messages=[{"role": "alice", "content": "Different follow-up note parsnip."}])
    third = seeded_client.post("/add", json=payload2)
    assert third.status_code == 200
    out = seeded_client.post("/search", json={"query": "parsnip", "user_id": "alice", "top_k": 100})
    assert any("parsnip" in i["content"] for i in out.json()["data"])


def test_g4_search_returns_evidence_only(seeded_client):
    out = seeded_client.post("/search", json={"query": "dog named", "user_id": "alice", "top_k": 100})
    body = out.json()
    assert set(body.keys()) == {"data"}  # no answer field anywhere in the response
    for item in body["data"]:
        assert set(item.keys()) == {"id", "content", "score", "created_at"}
        assert item["id"] and item["content"]


def test_g13_add_response_shape_and_exact_echo(seeded_client):
    resp = seeded_client.post(
        "/add",
        json={
            "request_id": "echo-1",
            "messages": [{"role": "alice", "content": "echo probe"}],
            "user_id": "alice",
            "session_id": "sess-echo",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["request_id"] == "echo-1"
    assert body["user_id"] == "alice"
    assert body["session_id"] == "sess-echo"
    assert set(body.keys()) == {"success", "request_id", "user_id", "session_id"}


def test_g14_never_202_or_task_ids(seeded_client):
    resp = seeded_client.post(
        "/add",
        json={
            "request_id": "r14",
            "messages": [{"role": "a", "content": "c"}] * 50,
            "user_id": "alice",
            "session_id": "s14",
        },
    )
    # Tightened in V0.5 (Judge O3): the old assertion had an always-true escape
    # (`banned not in text or '"success"' in text`). No escape branches now.
    assert resp.status_code == 200  # synchronous 200, never an accepted-for-processing code
    body = resp.json()
    assert set(body.keys()) == {"success", "request_id", "user_id", "session_id"}
    assert body["success"] is True
    assert body["request_id"] == "r14" and body["user_id"] == "alice" and body["session_id"] == "s14"
    text = resp.text.lower()
    for banned in ("task_id", "taskid", "poll", "status_url", "queued", "202"):
        assert banned not in text


def test_g15_empty_result_is_empty_array(seeded_client):
    out = seeded_client.post(
        "/search", json={"query": "xylophone neon glacier", "user_id": "alice", "top_k": 100}
    )
    assert out.status_code == 200
    assert out.json() == {"data": []}


def test_g16_relevance_desc_and_top_k_cap(client_factory):
    c = client_factory()
    msgs = [{"role": "u", "content": f"gamma unit number {i} about llamas"} for i in range(30)]
    c.post(
        "/add",
        json={"request_id": "bulk", "messages": msgs, "user_id": "bulk-user", "session_id": "s"},
    )
    out = c.post("/search", json={"query": "gamma llamas", "user_id": "bulk-user", "top_k": 100})
    data = out.json()["data"]
    assert 0 < len(data) <= 100
    scores = [i["score"] for i in data]
    assert scores == sorted(scores, reverse=True)  # relevance descending
    out5 = c.post("/search", json={"query": "gamma llamas", "user_id": "bulk-user", "top_k": 5})
    assert len(out5.json()["data"]) <= 5


def test_g17_session_id_not_a_search_filter(seeded_client):
    # alice's evidence spans sess-1 and sess-2; a single search must see both
    out = seeded_client.post("/search", json={"query": "dog piano", "user_id": "alice", "top_k": 100})
    contents = " ".join(i["content"] for i in out.json()["data"])
    assert "Barkley" in contents
    assert "piano" in contents


def test_g18_platform_never_sends_these_fields(seeded_client):
    payload = {
        "request_id": "r18",
        "messages": [{"role": "alice", "content": "field tolerance probe thistle."}],
        "user_id": "alice",
        "session_id": "sess-18",
        "metadata": {"x": 1},
        "app_id": "app",
        "agent_id": "agent",
        "async_mode": True,
    }
    resp = seeded_client.post("/add", json=payload)
    assert resp.status_code == 200
    out = seeded_client.post("/search", json={"query": "thistle", "user_id": "alice", "top_k": 10})
    assert any("thistle" in i["content"] for i in out.json()["data"])


def test_g19_error_code_semantics(seeded_client):
    # missing required field → 422 (contract format, no retry)
    r = seeded_client.post("/search", json={"query": "x", "user_id": "alice"})
    assert r.status_code == 422
    # wrong type → 422
    r = seeded_client.post("/search", json={"query": "x", "user_id": "alice", "top_k": "many"})
    assert r.status_code == 422
    # empty required string → 422
    r = seeded_client.post("/add", json={"request_id": "", "messages": [], "user_id": "", "session_id": ""})
    assert r.status_code == 422
    # unknown path → 404
    r = seeded_client.post("/nope", json={})
    assert r.status_code == 404
    # malformed JSON → 400-class (both are no-retry per Contract §7)
    r = seeded_client.post("/add", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code in (400, 422)


def test_g19_auth_401_when_configured(tmp_config, monkeypatch):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AML_API_KEY", "sekrit")
    app = create_app(tmp_config())
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # health stays unauthenticated (G20)
        r = c.post(
            "/add",
            json={"request_id": "a", "messages": [{"role": "r", "content": "c"}], "user_id": "u", "session_id": "s"},
        )
        assert r.status_code == 401  # no-retry class
        r = c.post(
            "/add",
            json={"request_id": "a", "messages": [{"role": "r", "content": "c"}], "user_id": "u", "session_id": "s"},
            headers={"Authorization": "Bearer sekrit"},
        )
        assert r.status_code == 200
        r = c.post(
            "/add",
            json={"request_id": "a2", "messages": [{"role": "r", "content": "c"}], "user_id": "u", "session_id": "s"},
            headers={"Authorization": "Token sekrit"},
        )
        assert r.status_code == 200  # AML apply-form default scheme
        r = c.post(
            "/search", json={"query": "c", "user_id": "u", "top_k": 5}, headers={"X-Api-Key": "sekrit"}
        )
        assert r.status_code == 200


def test_g20_health_unauthenticated_2xx(seeded_client):
    assert seeded_client.get("/health").status_code == 200
    assert seeded_client.get("/health").json()["status"] == "ok"


def test_g9_ops_journal_records_every_call(tmp_path, client_factory):
    import json

    c = client_factory()
    c.post(
        "/add",
        json={
            "request_id": "ops1",
            "messages": [{"role": "u", "content": "ops journal probe dandelion."}],
            "user_id": "ops-user",
            "session_id": "s",
        },
    )
    c.post("/search", json={"query": "dandelion", "user_id": "ops-user", "top_k": 10})
    lines = [
        json.loads(line)
        for line in (tmp_path / "ops" / "ops.jsonl").read_text().splitlines()
        if line.strip()
    ]
    adds = [rec for rec in lines if rec["op"] == "add" and rec["status"] == 200]
    searches = [rec for rec in lines if rec["op"] == "search" and rec["status"] == 200]
    assert adds and searches
    for rec in adds + searches:
        for field in ("latency_ms", "model_calls", "tokens", "cost_usd"):
            assert field in rec
    for rec in searches:
        for knob in ("RETURN_SIZE", "ORDERING_POLICY", "NEIGHBOR_POLICY"):
            assert knob in rec
    assert all(rec["model_calls"] == 0 for rec in searches + adds)  # NO_LLM: real zeros
