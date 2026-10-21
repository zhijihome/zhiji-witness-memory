"""G22 — the three return-surface knobs are config-driven and actually move
the returned surface; every ledger record carries them (ledger side tested in
test_ledger.py)."""

from __future__ import annotations

from aml_store import AddPayload, MessageIn


def _bulk(client, n=25):
    msgs = [{"role": "u", "content": f"kappa widget number {i} in the parade"} for i in range(n)]
    client.post(
        "/add",
        json={"request_id": "bulk", "messages": msgs, "user_id": "knob-user", "session_id": "s"},
    )


def test_return_size_changes_surface(client_factory):
    c1 = client_factory()  # default RETURN_SIZE=top_k
    _bulk(c1)
    out = c1.post("/search", json={"query": "kappa widget parade", "user_id": "knob-user", "top_k": 100})
    default_n = len(out.json()["data"])
    assert default_n > 5

    c2 = client_factory({"surface": {"RETURN_SIZE": 5}})
    _bulk(c2)
    out = c2.post("/search", json={"query": "kappa widget parade", "user_id": "knob-user", "top_k": 100})
    assert len(out.json()["data"]) == 5  # knob applied, still ≤ top_k

    c3 = client_factory({"surface": {"RETURN_SIZE": 3}})
    _bulk(c3)
    out = c3.post("/search", json={"query": "kappa widget parade", "user_id": "knob-user", "top_k": 2})
    assert len(out.json()["data"]) == 2  # never exceeds top_k even if RETURN_SIZE bigger


def test_ordering_policy_changes_surface(client_factory):
    stamps = [1683000000000, 1672000000000, 1694000000000]  # deliberately non-monotonic
    for overrides, expect_temporal in (
        (None, False),
        ({"surface": {"ORDERING_POLICY": "temporal_asc"}}, True),
    ):
        c = client_factory(overrides)
        for i, ts in enumerate(stamps):
            c.post(
                "/add",
                json={
                    "request_id": f"o{i}",
                    "messages": [{"role": "u", "timestamp": ts, "content": f"ordering omega token {i}"}],
                    "user_id": "ord-user",
                    "session_id": f"s{i}",
                },
            )
        out = c.post("/search", json={"query": "ordering omega token", "user_id": "ord-user", "top_k": 100})
        data = out.json()["data"]
        assert len(data) == 3
        if expect_temporal:
            created = [d["created_at"] for d in data]
            assert created == sorted(created)  # oldest first
        else:
            scores = [d["score"] for d in data]
            assert scores == sorted(scores, reverse=True)


def test_neighbor_policy_changes_surface(client_factory):
    msgs = [
        {"role": "u", "timestamp": 1683000000000, "content": "sigma preamble smalltalk"},
        {"role": "u", "timestamp": 1683000600000, "content": "sigma core fact the ferry leaves at noon"},
        {"role": "u", "timestamp": 1683001200000, "content": "sigma postamble smalltalk again"},
    ]
    c1 = client_factory()  # NEIGHBOR_POLICY off
    c1.post("/add", json={"request_id": "n", "messages": msgs, "user_id": "nb-user", "session_id": "s"})
    out = c1.post("/search", json={"query": "ferry leaves noon", "user_id": "nb-user", "top_k": 100})
    contents = [i["content"] for i in out.json()["data"]]
    assert any("ferry" in x for x in contents)
    assert not any("preamble" in x for x in contents)  # no expansion when off

    c2 = client_factory({"surface": {"NEIGHBOR_POLICY": "window:1"}})
    c2.post("/add", json={"request_id": "n", "messages": msgs, "user_id": "nb-user", "session_id": "s"})
    out = c2.post("/search", json={"query": "ferry leaves noon", "user_id": "nb-user", "top_k": 100})
    contents = [i["content"] for i in out.json()["data"]]
    assert any("preamble" in x for x in contents)  # adjacent context now present


def test_knob_values_appear_in_search_trace(tmp_config):
    from aml_api.service import MemoryService

    svc = MemoryService(
        tmp_config(
            {"surface": {"RETURN_SIZE": 7, "ORDERING_POLICY": "temporal_asc", "NEIGHBOR_POLICY": "window:2"}}
        )
    )
    svc.add(
        AddPayload(
            request_id="t",
            user_id="u",
            session_id="s",
            messages=[MessageIn(role="a", content="trace probe")],
        )
    )
    out = svc.search("u", "trace probe", 100)
    assert out.trace["return_size_effective"] == 7
    assert out.trace["ordering_policy"] == "temporal_asc"
    assert out.trace["neighbor_window"] == 2
    svc.close()
