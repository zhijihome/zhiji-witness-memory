"""Shared fixtures for the V0 test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "04_bench"))

from aml_config import load_config  # noqa: E402

import _bootstrap  # noqa: E402,F401


@pytest.fixture()
def tmp_config(tmp_path):
    def _make(overrides: dict | None = None):
        base = {
            "storage": {"db_path": str(tmp_path / "test.db")},
            "ops_log": {"enabled": True, "dir": str(tmp_path / "ops")},
            # Unit suite stays hermetic and pinned to the V0 retrieval world:
            # hashed dense (no model downloads) + exact on + neighbor off.
            # The shipped V0.5 defaults (sentence_transformers, norm_sum/minmax
            # fusion, exact off, window:1) have their own tests
            # (test_v05_hardening.py) and a full-bench confirmation row.
            "dense": {"backend": "hashed_ngram"},
            "retrievers": {"exact": True},
            "surface": {"NEIGHBOR_POLICY": "off"},
        }
        merged = _deep(base, overrides or {})
        return load_config(overrides=merged)

    return _make


def _deep(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep(out[k], v)
        else:
            out[k] = v
    return out


@pytest.fixture()
def client_factory(tmp_config):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    def _make(overrides: dict | None = None):
        app = create_app(tmp_config(overrides))
        return TestClient(app)

    return _make


@pytest.fixture()
def seeded_client(client_factory):
    """Client with a known corpus for user 'alice' across two sessions."""

    c = client_factory()
    c.post(
        "/add",
        json={
            "request_id": "req-a1",
            "messages": [
                {"role": "alice", "timestamp": 1683000000000, "content": "I adopted a dog named Barkley."},
                {"role": "bob", "timestamp": 1683000060000, "content": "Congratulations! How old is he?"},
                {"role": "alice", "timestamp": 1683000120000, "content": "Barkley is four years old."},
            ],
            "user_id": "alice",
            "session_id": "sess-1",
        },
    )
    c.post(
        "/add",
        json={
            "request_id": "req-a2",
            "messages": [
                {"role": "alice", "timestamp": 1684000000000, "content": "I started learning piano on 12 May 2023."},
            ],
            "user_id": "alice",
            "session_id": "sess-2",
        },
    )
    c.post(
        "/add",
        json={
            "request_id": "req-b1",
            "messages": [
                {"role": "carol", "content": "My dog is named Rex and hates tennis balls."},
            ],
            "user_id": "bob",
            "session_id": "sess-9",
        },
    )
    return c
