"""G23/G24/G25 — /health behavior-config echo + runner mismatch gating.

Integrity rule: retrieval behavior is decided by the SERVER's effective config
(serve.py --set ...); the ledger label comes from the CLIENT's config
(run_bench.py --set ...). The runner must refuse to run — and must write
nothing at all — when the two behavior-relevant fingerprints disagree.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_bench"))

from aml_config import load_config, merge_set_overrides  # noqa: E402
from benchlib.runner import ConfigMismatchError, run_benchmark  # noqa: E402

import _bootstrap  # noqa: E402,F401


def _mini_conversation() -> object:
    # kept local so this file stands alone (same shape as test_ledger_and_bench)
    from benchlib.locomo_loader import Conversation, LocomoQuestion, LocomoTurn

    t = 1683000000000
    turns1 = [
        LocomoTurn(dia_id="D1:1", speaker="alice", text="I adopted a dog named Barkley in May 2023.", ts_ms=t),
        LocomoTurn(dia_id="D1:2", speaker="bob", text="Wonderful! How old is Barkley?", ts_ms=t + 60_000),
        LocomoTurn(dia_id="D1:3", speaker="alice", text="Barkley is four years old.", ts_ms=t + 120_000),
    ]
    turns2 = [
        LocomoTurn(dia_id="D2:1", speaker="alice", text="I started piano lessons last month.", ts_ms=t + 86_400_000),
    ]
    questions = [
        LocomoQuestion(question="What is the name of the dog?", answer="Barkley", evidence=("D1:1",), category=1),
        LocomoQuestion(question="How old is the dog?", answer="four years old", evidence=("D1:3",), category=1),
        LocomoQuestion(question="When was the dog adopted?", answer="May 2023", evidence=("D1:1",), category=3),
    ]
    return Conversation(sample_id="mini-1", sessions=[turns1, turns2], questions=questions)


# ---------------------------------------------------------------------------
# G23 — /health echoes the server's behavior-relevant config


def test_g23_health_echoes_server_behavior_config(client_factory, tmp_config):
    overrides = {"retrievers": {"dense": False}, "surface": {"RETURN_SIZE": 20, "NEIGHBOR_POLICY": "window:1"}}
    c = client_factory(overrides)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["retrievers_on"] == ["exact", "bm25"]
    assert body["RETURN_SIZE"] == 20
    assert body["ORDERING_POLICY"] == "relevance_desc"
    assert body["NEIGHBOR_POLICY"] == "window:1"
    assert body["dense_backend"] == "hashed_ngram"
    assert len(body["config_fingerprint"]) == 64
    # the echo equals the fingerprint computed over the same effective config
    # the fixture built (infra paths injected by the fixture must not matter)
    assert body["config_fingerprint"] == tmp_config(overrides).behavior_fingerprint()


def test_g23_behavior_fingerprint_ignores_infra_and_tracks_behavior():
    base = load_config()
    moved_infra = load_config(
        overrides={
            "storage": {"db_path": "/tmp/anything.db"},
            "server": {"port": 9999},
            "ops_log": {"dir": "/tmp/ops"},
        }
    )
    assert base.behavior_fingerprint() == moved_infra.behavior_fingerprint()
    comment_only = load_config(overrides={"_comment": "prose, not configuration"})
    assert base.behavior_fingerprint() == comment_only.behavior_fingerprint()
    toggled_retriever = load_config(overrides={"retrievers": {"dense": False}})
    assert base.behavior_fingerprint() != toggled_retriever.behavior_fingerprint()
    toggled_surface = load_config(overrides={"surface": {"RETURN_SIZE": 5}})
    assert base.behavior_fingerprint() != toggled_surface.behavior_fingerprint()
    toggled_backend = load_config(overrides={"dense": {"backend": "hashed_ngram"}})
    assert base.behavior_fingerprint() != toggled_backend.behavior_fingerprint()
    toggled_device = load_config(overrides={"dense": {"device": "cpu"}})
    assert base.behavior_fingerprint() != toggled_device.behavior_fingerprint()
    toggled_fusion = load_config(overrides={"fusion": {"method": "rrf"}})  # default is norm_sum
    assert base.behavior_fingerprint() != toggled_fusion.behavior_fingerprint()


def test_g23_set_overrides_deep_merge():
    """Repeated --set under one top-level key must deep-merge (B1 combos)."""
    merged = merge_set_overrides(["retrievers.exact=true", "retrievers.bm25=false", "retrievers.dense=false"])
    cfg = load_config(overrides=merged)
    assert cfg.retrievers.on() == ["exact"]  # exact pinned: the V0.5 default is off
    merged = merge_set_overrides(["surface.RETURN_SIZE=10", "surface.ORDERING_POLICY=temporal_asc"])
    cfg = load_config(overrides=merged)
    assert cfg.surface.RETURN_SIZE == 10
    assert cfg.surface.ORDERING_POLICY == "temporal_asc"


# ---------------------------------------------------------------------------
# G24 — mismatch aborts the runner before any write


def test_g24_runner_aborts_on_mismatch_and_writes_nothing(tmp_config, tmp_path):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    server_cfg = tmp_config({"retrievers": {"dense": False}})  # server runs dense=off …
    client_cfg = tmp_config({})  # … client labels the default (dense on)
    results = tmp_path / "results"

    app = create_app(server_cfg)
    with TestClient(app) as client:
        with pytest.raises(ConfigMismatchError) as excinfo:
            run_benchmark(
                client,
                [_mini_conversation()],
                client_cfg,
                run_id="g24-mismatch",
                results_root=results,
                label="g24",
            )
    msg = str(excinfo.value)
    assert "CONFIG MISMATCH" in msg
    assert "retrievers_on" in msg and "dense_backend" in msg
    assert not (results / "ledger" / "ledger.jsonl").exists()
    assert not (results / "ledger" / "ledger.csv").exists()
    assert not (results / "runs" / "g24-mismatch").exists()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_healthy(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001 — connection refused until up
            last_error = exc
        time.sleep(0.3)
    raise RuntimeError(f"service did not become healthy: {last_error}")


@pytest.mark.skipif(
    not (REPO / "04_bench" / "data" / "locomo10.json").exists(), reason="LOCOMO data not fetched"
)
def test_g24_g25_subprocess_server_mismatch_and_match(tmp_path):
    """End-to-end: real serve.py + real run_bench.py.

    mismatch → exit 3, zero ledger lines; same --set on both sides → one new
    ledger line whose SERVER_CONFIG equals CONFIG (G25).
    """
    results = tmp_path / "results"
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    db_path = db_dir / "arm.db"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "serve.py"),
            "--port",
            str(port),
            "--set",
            f"storage.db_path={db_path}",
            "--set",
            "retrievers.dense=false",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_healthy(port)

        def bench(*extra: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [
                    sys.executable,
                    str(REPO / "04_bench" / "run_bench.py"),
                    "--base-url",
                    base_url,
                    "--results-root",
                    str(results),
                    "--run-id",
                    "g24-sub",
                    *extra,
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=600,
            )

        # mismatch: server runs dense=off, client labels default (dense on)
        mismatch = bench()
        assert mismatch.returncode == 3, mismatch.stdout + mismatch.stderr
        assert "CONFIG MISMATCH" in mismatch.stderr
        assert not (results / "ledger" / "ledger.jsonl").exists()

        # match: same --set on both sides
        match = bench("--label", "g25-match", "--set", "retrievers.dense=false")
        assert match.returncode == 0, match.stdout + match.stderr
        ledger_path = results / "ledger" / "ledger.jsonl"
        lines = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        record = lines[0]
        assert record["SERVER_CONFIG"] == record["CONFIG"]
        assert record["RETRIEVERS_ON"] == "bm25"  # V0.5 defaults: exact off, dense off via --set
        assert record["FAILURE_CASES"] == 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


# ---------------------------------------------------------------------------
# G25 — every new ledger record carries SERVER_CONFIG, equal under normal runs


def test_g25_normal_run_records_equal_config_fingerprints(tmp_config, tmp_path):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    cfg = tmp_config()
    results = tmp_path / "results"
    app = create_app(cfg)
    with TestClient(app) as client:
        record = run_benchmark(
            client,
            [_mini_conversation()],
            cfg,
            run_id="g25-normal",
            results_root=results,
            label="g25",
        )
    assert record["SERVER_CONFIG"] == record["CONFIG"]
    lines = [json.loads(line) for line in (results / "ledger" / "ledger.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["SERVER_CONFIG"] == lines[0]["CONFIG"]
