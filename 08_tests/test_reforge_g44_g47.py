"""Reforge R1 gates — G44 (official endpoint hard gate), G45 (endpoint-bound
cache identity + migration), G46 (LLM provenance in the ledger), G47 (probe
block semantics), plus the two R1/R3 invariants: the restored authority
workorder stays byte-identical and the historical ledger stays append-only.

Unit/hermetic: the LLM is always a fake or a mocked transport; the only live
artifact checks are byte/hash comparisons against git history.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_bench"))

from aml_adapter import (  # noqa: E402
    AMLModelAdapter,
    OfficialEndpointViolationError,
)
from aml_atomic import ExtractionCache, cache_key  # noqa: E402
from aml_atomic.cache import cache_fingerprint, fingerprint_of_db  # noqa: E402
from aml_atomic.extractors import prompt_version  # noqa: E402
from aml_config import load_config  # noqa: E402
from aml_config.config import (  # noqa: E402
    OFFICIAL_ENDPOINT_HOST,
    ConfigError,
    canonical_endpoint_identity,
    endpoint_allowed_for_coordinate,
)
from aml_store.model import content_sha256  # noqa: E402
from benchlib.ledger import (  # noqa: E402
    PROV_KEY_LEAK,
    PROV_LEGACY,
    PROV_MALFORMED,
    PROV_NULL,
    PROV_OK,
    audit_ledger_key_hygiene,
    audit_ledger_llm_provenance,
    llm_provenance_relation,
)

import _bootstrap  # noqa: E402,F401

EVIDENCE_COMMIT = "a072910"  # judge report evidence baseline (R1 append-only anchor)
AUTHORITY_COMMIT = "0371cfb"  # historical location of the restored workorder (R3)


# ---------------------------------------------------------------------------
# G44 — official coordinate forces the official endpoint


def test_g44_identity_is_normalized_hostname_only():
    assert canonical_endpoint_identity("https://api.openai.com/v1") == "api.openai.com"
    assert canonical_endpoint_identity("https://API.OpenAI.COM:443/v1?x=1#frag") == "api.openai.com"
    # userinfo, port, path, query and fragments never participate — least of
    # all any credential material appended to the URL
    assert canonical_endpoint_identity("https://user:sk-secret@relay.example.com/v1?key=sk-x") == (
        "relay.example.com"
    )
    assert "sk-" not in canonical_endpoint_identity("https://relay.example.com/v1?api_key=sk-abc")
    with pytest.raises(ValueError):
        canonical_endpoint_identity("not a url")


def test_g44_official_coordinate_with_official_endpoint_allowed(monkeypatch):
    monkeypatch.setenv("AML_OFFICIAL_MODEL_API_KEY", "test-key")
    adapter = AMLModelAdapter(
        tier="AML_OFFICIAL_MODEL",
        coordinate="OFFICIAL_SMOKE",
        official_base_url="https://api.openai.com/v1",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert adapter.complete("ping").text == "ok"  # OFFICIAL + official → allow


def test_g44_official_coordinate_with_relay_endpoint_rejected_before_call():
    with pytest.raises(OfficialEndpointViolationError):
        AMLModelAdapter(
            tier="AML_OFFICIAL_MODEL",
            coordinate="OFFICIAL_FULL",
            official_base_url="https://api.relay.test/v1",
        )


def test_g44_relay_swap_after_construction_still_rejected_before_call(monkeypatch):
    # defense in depth: even if the dataclass is mutated after a clean
    # construction, the gate fires again immediately BEFORE the model call
    monkeypatch.setenv("AML_OFFICIAL_MODEL_API_KEY", "test-key")
    adapter = AMLModelAdapter(
        tier="AML_OFFICIAL_MODEL",
        coordinate="OFFICIAL_SMOKE",
        official_base_url="https://api.openai.com/v1",
    )
    calls = {"n": 0}
    adapter._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))  # pragma: no cover
    )
    real_post = adapter._client.post

    def counting_post(*a, **kw):
        calls["n"] += 1
        return real_post(*a, **kw)

    adapter._client.post = counting_post
    adapter.official_base_url = "https://api.relay.test/v1"
    with pytest.raises(OfficialEndpointViolationError):
        adapter.complete("ping")
    assert calls["n"] == 0  # hard fail BEFORE the model call, zero requests


def test_g44_local_coordinate_with_relay_endpoint_allowed():
    adapter = AMLModelAdapter(
        tier="AML_OFFICIAL_MODEL",
        coordinate="LOCAL",
        official_base_url="https://api.relay.test/v1",
    )
    assert adapter.official_base_url.endswith("relay.test/v1")  # LOCAL + relay → allow


@pytest.mark.parametrize("coordinate", ("OFFICIAL_SMOKE", "OFFICIAL_FULL"))
def test_g44_config_layer_hard_fails_on_official_relay(coordinate):
    with pytest.raises(ConfigError, match="official endpoint"):
        load_config(
            overrides={
                "adapter": {
                    "tier": "AML_OFFICIAL_MODEL",
                    "coordinate": coordinate,
                    "official_base_url": "https://api.relay.test/v1",
                }
            }
        )


def test_g44_config_accepts_official_official_and_local_relay():
    ok_official = load_config(
        overrides={
            "adapter": {
                "tier": "AML_OFFICIAL_MODEL",
                "coordinate": "OFFICIAL_SMOKE",
                "official_base_url": f"https://{OFFICIAL_ENDPOINT_HOST}/v1",
            }
        }
    )
    assert ok_official.adapter_cfg["coordinate"] == "OFFICIAL_SMOKE"
    ok_local = load_config(
        overrides={
            "adapter": {
                "tier": "AML_OFFICIAL_MODEL",
                "coordinate": "LOCAL",
                "official_base_url": "https://api.relay.test/v1",
            }
        }
    )
    assert ok_local.adapter_cfg["coordinate"] == "LOCAL"
    with pytest.raises(ConfigError, match="coordinate"):
        load_config(overrides={"adapter": {"coordinate": "OFFICIAL_WHENEVER"}})


def test_g44_service_refuses_to_start_even_if_config_validation_bypassed():
    # the adapter gate is the last line: build an AMLConfig directly (no
    # load_config validation) and hand it to the service factory
    from aml_api import create_app
    from aml_config.config import AMLConfig

    raw = json.loads((REPO / "10_config" / "defaults.json").read_text())
    raw["adapter"].update(
        {
            "tier": "AML_OFFICIAL_MODEL",
            "coordinate": "OFFICIAL_FULL",
            "official_base_url": "https://api.relay.test/v1",
            "spend_state_path": "/tmp/g44-spend.json",
        }
    )
    raw["storage"]["db_path"] = "/tmp/g44.db"
    with pytest.raises(OfficialEndpointViolationError):
        create_app(config=AMLConfig(raw=raw))


def test_g44_endpoint_gate_helper_matrix():
    assert endpoint_allowed_for_coordinate("OFFICIAL_SMOKE", "https://api.openai.com/v1")
    assert not endpoint_allowed_for_coordinate("OFFICIAL_FULL", "https://api.openai.com.evil.com/v1")
    assert not endpoint_allowed_for_coordinate("OFFICIAL_SMOKE", "https://api.relay.test/v1")
    assert endpoint_allowed_for_coordinate("LOCAL", "https://api.relay.test/v1")


# ---------------------------------------------------------------------------
# G45 — endpoint-aware cache identity


def _v1_key(pv: str, model: str, gran: str, ch: str) -> str:
    return hashlib.sha256("\x1f".join((pv, model, gran, ch)).encode("utf-8")).hexdigest()


def test_g45_cache_store_is_endpoint_scoped(tmp_path):
    cache = ExtractionCache(tmp_path / "c" / "extraction_cache.sqlite3")
    ka = cache_key("p", "m", "fine", "ch1", "api.openai.com")
    kb = cache_key("p", "m", "fine", "ch1", "api.relay.test")
    assert ka != kb  # same model/content via different endpoints → different identity
    cache.put(ka, ["official fact"], model_id="m", prompt_version="p", granularity="fine",
              endpoint_identity="api.openai.com", created_at="t")
    assert cache.get(ka) == ["official fact"]
    assert cache.get(kb) is None  # no silent cross-endpoint hit
    rows = cache.rows()
    assert rows[0]["endpoint_identity"] == "api.openai.com"
    cache.close()


def test_g45_pipeline_cache_never_mixes_endpoints(tmp_path):
    from aml_store import CompetitionMemoryStore

    sys.path.insert(0, str(REPO / "08_tests"))
    from test_atomic import FakeLLMAdapter, _atomic_config, _units

    cache_dir = tmp_path / "cache"
    store = CompetitionMemoryStore(tmp_path / "p.db")
    cfg = _atomic_config(tmp_path / "cfg")
    cfg.raw["extraction_cache"]["dir"] = str(cache_dir)
    relay_a = FakeLLMAdapter(official_base_url="https://api.relay.test/v1")
    official = FakeLLMAdapter(official_base_url="https://api.openai.com/v1")
    relay_b = FakeLLMAdapter(official_base_url="https://api.relay.test/v1")

    p1 = _pipeline(cfg, relay_a, store)
    p1.derive(_units(tmp_path))
    assert relay_a.usage["calls"] == 3
    assert p1.endpoint_identity == "api.relay.test"
    p2 = _pipeline(cfg, official, store)
    p2.derive(_units(tmp_path))
    assert official.usage["calls"] == 3  # different endpoint → all misses
    assert p2.endpoint_identity == "api.openai.com"
    p3 = _pipeline(cfg, relay_b, store)
    p3.derive(_units(tmp_path))
    assert relay_b.usage["calls"] == 0  # same endpoint + same inputs → deterministic hits
    store.close()


def _pipeline(cfg, adapter, store):
    from aml_atomic import AtomicPipeline

    return AtomicPipeline(cfg, adapter, store)  # type: ignore[arg-type]


def test_g45_legacy_db_fingerprint_bytes_unchanged(tmp_path):
    # a hand-built v1-schema db must keep its exact pre-v2 fingerprint: the
    # frozen-artifact audit trail must not be disturbed by the schema bump
    db = tmp_path / "legacy" / "extraction_cache.sqlite3"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE extraction_cache (key TEXT PRIMARY KEY, facts_json TEXT NOT NULL,"
        " model_id TEXT NOT NULL, prompt_version TEXT NOT NULL, granularity TEXT NOT NULL,"
        " created_at TEXT NOT NULL)"
    )
    v1_rows = [
        {
            "key": _v1_key("p", "m", "fine", "ch"),
            "facts": ["a fact"],
            "model_id": "m",
            "prompt_version": "p",
            "granularity": "fine",
            "created_at": "t",
        }
    ]
    conn.execute(
        "INSERT INTO extraction_cache VALUES (?,?,?,?,?,?)",
        (v1_rows[0]["key"], json.dumps(["a fact"]), "m", "p", "fine", "t"),
    )
    conn.commit()
    conn.close()
    assert fingerprint_of_db(db) == cache_fingerprint(v1_rows)
    # and a v2 db with the SAME logical content fingerprints differently —
    # the endpoint identity is part of the audited artifact
    v2 = ExtractionCache(tmp_path / "v2" / "extraction_cache.sqlite3")
    v2.put(cache_key("p", "m", "fine", "ch", "api.relay.test"), ["a fact"], model_id="m",
           prompt_version="p", granularity="fine", endpoint_identity="api.relay.test", created_at="t")
    assert v2.fingerprint() != cache_fingerprint(v1_rows)
    v2.close()


def test_g46_no_llm_off_row_has_stable_none_semantics(tmp_path):
    record = _mini_bench_record(tmp_path, {"mode": "off"})
    assert record["LLM_ENDPOINT"] == "none"
    assert record["LLM_MODEL_ID"] == "none"
    assert record["PROMPT_VERSION"] == "none"
    assert record["ATOMIC_PROMPT_VERSION"] == ""


def test_g46_rule_row_has_explicit_provenance(tmp_path):
    from aml_atomic.extractors import RULE_VERSION

    record = _mini_bench_record(tmp_path, {"granularity": "fine"})
    assert record["ATOMIC_MODE"] == "both" and record["ATOMIC_EXTRACTOR"] == "rule"
    assert record["LLM_ENDPOINT"] == "none"
    assert record["LLM_MODEL_ID"] == "none"
    assert record["PROMPT_VERSION"] == f"{RULE_VERSION}-fine"
    assert record["COORDINATE"] == "LOCAL"


def test_g46_provenance_relation_accounting():
    legacy = {"RUN_ID": "old"}
    ok_relay = {"LLM_ENDPOINT": "api.relay.test", "LLM_MODEL_ID": "gpt-4o-mini-2024-07-18", "PROMPT_VERSION": "p"}
    ok_none = {"LLM_ENDPOINT": "none", "LLM_MODEL_ID": "none", "PROMPT_VERSION": "none"}
    null_violation = {"LLM_ENDPOINT": None, "LLM_MODEL_ID": "m", "PROMPT_VERSION": "p"}
    malformed = {"LLM_ENDPOINT": "https://api.relay.test/v1?k=sk-abc", "LLM_MODEL_ID": "m", "PROMPT_VERSION": "p"}
    assert llm_provenance_relation(legacy) == PROV_LEGACY
    assert llm_provenance_relation(ok_relay) == PROV_OK
    assert llm_provenance_relation(ok_none) == PROV_OK
    assert llm_provenance_relation(null_violation) == PROV_NULL
    assert llm_provenance_relation(malformed) == PROV_MALFORMED
    leak = {"LLM_ENDPOINT": "api.openai.com", "LLM_MODEL_ID": "m", "PROMPT_VERSION": "p", # synthetic key material assembled at runtime so no sk- literal lands in
    # source (a tree-wide leak scan must stay 0-hit)
    "LABEL": "sk-" + "ABCDEF1234567890"}
    audit = audit_ledger_llm_provenance([legacy, ok_relay, ok_none, null_violation, malformed, leak])
    assert audit["counts"][PROV_LEGACY] == 1
    assert audit["counts"][PROV_OK] == 2
    assert audit["counts"][PROV_NULL] == 1
    assert audit["counts"][PROV_MALFORMED] == 1
    assert audit["counts"][PROV_KEY_LEAK] == 1
    assert audit_ledger_key_hygiene([legacy, ok_relay]) == []


def test_g46_historical_ledger_rows_unchanged_append_only():
    ledger = REPO / "06_results" / "ledger" / "ledger.jsonl"
    evidence = subprocess.run(
        ["git", "show", f"{EVIDENCE_COMMIT}:06_results/ledger/ledger.jsonl"],
        capture_output=True, text=True, check=True, cwd=REPO,
    ).stdout.splitlines()
    current = ledger.read_text().splitlines()
    assert len(current) >= len(evidence)
    assert current[: len(evidence)] == evidence  # append-only: history byte-identical


def test_g46_ledger_check_cli_accepts_current_ledger():
    proc = subprocess.run(
        [sys.executable, str(REPO / "04_bench" / "ledger_check.py")],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "provenance_legacy_missing" in proc.stdout


# ---------------------------------------------------------------------------
# G47 — probe block semantics


def test_g47_probe_blocked_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("AML_OFFICIAL_MODEL_API_KEY", raising=False)
    out = tmp_path / "probe.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "g47",
            "--limit",
            "5",
            "--out",
            str(out),
            "--dataset",
            str(REPO / "04_bench" / "data" / "locomo10.json"),
            "--spend-state",
            str(tmp_path / "spend" / "spend_state.json"),
        ],
    )
    mod_path = REPO / "06_results" / "g47_consistency_probe.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("g47_probe", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
    report = json.loads(out.read_text())
    assert report["result"] == "BLOCKED_NO_OFFICIAL_KEY"
    assert report["official_key_available"] is False
    assert report["probe_attempted"] is False
    assert report["relay_endpoint"] and report["official_endpoint"]
    assert report["model_id"] and report["prompt_version"]
    assert report["spend_usd"] == 0.0
    assert "sk-" not in json.dumps(report)


# ---------------------------------------------------------------------------
# R3 — authority restoration stays byte-identical


def test_r3_authority_workorder_byte_identity():
    restored = REPO / "07_workorders" / "WORKORDER_AML_V1_ATOMIC_R1.md"
    source = subprocess.run(
        ["git", "show", f"{AUTHORITY_COMMIT}:07_workorders/WORKORDER_AML_V1_ATOMIC_R1.md"],
        capture_output=True, check=True, cwd=REPO,
    ).stdout
    assert restored.read_bytes() == source
    assert hashlib.sha256(restored.read_bytes()).hexdigest() == hashlib.sha256(source).hexdigest()
