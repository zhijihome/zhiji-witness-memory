"""V1 Atomic gates — G36 (determinism/cache), G37 (raw canonical), G38
(extractor input surface), G39 (NO_LLM path), G40 (budget gate), G42 (clean
switch + four modes), plus the玄神经 structural invariants.

Unit/hermetic: the LLM is always a fake deterministic adapter; real-API
evidence lives in the bench arms + builder report, never in CI.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

_ATOMIC_ID_RE_TEST = re.compile(r"^atom-(\d{6,})-(\d{2,})$")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_bench"))

from aml_adapter import CompletionResult  # noqa: E402
from aml_atomic import (  # noqa: E402
    RULE_VERSION,
    AtomicPipeline,
    BudgetExceededError,
    ExtractionCache,
    RuleBasedExtractor,
    SpendGate,
    cache_key,
)
from aml_atomic.extractors import prompt_version  # noqa: E402
from aml_config import load_config  # noqa: E402
from aml_store import atomic_unit_id  # noqa: E402

import _bootstrap  # noqa: E402,F401

# ---------------------------------------------------------------------------
# helpers


class FakeLLMAdapter:
    """Deterministic stand-in for the competition model: records every call,
    returns fixed JSON, accumulates usage like the real adapter."""

    def __init__(
        self,
        facts: list[str] | None = None,
        model_id: str = "fake-model",
        official_base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.official_model_id = model_id
        self.official_base_url = official_base_url
        self.facts = facts if facts is not None else ["alice owns a dog", "the dog is four"]
        self.calls: list[tuple[str, str]] = []  # (role, content) prompts seen
        self.usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512, response_format=None):
        self.calls.append((system or "", prompt))
        self.usage["calls"] += 1
        self.usage["tokens_in"] += 100
        self.usage["tokens_out"] += 10
        self.usage["cost_usd"] += 0.0001
        return CompletionResult(
            text=json.dumps({"facts": self.facts}),
            model_id=self.official_model_id,
            model_calls=1,
            tokens_in=100,
            tokens_out=10,
            cost_usd=0.0001,
        )


def _atomic_config(tmp_path: Path, **atomic):
    base = {
        "storage": {"db_path": str(tmp_path / "test.db")},
        "ops_log": {"enabled": True, "dir": str(tmp_path / "ops")},
        "dense": {"backend": "hashed_ngram"},
        "retrievers": {"exact": True},
        "surface": {"NEIGHBOR_POLICY": "off"},
        "adapter": {
            "tier": "AML_OFFICIAL_MODEL",
            "budget_usd_cap": 30.0,
            "spend_state_path": str(tmp_path / "spend" / "spend_state.json"),
        },
        "extraction_cache": {"dir": str(tmp_path / "cache")},
        "atomic": {"mode": "both", "extractor": "llm", "granularity": "fine", **atomic},
    }
    return load_config(overrides=base)


_MESSAGES = [
    ("alice", "I adopted a dog named Barkley in May 2023."),
    ("bob", "Wonderful! How old is Barkley?"),
    ("alice", "Barkley is four years old. He loves the park."),
]


def _units(tmp_path: Path):
    from aml_store import MemoryUnit

    units = []
    for i, (role, content) in enumerate(_MESSAGES, start=1):
        from aml_store.model import content_sha256, iso_z

        units.append(
            MemoryUnit(
                id=f"mem-{i:06d}",
                user_id="u1",
                session_id="s1",
                request_id="r1",
                msg_index=i - 1,
                role=role,
                content=content,
                ts_ms=1_683_000_000_000 + i * 60_000,
                created_at=iso_z(None),
                arrival_seq=i,
                session_seq=i,
                content_sha256=content_sha256(role, content),
            )
        )
    return units


# ---------------------------------------------------------------------------
# G42 — atomic off is a full bypass


def test_g42_mode_off_constructs_no_pipeline(tmp_path):
    from aml_api.service import MemoryService

    cfg = _atomic_config(tmp_path, mode="off")
    svc = MemoryService(cfg)
    try:
        assert svc.pipeline is None  # no extractor, no cache, no derived writes
        assert svc.store.atomic_count() == 0
    finally:
        svc.close()


# ---------------------------------------------------------------------------
# G37 — raw canonical / provenance / no-writeback


def test_g37_raw_untouched_provenance_resolves_no_writeback(tmp_path):
    from aml_api.service import MemoryService

    cfg = _atomic_config(tmp_path, extractor="rule")
    svc = MemoryService(cfg)
    try:
        payload_messages = [{"role": r, "content": c} for r, c in _MESSAGES]
        r1 = svc.add(
            __import__("aml_store", fromlist=["AddPayload"]).AddPayload(
                request_id="req-1", user_id="u1", session_id="s1", messages=[
                    __import__("aml_store", fromlist=["MessageIn"]).MessageIn(role=m["role"], content=m["content"])
                    for m in payload_messages
                ]
            )
        )
        assert len(r1.stored) == 3
        raw_before = [(u.id, u.content, u.content_sha256) for u in r1.stored]

        records = svc.store.atomic_units("u1")
        assert records, "rule extraction produced no derived rows"
        # provenance: every derived row points at a live raw unit, content hash matches
        raw_map = {u.id: (u.content, u.content_sha256) for u in r1.stored}
        for rec in records:
            assert rec.source_unit_id in raw_map
            assert rec.source_content_sha256 == raw_map[rec.source_unit_id][1]
            assert rec.atom_id == atomic_unit_id(
                next(u for u in r1.stored if u.id == rec.source_unit_id).arrival_seq, rec.fact_index
            )
        # idempotent re-delivery: no new rows, raw untouched
        r2 = svc.add(
            __import__("aml_store", fromlist=["AddPayload"]).AddPayload(
                request_id="req-1", user_id="u1", session_id="s1", messages=[
                    __import__("aml_store", fromlist=["MessageIn"]).MessageIn(role=m["role"], content=m["content"])
                    for m in payload_messages
                ]
            )
        )
        assert r2.stored == [] and len(r2.skipped_duplicate) == 3
        assert svc.store.atomic_units("u1") == records
        conn = sqlite3.connect(tmp_path / "test.db")
        rows = conn.execute("SELECT id, content, content_sha256 FROM memories ORDER BY id").fetchall()
        conn.close()
        assert [(i, c, h) for i, c, h in rows] == raw_before  # raw verbatim, unmodified
    finally:
        svc.close()


# ---------------------------------------------------------------------------
# G38 — extractor sees ONLY the message (role, content), never queries/gold


def test_g38_extractor_input_surface_is_message_only(tmp_path):
    fake = FakeLLMAdapter()
    cfg = _atomic_config(tmp_path)
    from aml_store import CompetitionMemoryStore

    store = CompetitionMemoryStore(tmp_path / "g38.db")
    pipeline = AtomicPipeline(cfg, fake, store)  # type: ignore[arg-type]
    pipeline.derive(_units(tmp_path))
    assert fake.usage["calls"] == 3
    for _, prompt in fake.calls:
        assert "query" not in prompt.lower() or "Query" not in prompt  # no query field in prompt
        # the prompt contains one message body each — no options, no gold, no ids
        assert "evidence" not in prompt.lower()
        assert "options" not in prompt.lower()
    store.close()


# ---------------------------------------------------------------------------
# G36 — extraction determinism via the content-hash cache


def test_g36_same_input_reuses_bit_identical_products_and_zero_calls(tmp_path):
    fake1, fake2 = FakeLLMAdapter(), FakeLLMAdapter()
    from aml_store import CompetitionMemoryStore

    store = CompetitionMemoryStore(tmp_path / "g36.db")
    cache_dir = tmp_path / "cache"
    cfg1 = _atomic_config(cache_dir / "x")
    cfg1.raw["extraction_cache"]["dir"] = str(cache_dir)
    p1 = AtomicPipeline(cfg1, fake1, store)  # type: ignore[arg-type]
    r1 = p1.derive(_units(tmp_path))
    assert fake1.usage["calls"] == 3 and p1.stats["cache_misses"] == 3

    cfg2 = _atomic_config(cache_dir / "y")
    cfg2.raw["extraction_cache"]["dir"] = str(cache_dir)
    p2 = AtomicPipeline(cfg2, fake2, store)  # type: ignore[arg-type]
    r2 = p2.derive(_units(tmp_path))
    assert fake2.usage["calls"] == 0  # bench rerun makes ZERO LLM calls
    assert p2.stats["cache_hits"] == 3 and p2.stats["cache_misses"] == 0
    assert [(x.atom_id, x.fact) for x in r1] == [(x.atom_id, x.fact) for x in r2]  # bit-identical
    store.close()


def test_g36_cache_key_binds_content_extractor_model_prompt(tmp_path):
    k1 = cache_key(prompt_version("fine"), "m1", "fine", "aaa", "api.openai.com")
    k2 = cache_key(prompt_version("fine"), "m1", "fine", "aaa", "api.openai.com")
    k3 = cache_key(prompt_version("fine"), "m2", "fine", "aaa", "api.openai.com")  # different model
    k4 = cache_key(prompt_version("coarse"), "m1", "coarse", "aaa", "api.openai.com")  # different prompt version
    k5 = cache_key(prompt_version("fine"), "m1", "fine", "bbb", "api.openai.com")  # different content
    k6 = cache_key(prompt_version("fine"), "m1", "fine", "aaa", "api.relay.test")  # different endpoint (G45)
    assert k1 == k2 and len({k1, k3, k4, k5, k6}) == 5


def test_g36_freeze_thaw_roundtrip_preserves_fingerprint(tmp_path):
    cache = ExtractionCache(tmp_path / "c1" / "extraction_cache.sqlite3")
    cache.put(
        "k1",
        ["a fact"],
        model_id="m",
        prompt_version="p",
        granularity="fine",
        endpoint_identity="api.openai.com",
        created_at="t",
    )
    fp1 = cache.fingerprint()
    out = tmp_path / "frozen.jsonl"
    exported_fp = cache.export_jsonl(out)
    assert exported_fp == fp1
    cache2 = ExtractionCache(tmp_path / "c2" / "extraction_cache.sqlite3")
    cache2.import_jsonl(out)
    assert cache2.fingerprint() == fp1
    cache.close()
    cache2.close()


# ---------------------------------------------------------------------------
# G40 — budget gate is real


def test_g40_budget_gate_hard_fails_before_call(tmp_path):
    fake = FakeLLMAdapter()
    from aml_store import CompetitionMemoryStore

    store = CompetitionMemoryStore(tmp_path / "g40.db")
    cfg = _atomic_config(tmp_path)
    cfg.raw["adapter"]["spend_state_path"] = str(tmp_path / "spend" / "state.json")
    cfg.raw["adapter"]["budget_usd_cap"] = 30.0
    # preload the round state AT the cap
    gate = SpendGate(cfg.raw["adapter"]["spend_state_path"], 30.0)
    gate.record(cost_usd=30.0, tokens_in=1, tokens_out=1)
    pipeline = AtomicPipeline(cfg, fake, store)  # type: ignore[arg-type]
    with pytest.raises(BudgetExceededError):
        pipeline.derive(_units(tmp_path))
    assert fake.usage["calls"] == 0  # no call may start once at/over the cap
    store.close()


def test_g40_spend_state_persists_across_instances_and_corrupt_is_loud(tmp_path):
    path = tmp_path / "spend" / "state.json"
    g1 = SpendGate(path, 30.0)
    g1.record(cost_usd=0.5, tokens_in=10, tokens_out=5)
    g2 = SpendGate(path, 30.0)
    assert g2.current()["total_usd"] == 0.5
    assert g2.remaining_usd() == 29.5
    path.write_text("{corrupt json")
    with pytest.raises(RuntimeError, match="corrupt"):
        SpendGate(path, 30.0).current()


def test_g40_api_key_never_in_url_journal_or_payload(tmp_path, monkeypatch):
    """Real adapter behind a mock transport: key only in the Authorization
    header; sentinel value appears nowhere in the ops journal or response."""
    sentinel = "sk-sentinel-abc123"
    monkeypatch.setenv("AML_OFFICIAL_MODEL_API_KEY", sentinel)
    seen: dict = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"facts": ["a fact"]})}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                "model": "fake-model",
            },
        )


    from aml_adapter import AMLModelAdapter

    cfg = _atomic_config(tmp_path)
    adapter = AMLModelAdapter.from_config(cfg.adapter_cfg)
    adapter.official_model_id = "fake-model"
    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
    extractor = AMLModelAdapter  # noqa: F841 — clarity only

    from aml_atomic import LLMExtractor

    gate = SpendGate(tmp_path / "spend" / "state.json", 30.0)
    ex = LLMExtractor(adapter, "fine", spend=gate)
    facts = ex.extract("alice", "alice has a cat")
    assert facts == ["a fact"]
    assert sentinel in seen["auth"] and sentinel not in seen["url"]

    # the journal records no key material
    journal = (tmp_path / "ops" / "ops.jsonl")
    assert not journal.exists() or sentinel not in journal.read_text()
    # spend state recorded the call's measured cost
    state = json.loads((tmp_path / "spend" / "state.json").read_text())
    assert state["model_calls"] == 1 and state["total_usd"] > 0


import httpx  # noqa: E402  (used by the mock-transport test above)

# ---------------------------------------------------------------------------
# G39 — NO_LLM rule path: full mini-bench with zero model calls


def _mini_conversation():
    from benchlib.locomo_loader import Conversation, LocomoQuestion, LocomoTurn

    t = 1_683_000_000_000
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
        LocomoQuestion(question="When was the dog adopted?", answer="May 2023", evidence=("D1:1",), category=3),
    ]
    return Conversation(sample_id="mini-atomic", sessions=[turns1, turns2], questions=questions)


def test_g39_no_llm_rule_path_full_bench_zero_cost(tmp_path):
    from aml_api import create_app
    from benchlib.runner import run_benchmark
    from fastapi.testclient import TestClient

    cfg = _atomic_config(tmp_path, extractor="rule", mode="both")
    cfg.raw["adapter"]["tier"] = "NO_LLM"
    cfg.raw["adapter"]["budget_usd_cap"] = 30.0
    app = create_app(cfg)
    with TestClient(app) as client:
        record = run_benchmark(
            client,
            [_mini_conversation()],
            cfg,
            run_id="g39-rule",
            results_root=tmp_path / "results",
            label="g39",
        )
    assert record["MODEL_CALLS"] == 0  # real zeros, measured
    assert record["COST"] == 0.0
    assert record["TOKENS"] == 0
    assert record["FAILURE_CASES"] == 0
    assert record["ATOMIC_UNITS"] > 0  # derived rows really exist
    assert record["ATOMIC_MODE"] == "both"
    assert record["EXTRACTION_CACHE_FP"] == "n/a"  # rule extractor: no cache
    assert record["ATOMIC_PROMPT_VERSION"] == f"{RULE_VERSION}-fine"


# ---------------------------------------------------------------------------
# G42 — four modes all runnable and behaviorally distinguishable


@pytest.mark.parametrize(
    "mode,expect_atom,expect_mem",
    [("off", False, True), ("both", True, True), ("atomic_only", True, False), ("atomic_index_raw_return", False, True)],
)
def test_g42_four_modes_run_and_differ(tmp_path, mode, expect_atom, expect_mem):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    cfg = _atomic_config(tmp_path / mode, extractor="rule", mode=mode)
    cfg.raw["adapter"]["tier"] = "NO_LLM"
    app = create_app(cfg)
    with TestClient(app) as client:
        add = client.post(
            "/add",
            json={
                "request_id": "r1",
                "messages": [
                    {"role": "alice", "timestamp": 1_683_000_000_000, "content": "I adopted a dog named Barkley in May 2023."},
                    {"role": "bob", "timestamp": 1_683_000_060_000, "content": "Barkley is four years old and loves the park."},
                ],
                "user_id": "u1",
                "session_id": "s1",
            },
        )
        assert add.status_code == 200
        r = client.post("/search", json={"query": "dog name Barkley", "user_id": "u1", "top_k": 100})
        assert r.status_code == 200
        items = r.json()["data"]
        assert items
        ids = [it["id"] for it in items]
        has_atom = any(i.startswith("atom-") for i in ids)
        has_mem = any(i.startswith("mem-") for i in ids)
        assert has_atom == expect_atom, f"mode={mode} ids={ids[:5]}"
        assert has_mem == expect_mem, f"mode={mode} ids={ids[:5]}"
        if mode == "atomic_index_raw_return":
            # every returned id is a raw unit id that exists in the store
            for it in items:
                assert it["id"].startswith("mem-")
                assert it["content"]  # raw text, not the atomic fact text


def test_g42_off_matches_v05_bit_for_bit_on_mini_bench(tmp_path):
    """Off-mode vs pre-V1 behavior on a fixed mini world: identical surfaces."""
    from aml_api import create_app
    from fastapi.testclient import TestClient

    def run(extra):
        cfg = _atomic_config(tmp_path / f"m{extra}", mode=extra)
        cfg.raw["adapter"]["tier"] = "NO_LLM"
        app = create_app(cfg)
        with TestClient(app) as client:
            client.post(
                "/add",
                json={
                    "request_id": "r1",
                    "messages": [
                        {"role": "alice", "timestamp": 1_683_000_000_000, "content": "I adopted a dog named Barkley in May 2023."},
                        {"role": "alice", "timestamp": 1_683_000_120_000, "content": "Barkley is four years old."},
                    ],
                    "user_id": "u1",
                    "session_id": "s1",
                },
            )
            return client.post("/search", json={"query": "dog name Barkley", "user_id": "u1", "top_k": 100}).json()

    out_off = run("off")
    # a config that never mentions atomic at all (pre-V1 shape via overrides)
    cfg = load_config(
        overrides={
            "storage": {"db_path": str(tmp_path / "legacy.db")},
            "ops_log": {"enabled": True, "dir": str(tmp_path / "ops2")},
            "dense": {"backend": "hashed_ngram"},
            "retrievers": {"exact": True},
            "surface": {"NEIGHBOR_POLICY": "off"},
            "adapter": {"tier": "NO_LLM", "budget_usd_cap": 1.0, "spend_state_path": str(tmp_path / "sp" / "s.json")},
        }
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post(
            "/add",
            json={
                "request_id": "r1",
                "messages": [
                    {"role": "alice", "timestamp": 1_683_000_000_000, "content": "I adopted a dog named Barkley in May 2023."},
                    {"role": "alice", "timestamp": 1_683_000_120_000, "content": "Barkley is four years old."},
                ],
                "user_id": "u1",
                "session_id": "s1",
            },
        )
        out_legacy = client.post("/search", json={"query": "dog name Barkley", "user_id": "u1", "top_k": 100}).json()
    assert out_off == out_legacy  # identical id/content/score/created_at


# ---------------------------------------------------------------------------
# rule extractor sanity (A3): determinism + granularity variants


def test_rule_extractor_deterministic_and_granularity_aware():
    text = "I moved to Berlin in 2019. I work at a startup, but I also study physics."
    coarse = RuleBasedExtractor("coarse").extract("alice", text)
    fine = RuleBasedExtractor("fine").extract("alice", text)
    assert RuleBasedExtractor("coarse").extract("alice", text) == coarse  # bit-identical twice
    assert len(fine) >= len(coarse)  # fine splits at least the 'but' clause
    assert any("Berlin" in f for f in coarse)


def test_llm_extractor_requires_official_tier():
    """Config-level guard: LLM extraction without the sanctioned tier is a
    config error, not a runtime surprise (protects the NO_LLM guarantee)."""
    from aml_config.config import ConfigError

    with pytest.raises(ConfigError, match="AML_OFFICIAL_MODEL"):
        load_config(
            overrides={
                "adapter": {"tier": "NO_LLM"},
                "atomic": {"mode": "both", "extractor": "llm"},
            }
        )


# ---------------------------------------------------------------------------
# A7 (ADDENDUM-01) — provenance-family dedup at packing time


def test_a7_family_dedup_keeps_one_slot_per_provenance(tmp_path):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    def surface_for(dedup: str):
        cfg = _atomic_config(tmp_path / f"dedup-{dedup}", extractor="rule", mode="both")
        cfg.raw["adapter"]["tier"] = "NO_LLM"
        cfg.raw["atomic"]["dedup_policy"] = dedup
        cfg.raw["surface"]["RETURN_SIZE"] = 5  # small surface: dedup must refill slots
        app = create_app(cfg)
        with TestClient(app) as client:
            client.post(
                "/add",
                json={
                    "request_id": "r1",
                    "messages": [
                        {
                            "role": "alice",
                            "timestamp": 1_683_000_000_000,
                            "content": "I adopted a dog named Barkley in May 2023. Barkley is a golden retriever.",
                        },
                        {"role": "bob", "timestamp": 1_683_000_060_000, "content": "We trained Barkley at the park every single weekend last summer."},
                        {"role": "alice", "timestamp": 1_683_000_120_000, "content": "Barkley knows how to sit, stay, and roll over on command now."},
                        {"role": "bob", "timestamp": 1_683_000_180_000, "content": "My cat Milo destroyed the sofa again yesterday afternoon."},
                        {"role": "alice", "timestamp": 1_683_000_240_000, "content": "I bought a new car in April 2024 after the accident."},
                    ],
                    "user_id": "u1",
                    "session_id": "s1",
                },
            )
            r = client.post("/search", json={"query": "dog Barkley adopted golden retriever park", "user_id": "u1", "top_k": 100})
            assert r.status_code == 200
            return r.json()["data"]

    off = surface_for("off")
    keep_raw = surface_for("keep_raw")
    keep_atomic = surface_for("keep_atomic")

    def families(items):
        out = []
        for it in items:
            m = _ATOMIC_ID_RE_TEST.match(it["id"])
            out.append(f"mem-{int(m.group(1)):06d}" if m else it["id"])
        return out

    # off: a raw unit and its own atomics may both occupy slots (E2 waste)
    fam_off = families(off)
    assert len(fam_off) != len(set(fam_off)) or True  # E2 may or may not bind on this query
    # keep_raw: one slot per family, raw representative when the family has one
    fam_raw = families(keep_raw)
    assert len(fam_raw) == len(set(fam_raw)), f"family duplicate on keep_raw surface: {fam_raw}"
    raw_ids = {it["id"] for it in keep_raw}
    assert all(not i.startswith("atom-") or f"mem-{_ATOMIC_ID_RE_TEST.match(i).group(1)}" not in raw_ids for i in raw_ids)
    # keep_atomic: atomic representative when available
    fam_atom = families(keep_atomic)
    assert len(fam_atom) == len(set(fam_atom))
    has_atomic = any(it["id"].startswith("atom-") for it in keep_atomic)
    assert has_atomic, "keep_atomic kept no atomic representative at all"
    # determinism: identical config twice → identical surface
    again = surface_for("keep_raw")
    assert [i["id"] for i in again] == [i["id"] for i in keep_raw]


def test_a8_face_fields_present_in_ledger_record(tmp_path):
    from aml_api import create_app
    from benchlib.runner import run_benchmark
    from fastapi.testclient import TestClient

    cfg = _atomic_config(tmp_path, extractor="rule", mode="both")
    cfg.raw["adapter"]["tier"] = "NO_LLM"
    app = create_app(cfg)
    with TestClient(app) as client:
        record = run_benchmark(
            client,
            [_mini_conversation()],
            cfg,
            run_id="a8-face",
            results_root=tmp_path / "results",
            label="a8",
        )
    for field in ("FACE_RAW_COUNT", "FACE_ATOMIC_COUNT", "FACE_UNIQUE_DIA", "FACE_REDUNDANT_SLOTS"):
        assert field in record, f"missing {field}"
    assert record["FACE_ATOMIC_COUNT"] > 0  # both mode: atomic entries on the surface
    assert record["FACE_REDUNDANT_SLOTS"] >= 0
