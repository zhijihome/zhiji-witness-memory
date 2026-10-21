"""AMLModelAdapter tiers, usage accounting, and model-id containment
(workorder deliverables 12/13; NO_LLM path deliverable 14)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from aml_adapter import AMLModelAdapter, LLMUnavailableError

REPO = Path(__file__).resolve().parent.parent


def test_no_llm_tier_forbids_calls_and_stays_at_zero():
    adapter = AMLModelAdapter(tier="NO_LLM")
    assert adapter.llm_enabled is False
    with pytest.raises(LLMUnavailableError):
        adapter.complete("hello")
    assert adapter.usage.model_calls == 0
    assert adapter.usage.tokens_in == 0
    assert adapter.usage.cost_usd == 0.0


def test_local_surrogate_is_deterministic_and_free():
    adapter = AMLModelAdapter(tier="LOCAL_SURROGATE")
    r1 = adapter.complete("one two three four five")
    r2 = adapter.complete("one two three four five")
    assert r1.text == r2.text
    assert r1.cost_usd == 0.0
    assert adapter.usage.model_calls == 2
    assert adapter.usage.tokens_in > 0


def test_official_tier_uses_configured_model_id_and_accounts():
    adapter = AMLModelAdapter(
        tier="AML_OFFICIAL_MODEL",
        official_model_id="gpt-4o-mini",
        official_base_url="http://mock.test/v1",
        price_per_1m_input_usd=0.15,
        price_per_1m_output_usd=0.60,
    )
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
    import os

    os.environ.setdefault("AML_OFFICIAL_MODEL_API_KEY", "test-key")
    result = adapter.complete("ping")
    assert result.text == "ok"
    assert captured["body"]["model"] == "gpt-4o-mini"  # from config, not hardcoded elsewhere
    assert "api-key" not in captured["url"] and "test-key" not in captured["url"]  # key never in URL
    assert captured["auth"] == "Bearer test-key"
    assert adapter.usage.model_calls == 1
    assert adapter.usage.tokens_in == 100 and adapter.usage.tokens_out == 50
    assert adapter.usage.cost_usd == pytest.approx(100 / 1e6 * 0.15 + 50 / 1e6 * 0.60)


def test_official_tier_custom_model_id_flows_through():
    adapter = AMLModelAdapter(tier="AML_OFFICIAL_MODEL", official_model_id="other-model-id")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
    import os

    old = os.environ.get("AML_OFFICIAL_MODEL_API_KEY")
    os.environ["AML_OFFICIAL_MODEL_API_KEY"] = "k"
    try:
        adapter.complete("ping")
        assert captured["model"] == "other-model-id"
    finally:
        if old is None:
            del os.environ["AML_OFFICIAL_MODEL_API_KEY"]
        else:
            os.environ["AML_OFFICIAL_MODEL_API_KEY"] = old


def _py_files(folder: Path):
    for py in folder.rglob("*.py"):
        if py.name.startswith("._") or "__pycache__" in py.parts:
            continue  # macOS AppleDouble junk on exFAT volumes
        yield py


def test_model_id_not_hardcoded_in_store_or_retrieval():
    """Deliverable 13: the official model id may live only in adapter + config."""
    banned_dirs = ["01_store", "02_retrieval", "09_api"]
    for folder in banned_dirs:
        for py in _py_files(REPO / folder):
            text = py.read_text(encoding="utf-8")
            assert "gpt-4o-mini" not in text, f"model id leaked into {py}"


def test_official_model_id_in_defaults_only_config_and_adapter():
    allowed = {
        REPO / "03_adapter" / "adapter.py",
        REPO / "10_config" / "defaults.json",
    }
    for py in _py_files(REPO / "10_config"):
        if "gpt-4o-mini" in py.read_text(encoding="utf-8"):
            pytest.fail(f"unexpected model id in config code: {py}")
    assert any("gpt-4o-mini" in p.read_text(encoding="utf-8") for p in allowed)
