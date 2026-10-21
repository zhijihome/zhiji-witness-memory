"""AMLModelAdapter (architecture layer 00) — the single LLM gateway.

Three tiers (Architecture C3):
  * NO_LLM            — LLM calls are structurally impossible; `complete`
                        raises. The whole V0 stack runs in this tier.
  * LOCAL_SURROGATE   — deterministic local stand-in used only for cheap smoke
                        plumbing. NOT an LLM; its outputs must never be scored
                        as anything other than LOCAL coordinate results.
  * AML_OFFICIAL_MODEL — the competition model (default id `gpt-4o-mini`, from
                        config only) behind an OpenAI-compatible Chat Completions
                        API. The model id deliberately lives here and in config;
                        retrieval/storage code never references it (workorder
                        deliverable 13).

Usage accounting (model_calls / tokens / cost) is accumulated here so the API
ops log and the bench ledger can report exact numbers per tier.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from aml_config.config import (
    COORDINATES,
    OFFICIAL_COORDINATES,
    canonical_endpoint_identity,
    endpoint_allowed_for_coordinate,
)

DEFAULT_OFFICIAL_MODEL_ID = "gpt-4o-mini"

# Bounded retry per Contract §7: 408/425/429 (timeout/rate-limit) and
# 5xx/server errors are retryable; everything else fails immediately.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_S = (1.0, 3.0)

TIERS = ("NO_LLM", "LOCAL_SURROGATE", "AML_OFFICIAL_MODEL")


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM call is attempted in a tier that forbids it."""


class OfficialEndpointViolationError(RuntimeError):
    """G44 (ADDENDUM-02): an OFFICIAL_* coordinate is configured but the LLM
    endpoint is not the contract's official OpenAI endpoint. Raised BEFORE any
    model call — never downgraded to a warning, never silently fallback."""


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model_id: str
    model_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


@dataclass
class Usage:
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        self.model_calls += other.model_calls
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.cost_usd += other.cost_usd


@dataclass
class AMLModelAdapter:
    tier: str = "NO_LLM"
    official_model_id: str = DEFAULT_OFFICIAL_MODEL_ID
    official_base_url: str = "https://api.openai.com/v1"
    official_api_key_env: str = "AML_OFFICIAL_MODEL_API_KEY"
    coordinate: str = "LOCAL"
    price_per_1m_input_usd: float = 0.15
    price_per_1m_output_usd: float = 0.60
    request_timeout_s: float = 60.0
    usage: Usage = field(default_factory=Usage)
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; expected one of {TIERS}")
        if self.coordinate not in COORDINATES:
            raise ValueError(f"unknown coordinate {self.coordinate!r}; expected one of {COORDINATES}")
        self.assert_endpoint_allowed()

    def assert_endpoint_allowed(self) -> None:
        """G44 hard gate: OFFICIAL_* coordinates demand the official endpoint.
        Raises before any model call; LOCAL relays pass untouched."""
        if self.coordinate in OFFICIAL_COORDINATES and not endpoint_allowed_for_coordinate(
            self.coordinate, self.official_base_url
        ):
            raise OfficialEndpointViolationError(
                f"coordinate {self.coordinate} requires endpoint host "
                f"'{canonical_endpoint_identity('https://api.openai.com/v1')}' but "
                f"official_base_url is {self.official_base_url!r} — refusing to call "
                "(evaluation content must not be sent to a third-party relay)"
            )

    # -- construction from config ------------------------------------------

    @classmethod
    def from_config(cls, adapter_cfg: dict[str, Any]) -> AMLModelAdapter:
        return cls(
            tier=str(adapter_cfg.get("tier", "NO_LLM")),
            official_model_id=str(adapter_cfg.get("official_model_id", DEFAULT_OFFICIAL_MODEL_ID)),
            official_base_url=str(adapter_cfg.get("official_base_url", "https://api.openai.com/v1")),
            official_api_key_env=str(adapter_cfg.get("official_api_key_env", "AML_OFFICIAL_MODEL_API_KEY")),
            coordinate=str(adapter_cfg.get("coordinate", "LOCAL")),
            price_per_1m_input_usd=float(adapter_cfg.get("price_per_1m_input_usd", 0.15)),
            price_per_1m_output_usd=float(adapter_cfg.get("price_per_1m_output_usd", 0.60)),
            request_timeout_s=float(adapter_cfg.get("request_timeout_s", 60.0)),
        )

    # -- completions ---------------------------------------------------------

    @property
    def llm_enabled(self) -> bool:
        return self.tier in ("LOCAL_SURROGATE", "AML_OFFICIAL_MODEL")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        if self.tier == "NO_LLM":
            raise LLMUnavailableError("NO_LLM tier: LLM calls are disabled by configuration")
        if self.tier == "LOCAL_SURROGATE":
            return self._complete_surrogate(prompt, max_tokens)
        return self._complete_official(prompt, system=system, max_tokens=max_tokens, response_format=response_format)

    # -- LOCAL_SURROGATE -----------------------------------------------------
    # Deterministic extractive echo — plumbing tests only, never a real answerer.

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.split()))

    def _complete_surrogate(self, prompt: str, max_tokens: int) -> CompletionResult:
        words = prompt.split()
        text = " ".join(words[:max_tokens])
        tokens_in = self._estimate_tokens(prompt)
        tokens_out = self._estimate_tokens(text)
        self.usage.add(Usage(model_calls=1, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=0.0))
        return CompletionResult(
            text=text,
            model_id="local-surrogate-extractive",
            model_calls=1,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
        )

    # -- AML_OFFICIAL_MODEL ----------------------------------------------------

    def _official_api_key(self) -> str:
        return os.environ.get(self.official_api_key_env, "")

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.request_timeout_s)
        return self._client

    def _complete_official(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        api_key = self._official_api_key()
        if not api_key:
            raise LLMUnavailableError(
                f"{self.official_api_key_env} is not set; AML_OFFICIAL_MODEL tier requires it"
            )
        # G44 re-check at the last possible moment: hard fail BEFORE the model
        # call, even if the dataclass was mutated after construction.
        self.assert_endpoint_allowed()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # Determinism (V1 workorder A2): temperature 0 and a fixed seed on
        # every official call. The residual provider-side nondeterminism is
        # pinned out by the extraction cache, not by this flag alone.
        body: dict[str, Any] = {
            "model": self.official_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
        }
        if response_format is not None:
            body["response_format"] = response_format
        client = self._get_client()
        headers = {"Authorization": f"Bearer {api_key}"}  # key never in URL (Contract §7)
        url = f"{self.official_base_url.rstrip('/')}/chat/completions"
        response: Any = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = client.post(url, headers=headers, json=body)
            except Exception:  # network-level (timeout/transport): bounded retry
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)])
                    continue
                raise
            if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
                delay = _BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)]
                retry_after = response.headers.get("retry-after")
                if retry_after is not None:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(delay)
                continue
            response.raise_for_status()  # non-retryable classes fail immediately
            break
        if response is None:
            raise RuntimeError("unreachable: retry loop exited without response")
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        tokens_in = int(payload.get("usage", {}).get("prompt_tokens", 0))
        tokens_out = int(payload.get("usage", {}).get("completion_tokens", 0))
        cost = (
            tokens_in / 1_000_000 * self.price_per_1m_input_usd
            + tokens_out / 1_000_000 * self.price_per_1m_output_usd
        )
        self.usage.add(Usage(model_calls=1, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost))
        return CompletionResult(
            text=text,
            model_id=payload.get("model", self.official_model_id),
            model_calls=1,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
