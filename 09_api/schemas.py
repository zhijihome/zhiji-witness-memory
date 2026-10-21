"""Contract-facing request/response schemas (Contract §4/§5/§6/§7).

Field discipline:
  * only fields the platform actually sends are modeled — `metadata`,
    `app_id`, `agent_id`, `async_mode` are NOT read anywhere (G18);
  * unknown extra fields are ignored, never rejected;
  * responses carry exactly the contract-mandated keys.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    timestamp: int | None = Field(default=None, ge=0)


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    request_id: str = Field(min_length=1)
    messages: list[MessageIn]
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1, le=100_000)


class SearchItem(BaseModel):
    id: str
    content: str
    score: float
    created_at: str


class SearchResponse(BaseModel):
    data: list[SearchItem]


class HealthResponse(BaseModel):
    status: str = "ok"
    tier: str
    version: str
    # G23: echo of the server's effective behavior-relevant config, so a bench
    # client can verify the label it is about to write describes reality.
    config_fingerprint: str
    retrievers_on: list[str]
    RETURN_SIZE: str | int
    ORDERING_POLICY: str
    NEIGHBOR_POLICY: str
    dense_backend: str
    # V0.5 (O2): the resolved dense inference device, plus cumulative adapter
    # usage since server start, so the bench runner can aggregate
    # MODEL_CALLS/TOKENS/COST from measured deltas instead of hardcoded zeros
    # (real zeros under NO_LLM). Not part of the behavior fingerprint: they are
    # observations, not configuration.
    dense_device: str = "none"
    model_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    # V1 Atomic: cumulative derived-unit count, spend-gate total (whole-round,
    # persisted across restarts), and pipeline cache counters — measured
    # observations like the usage counters above, never config.
    atomic_units: int = 0
    spend_total_usd: float | None = None
    atomic_cache_hits: int = 0
    atomic_cache_misses: int = 0
    # G44/G46 (ADDENDUM-02): bench coordinate and the actual LLM endpoint
    # identity (normalized hostname only — never a URL with credentials) plus
    # the requested model id, so the ledger records real LLM provenance from
    # the SERVER side. "none" under NO_LLM.
    coordinate: str = "LOCAL"
    llm_endpoint: str = "none"
    llm_model_id: str = "none"
    # V2 promotion audit (WORKORDER_AML_V2_R1 §3.2): the configured family
    # aggregation plus cumulative promotion counters since server start —
    # measured observations like the usage counters above, never config.
    family_agg: str = "max"
    families_promoted_by_atomic: int = 0
    families_gate_blocked: int = 0
    orphan_promotions: int = 0
