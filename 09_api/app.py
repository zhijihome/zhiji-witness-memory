"""AML HTTP surface (architecture layer 08): POST /add, POST /search, GET /health.

Contract compliance carried by this module:
  * /add returns 200 only after synchronous persistence; body is exactly
    {success: true, request_id, user_id, session_id} (G13); never 202/task id
    (G14);
  * /search returns only memory evidence — there is no answer field, no LLM
    call, no gold data anywhere on this path (G4); `data` items always carry
    non-empty id/content; empty result is [] (G15); ordering/size honor the
    return-surface knobs and top_k (G16); session_id is not a filter (G17);
  * /health is unauthenticated and always 2xx (G20);
  * validation failures are 422/400 (no-retry class per Contract §7);
  * authentication, when AML_API_KEY is configured, accepts Bearer / X-Api-Key
    (key never in URL); wrong key → 401 (no-retry class) (G19).

Every add/search operation is journaled to the ops log with latency,
model_calls, tokens and cost (G9), plus the three return-surface knob values.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aml_config.config import AMLConfig, canonical_endpoint_identity, load_config
from aml_store import AddPayload
from aml_store import MessageIn as StoreMessageIn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .schemas import AddRequest, AddResponse, HealthResponse, SearchRequest, SearchResponse
from .service import MemoryService

VERSION = "0.1.0"


class OpsJournal:
    """Append-only JSONL journal: one line per add/search (G9 evidence)."""

    def __init__(self, enabled: bool, directory: str) -> None:
        self.enabled = enabled
        self.path = Path(directory) / "ops.jsonl"
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict[str, Any]) -> None:
        if not self.enabled:
            return
        entry = {"ts": datetime.now(tz=UTC).isoformat(timespec="milliseconds"), **entry}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _query_digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _dense_device(cfg: AMLConfig) -> str:
    """Resolved dense inference device for reporting ('none' when the hashed
    backend needs no model, 'off' when dense is disabled). Resolution never
    loads model weights."""
    backend = str(cfg.dense_cfg.get("backend", "hashed_ngram"))
    if backend == "dashscope":
        return "remote-api"
    if backend != "sentence_transformers":
        return "none"
    if not cfg.retrievers.dense:
        return "off"
    from aml_retrieval.dense import resolve_embedding_device

    return resolve_embedding_device(str(cfg.dense_cfg.get("device", "auto")))


def create_app(config: AMLConfig | None = None, service: MemoryService | None = None) -> FastAPI:
    cfg = config or load_config(path=os.environ.get("AML_CONFIG_PATH") or None)
    svc = service or MemoryService(cfg)
    surface = cfg.surface
    api_key = os.environ.get("AML_API_KEY", "")  # empty → auth disabled
    journal = OpsJournal(bool(cfg.ops_cfg.get("enabled", True)), cfg.ops_cfg.get("dir", "06_results/ops"))
    # V2 promotion audit (WORKORDER_AML_V2_R1 §3.2): cumulative counters over
    # all searches since server start, exposed on /health so the bench runner
    # can ledger the per-run deltas — measured observations, never config.
    promotion_totals = {"families_promoted_by_atomic": 0, "families_gate_blocked": 0, "orphan_promotions": 0}

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        svc.close()

    app = FastAPI(
        title="zhiji-aml",
        version=VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )

    def _auth_fail(request: Request) -> JSONResponse | None:
        if not api_key:
            return None
        header = request.headers.get("authorization", "")
        bearer = header[7:].strip() if header.lower().startswith("bearer ") else ""
        x_key = request.headers.get("x-api-key", "")
        token = request.headers.get("token", "")
        if api_key not in {bearer, x_key, token}:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return None

    @app.get("/health")
    def health() -> HealthResponse:
        # never authenticated, always 2xx (G20); behavior knobs echoed for the
        # bench client's G24 pre-flight check; resolved dense device + cumulative
        # adapter usage for the runner's O2 measured aggregation; V1 atomic
        # observability (derived-unit count, whole-round spend total, cache hits)
        usage = svc.adapter.usage
        try:
            from aml_atomic import SpendGate

            spend_total: float | None = SpendGate(
                str(cfg.adapter_cfg["spend_state_path"]), float(cfg.adapter_cfg["budget_usd_cap"])
            ).current()["total_usd"]
        except Exception:
            spend_total = None  # unreadable state stays honest null, never fake zero
        pipeline_stats = svc.pipeline.stats if svc.pipeline is not None else {}
        return HealthResponse(
            status="ok",
            tier=svc.adapter.tier,
            version=VERSION,
            config_fingerprint=cfg.behavior_fingerprint(),
            retrievers_on=cfg.retrievers.on(),
            RETURN_SIZE=surface.RETURN_SIZE,
            ORDERING_POLICY=surface.ORDERING_POLICY,
            NEIGHBOR_POLICY=surface.NEIGHBOR_POLICY,
            dense_backend=str(cfg.dense_cfg.get("backend", "hashed_ngram")),
            dense_device=_dense_device(cfg),
            model_calls=int(usage.model_calls),
            tokens=int(usage.tokens_in + usage.tokens_out),
            cost_usd=round(usage.cost_usd, 8),
            atomic_units=int(svc.stats().get("atomic_units", 0)),
            spend_total_usd=None if spend_total is None else round(spend_total, 8),
            atomic_cache_hits=int(pipeline_stats.get("cache_hits", 0)),
            atomic_cache_misses=int(pipeline_stats.get("cache_misses", 0)),
            coordinate=svc.adapter.coordinate,
            llm_endpoint=(
                "none"
                if svc.adapter.tier == "NO_LLM"
                else canonical_endpoint_identity(svc.adapter.official_base_url)
            ),
            llm_model_id="none" if svc.adapter.tier == "NO_LLM" else svc.adapter.official_model_id,
            family_agg=str(cfg.atomic_cfg.get("family_agg", "max")),
            families_promoted_by_atomic=promotion_totals["families_promoted_by_atomic"],
            families_gate_blocked=promotion_totals["families_gate_blocked"],
            orphan_promotions=promotion_totals["orphan_promotions"],
        )

    @app.post("/add")
    def add(request: Request, payload: AddRequest) -> JSONResponse:
        denied = _auth_fail(request)
        if denied is not None:
            journal.record({"op": "add", "status": 401, "latency_ms": 0, "user_id": payload.user_id})
            return denied
        started = time.perf_counter()
        try:
            outcome = svc.add(
                AddPayload(
                    request_id=payload.request_id,
                    user_id=payload.user_id,
                    session_id=payload.session_id,
                    messages=[
                        StoreMessageIn(role=m.role, content=m.content, timestamp=m.timestamp)
                        for m in payload.messages
                    ],
                )
            )
            body = AddResponse(
                success=True,
                request_id=outcome.request_id,
                user_id=outcome.user_id,
                session_id=outcome.session_id,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            usage = svc.adapter.usage
            pipeline_stats = svc.pipeline.stats if svc.pipeline is not None else {}
            journal.record(
                {
                    "op": "add",
                    "status": 200,
                    "latency_ms": round(latency_ms, 3),
                    "request_id": payload.request_id,
                    "user_id": payload.user_id,
                    "session_id": payload.session_id,
                    "n_messages": len(payload.messages),
                    "units_stored": len(outcome.stored),
                    "units_skipped_duplicate": len(outcome.skipped_duplicate),
                    "atomic_units": svc.stats().get("atomic_units", 0),
                    "atomic_cache_hits": pipeline_stats.get("cache_hits", 0),
                    "atomic_cache_misses": pipeline_stats.get("cache_misses", 0),
                    "model_calls": usage.model_calls,
                    "tokens": usage.tokens_in + usage.tokens_out,
                    "cost_usd": round(usage.cost_usd, 8),
                }
            )
            return JSONResponse(status_code=200, content=json.loads(body.model_dump_json()))
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000.0
            journal.record(
                {"op": "add", "status": 500, "latency_ms": round(latency_ms, 3), "user_id": payload.user_id}
            )
            raise

    @app.post("/search")
    def search(request: Request, payload: SearchRequest) -> JSONResponse:
        denied = _auth_fail(request)
        if denied is not None:
            journal.record(
                {"op": "search", "status": 401, "latency_ms": 0, "user_id": payload.user_id}
            )
            return denied
        started = time.perf_counter()
        try:
            query = payload.query
            if payload.options and bool(cfg.search_cfg.get("include_options_in_query", False)):
                query = query + "\n" + "\n".join(payload.options)
            outcome = svc.search(payload.user_id, query, payload.top_k)
            items = [item.to_api_dict() for item in outcome.items]
            latency_ms = (time.perf_counter() - started) * 1000.0
            usage = svc.adapter.usage
            for counter in promotion_totals:
                promotion_totals[counter] += int(outcome.trace.get(counter, 0))
            journal.record(
                {
                    "op": "search",
                    "status": 200,
                    "latency_ms": round(latency_ms, 3),
                    "user_id": payload.user_id,
                    "query_digest": _query_digest(payload.query),
                    "top_k": payload.top_k,
                    "n_returned": len(items),
                    "retrievers_on": outcome.trace.get("retrievers_on"),
                    "RETURN_SIZE": surface.RETURN_SIZE,
                    "ORDERING_POLICY": surface.ORDERING_POLICY,
                    "NEIGHBOR_POLICY": surface.NEIGHBOR_POLICY,
                    "neighbor_added": outcome.trace.get("neighbor_added"),
                    "family_agg": outcome.trace.get("family_agg"),
                    "families_promoted_by_atomic": outcome.trace.get("families_promoted_by_atomic", 0),
                    "families_gate_blocked": outcome.trace.get("families_gate_blocked", 0),
                    "orphan_promotions": outcome.trace.get("orphan_promotions", 0),
                    "model_calls": usage.model_calls,
                    "tokens": usage.tokens_in + usage.tokens_out,
                    "cost_usd": round(usage.cost_usd, 8),
                }
            )
            return JSONResponse(status_code=200, content=SearchResponse(data=items).model_dump())
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000.0
            journal.record(
                {
                    "op": "search",
                    "status": 500,
                    "latency_ms": round(latency_ms, 3),
                    "user_id": payload.user_id,
                    "query_digest": _query_digest(payload.query),
                }
            )
            raise

    return app
