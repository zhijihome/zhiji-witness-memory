"""Benchmark runner orchestration (LOCAL coordinate only).

Drives a live service over HTTP exactly the way the platform would: adds each
conversation session as (chunked) Add requests, then answers each question with
one Search request plus the fixed surrogate answerer. Produces the Experiment
Ledger record (JSONL + CSV) and per-question details.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from aml_atomic.cache import fingerprint_of_db
from aml_atomic.extractors import RULE_VERSION, prompt_version
from aml_atomic.spend import SpendGate
from aml_config.config import AMLConfig, canonical_endpoint_identity

from .ledger import append_ledger_csv, append_ledger_jsonl, build_ledger_record
from .locomo_loader import CATEGORY_TO_AML_DIM, Conversation
from .scoring import evidence_recall, exact_match, token_f1
from .surrogate import surrogate_answer

VERSION = "0.2.0"

# V1 atomic ids embed their provenance: `atom-<source arrival seq>-<fact k>`.
# The harness maps a returned atomic entry to its source raw unit's dia_id —
# the honest LOCAL analog of "the gold evidence's content reached the answer
# surface (in derived form)". Applied uniformly to every arm; disclosed in the
# builder report.
_ATOMIC_ID_RE = re.compile(r"^atom-(\d{6,})-(\d{2,})$")


def _dia_of(unit_dia: dict[str, str], item_id: str) -> str:
    m = _ATOMIC_ID_RE.match(item_id)
    if m:
        item_id = f"mem-{int(m.group(1)):06d}"
    return unit_dia.get(item_id, "")


def get_git_commit() -> str:
    env = os.environ.get("AML_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


class ConfigMismatchError(RuntimeError):
    """Server's effective config disagrees with the client's bench config (G24)."""


def _config_knobs(config: AMLConfig, fingerprint: str) -> dict[str, Any]:
    return {
        "config_fingerprint": fingerprint,
        "retrievers_on": config.retrievers.on(),
        "RETURN_SIZE": config.surface.RETURN_SIZE,
        "ORDERING_POLICY": config.surface.ORDERING_POLICY,
        "NEIGHBOR_POLICY": config.surface.NEIGHBOR_POLICY,
        "dense_backend": str(config.dense_cfg.get("backend", "hashed_ngram")),
        "ATOMIC_MODE": str(config.atomic_cfg.get("mode", "off")),
    }


def _spend_total(config: AMLConfig) -> float | None:
    """Whole-round cumulative spend from the persisted gate state (client-side
    read of the same file the server writes). None = unreadable (honest)."""
    try:
        return round(
            SpendGate(
                str(config.adapter_cfg["spend_state_path"]),
                float(config.adapter_cfg["budget_usd_cap"]),
            ).current()["total_usd"],
            4,
        )
    except Exception:
        return None


def run_benchmark(
    client: httpx.Client,
    conversations: list[Conversation],
    config: AMLConfig,
    *,
    run_id: str,
    results_root: str | Path = "06_results",
    dataset_name: str = "locomo10.json (public proxy)",
    label: str = "",
    chunk_size: int = 20,
    top_k: int = 100,
    limit_questions: int | None = None,
    commit: str | None = None,
    write_ledger: bool = True,
    generic_roles: bool = False,
) -> dict[str, Any]:
    surface = config.surface
    results_root = Path(results_root)

    # G24 pre-flight: retrieval behavior is decided by the SERVER's config;
    # the ledger label comes from the CLIENT's. Compare behavior-relevant
    # fingerprints and abort before anything is written if they disagree.
    config_hash = config.behavior_fingerprint()
    health = client.get("/health")
    health.raise_for_status()
    health_body = health.json()
    server_tier = str(health_body.get("tier", "unknown"))
    server_config_hash = health_body.get("config_fingerprint")
    # G44/G46 (ADDENDUM-02): server-reported coordinate + LLM endpoint
    # identity + requested model id. Identity is the normalized hostname only
    # (no path/query/credentials). Fallbacks derive the same values from the
    # client config, which the G24 fingerprint equality below pins to the
    # server's — never a guess, never a fabricated endpoint name.
    server_coordinate = health_body.get("coordinate") or str(
        config.adapter_cfg.get("coordinate", "LOCAL")
    )
    if health_body.get("llm_endpoint") and server_tier != "NO_LLM":
        server_llm_endpoint = str(health_body["llm_endpoint"])
    else:
        server_llm_endpoint = "none" if server_tier == "NO_LLM" else canonical_endpoint_identity(
            str(config.adapter_cfg.get("official_base_url", "https://api.openai.com/v1"))
        )
    if health_body.get("llm_model_id") and server_tier != "NO_LLM":
        server_llm_model_id = str(health_body["llm_model_id"])
    else:
        server_llm_model_id = (
            "none" if server_tier == "NO_LLM" else str(config.adapter_cfg.get("official_model_id", ""))
        )
    if not isinstance(server_config_hash, str):
        raise ConfigMismatchError(
            "server /health does not expose config_fingerprint — the live service "
            "is older than this runner; restart it with the current serve.py"
        )
    if server_config_hash != config_hash:
        server_knobs = {
            "config_fingerprint": server_config_hash,
            "retrievers_on": health_body.get("retrievers_on"),
            "RETURN_SIZE": health_body.get("RETURN_SIZE"),
            "ORDERING_POLICY": health_body.get("ORDERING_POLICY"),
            "NEIGHBOR_POLICY": health_body.get("NEIGHBOR_POLICY"),
            "dense_backend": health_body.get("dense_backend"),
            "ATOMIC_MODE": health_body.get("ATOMIC_MODE", "(old server: not echoed)"),
        }
        client_knobs = _config_knobs(config, config_hash)
        lines = [
            "CONFIG MISMATCH between bench client and live service (G24) — "
            "aborting before any ledger/detail write.",
            "Retrieval behavior follows the SERVER (serve.py --set ...); the ledger "
            "label follows the CLIENT (run_bench.py --set ...).",
            "diff (client vs server):",
        ]
        for key, client_value in client_knobs.items():
            server_value = server_knobs.get(key)
            mark = "==" if client_value == server_value else "!="
            lines.append(f"  {key:20} {mark}  client={client_value!r}  server={server_value!r}")
        lines.append(
            "fix: pass the same behavior-relevant --set overrides to BOTH serve.py "
            "and run_bench.py (storage.db_path / server host-port / ops_log are exempt)."
        )
        raise ConfigMismatchError("\n".join(lines))

    # O2: measured usage aggregation. The service reports cumulative adapter
    # counters on /health; the ledger records the DELTA over this run. Under
    # NO_LLM these are real zeros (the adapter is never called), not constants.
    for field in ("model_calls", "tokens", "cost_usd", "atomic_units"):
        if health_body.get(field) is None:
            raise ConfigMismatchError(
                f"server /health does not expose usage counter {field!r} — the live "
                "service is older than this runner; restart it with the current serve.py"
            )
    usage_start = (
        int(health_body["model_calls"]),
        int(health_body["tokens"]),
        float(health_body["cost_usd"]),
        int(health_body["atomic_units"]),
        int(health_body.get("atomic_cache_hits", 0)),
        int(health_body.get("atomic_cache_misses", 0)),
        int(health_body.get("families_promoted_by_atomic", 0)),
        int(health_body.get("families_gate_blocked", 0)),
        int(health_body.get("orphan_promotions", 0)),
    )
    # V2 (WORKORDER_AML_V2_R1 §3.2): server-side echo of the family aggregation
    # in effect for this run. "unknown" only when the live service predates V2.
    family_agg_field = str(health_body.get("family_agg", "unknown"))
    dense_device = str(health_body.get("dense_device", "unknown"))

    detail_dir = results_root / "runs" / run_id
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / "questions.jsonl"

    add_latencies: list[float] = []
    search_latencies: list[float] = []
    failures: list[str] = []
    n_adds = 0
    n_searches = 0

    per_cat_f1: dict[int, list[float]] = {}
    per_cat_em: dict[int, list[float]] = {}
    all_f1: list[float] = []
    all_em: list[float] = []
    ev_recall_10: list[float] = []
    ev_recall_100: list[float] = []
    # A8 (ADDENDUM-01): return-face composition — how much evidence WIDTH the
    # surface really carries once atomic entries are resolved to their
    # provenance raw units
    face_raw: list[float] = []
    face_atomic: list[float] = []
    face_unique_dia: list[float] = []
    face_redundant: list[float] = []

    with open(detail_path, "w", encoding="utf-8") as detail_fh:
        for conv_i, conv in enumerate(conversations):
            user_id = f"bench:{run_id}:c{conv_i}"
            unit_dia: dict[str, str] = {}
            next_unit = 1
            # Official-payload shape: role carries generic chat roles, not the
            # participant name. Map each distinct speaker to a stable generic
            # role per conversation (first seen → user, second → assistant).
            role_map: dict[str, str] = {}

            def _role(speaker: str) -> str:
                if speaker not in role_map:
                    role_map[speaker] = "user" if len(role_map) % 2 == 0 else "assistant"
                return role_map[speaker]

            for sess_j, turns in enumerate(conv.sessions):
                for chunk_start in range(0, len(turns), chunk_size):
                    chunk = turns[chunk_start : chunk_start + chunk_size]
                    payload = {
                        "request_id": f"bench:{run_id}:c{conv_i}:s{sess_j}:k{chunk_start // chunk_size}",
                        "messages": [
                            {
                                "role": _role(t.speaker) if generic_roles else t.speaker,
                                "content": t.text,
                                "timestamp": t.ts_ms,
                            }
                            for t in chunk
                        ],
                        "user_id": user_id,
                        "session_id": f"bench:{run_id}:c{conv_i}:s{sess_j}",
                    }
                    started = time.perf_counter()
                    resp = client.post("/add", json=payload)
                    add_latencies.append((time.perf_counter() - started) * 1000.0)
                    n_adds += 1
                    if resp.status_code != 200:
                        failures.append(f"add c{conv_i}s{sess_j}: HTTP {resp.status_code}")
                        continue
                    for t in chunk:
                        unit_dia[f"mem-{next_unit:06d}"] = t.dia_id
                        next_unit += 1

            questions = conv.questions
            if limit_questions is not None:
                remaining = limit_questions - len(all_f1)
                if remaining <= 0:
                    break
                questions = questions[:remaining]
            for q in questions:
                started = time.perf_counter()
                resp = client.post("/search", json={"query": q.question, "user_id": user_id, "top_k": top_k})
                search_latencies.append((time.perf_counter() - started) * 1000.0)
                n_searches += 1
                if resp.status_code != 200:
                    failures.append(f"search c{conv_i} '{q.question[:40]}': HTTP {resp.status_code}")
                    continue
                items = resp.json().get("data", [])
                contents = [it["content"] for it in items]
                returned_dia = [_dia_of(unit_dia, it["id"]) for it in items]
                prediction = surrogate_answer(q.question, None, contents)
                f1 = token_f1(prediction, q.answer)
                em = exact_match(prediction, q.answer)
                r10 = evidence_recall(
                    [d for d in returned_dia[:10] if d], list(q.evidence)
                )
                r100 = evidence_recall([d for d in returned_dia if d], list(q.evidence))
                all_f1.append(f1)
                all_em.append(em)
                ev_recall_10.append(r10)
                ev_recall_100.append(r100)
                dias = [d for d in returned_dia if d]
                n_atomic = sum(1 for it in items if _ATOMIC_ID_RE.match(it["id"]))
                face_raw.append(float(len(items) - n_atomic))
                face_atomic.append(float(n_atomic))
                face_unique_dia.append(float(len(set(dias))))
                face_redundant.append(float(len(dias) - len(set(dias))))
                per_cat_f1.setdefault(q.category, []).append(f1)
                per_cat_em.setdefault(q.category, []).append(em)
                detail_fh.write(
                    json.dumps(
                        {
                            "conv": conv_i,
                            "sample_id": conv.sample_id,
                            "question": q.question,
                            "gold": q.answer,
                            "prediction": prediction,
                            "category": q.category,
                            "f1": f1,
                            "em": em,
                            "evidence_recall_at_10": r10,
                            "evidence_recall_at_100": r100,
                            "n_returned": len(items),
                            "top_returned": [
                                {"id": it["id"], "score": it.get("score"), "dia": _dia_of(unit_dia, it["id"])}
                                for it in items[:10]
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    dim_f1: dict[str, list[float]] = {}
    for cat, values in per_cat_f1.items():
        dim = CATEGORY_TO_AML_DIM.get(cat)
        if dim:
            dim_f1.setdefault(dim, []).extend(values)

    memory_only = [v for cat, values in per_cat_f1.items() if cat in (1, 2, 3) for v in values]

    end_health = client.get("/health")
    end_health.raise_for_status()
    end_body = end_health.json()
    model_calls = max(0, int(end_body.get("model_calls", usage_start[0])) - usage_start[0])
    tokens = max(0, int(end_body.get("tokens", usage_start[1])) - usage_start[1])
    cost = max(0.0, float(end_body.get("cost_usd", usage_start[2])) - usage_start[2])
    atomic_units_delta = max(0, int(end_body.get("atomic_units", usage_start[3])) - usage_start[3])
    cache_hits_delta = max(0, int(end_body.get("atomic_cache_hits", usage_start[4])) - usage_start[4])
    cache_misses_delta = max(0, int(end_body.get("atomic_cache_misses", usage_start[5])) - usage_start[5])
    promotions_delta = max(0, int(end_body.get("families_promoted_by_atomic", usage_start[6])) - usage_start[6])
    gate_blocked_delta = max(0, int(end_body.get("families_gate_blocked", usage_start[7])) - usage_start[7])
    orphan_delta = max(0, int(end_body.get("orphan_promotions", usage_start[8])) - usage_start[8])

    dense_backend = str(config.dense_cfg.get("backend", "hashed_ngram"))
    if dense_backend == "sentence_transformers":
        dense_model = (
            f"{config.dense_cfg.get('sentence_transformers_model', '')}"
            f"@{config.dense_cfg.get('sentence_transformers_revision', '')[:12]}"
        )
    elif dense_backend == "dashscope":
        dense_model = (
            f"{config.dense_cfg.get('api_model', '')}"
            f"@{config.dense_cfg.get('api_dimensions', '')}d"
        )
    else:
        dense_model = ""

    metrics: dict[str, Any] = {
        "VERSION": VERSION,
        "COMMIT": commit or get_git_commit(),
        "CONFIG": config_hash,
        "SERVER_CONFIG": server_config_hash,
        "MODEL": f"tier={server_tier};bm25=internal;dense={dense_backend};exact=internal",
        "COORDINATE": server_coordinate,
        "DATASET": dataset_name,
        "RETRIEVERS_ON": ",".join(config.retrievers.on()),
        "RETURN_SIZE": surface.RETURN_SIZE,
        "ORDERING_POLICY": surface.ORDERING_POLICY,
        "NEIGHBOR_POLICY": surface.NEIGHBOR_POLICY,
        "OVERALL": _mean(all_f1),
        "FACT": _mean(dim_f1.get("FACT", [])),
        "TEMPORAL": _mean(dim_f1.get("TEMPORAL", [])),
        "GOVERNANCE": None,  # no LOCAL proxy probe in V0 — honest null
        "MULTI_HOP": _mean(dim_f1.get("MULTI_HOP", [])),
        "PERSONALIZATION": None,
        "RULE": None,
        "SAFETY": None,
        "LATENCY_P50": round(_percentile(sorted(search_latencies), 0.50), 3),
        "LATENCY_P95": round(_percentile(sorted(search_latencies), 0.95), 3),
        "MODEL_CALLS": model_calls,
        "TOKENS": tokens,
        "COST": round(cost, 8),
        "FAILURE_CASES": len(failures),
        "RUN_ID": run_id,
        "TS": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "LABEL": label,
        "N_CONVS": len(conversations),
        "N_QUESTIONS": len(all_f1),
        "N_ADDS": n_adds,
        "N_SEARCHES": n_searches,
        "ADD_LATENCY_P50": round(_percentile(sorted(add_latencies), 0.50), 3),
        "ADD_LATENCY_P95": round(_percentile(sorted(add_latencies), 0.95), 3),
        "EM_OVERALL": _mean(all_em),
        "MEMORY_ONLY_OVERALL": _mean(memory_only),
        "EVIDENCE_RECALL_AT_10": _mean(ev_recall_10),
        "EVIDENCE_RECALL_AT_100": _mean(ev_recall_100),
        "LOCOMO_CAT1_F1": _mean(per_cat_f1.get(1, [])),
        "LOCOMO_CAT2_F1": _mean(per_cat_f1.get(2, [])),
        "LOCOMO_CAT3_F1": _mean(per_cat_f1.get(3, [])),
        "LOCOMO_CAT4_F1": _mean(per_cat_f1.get(4, [])),
        "LOCOMO_CAT5_F1": _mean(per_cat_f1.get(5, [])),
        "SURROGATE_ANSWERER": "fixed-extractive-v1 (deterministic, no LLM, no gold)",
        "SERVER_TIER": server_tier,
        "DENSE_DEVICE": dense_device,
        "DENSE_MODEL": dense_model,
        "TOP_K": top_k,
        "DETAIL_PATH": str(detail_path),
    }

    # V1 Atomic ledger fields (workorder A2/A4/G36/G43). Extractor/prompt
    # version and the extraction-cache fingerprint go into EVERY row so the
    # Judge can verify "same extraction products" across arms.
    atomic = config.atomic_cfg
    atomic_mode = str(atomic.get("mode", "off"))
    extractor_kind = str(atomic.get("extractor", "rule"))
    granularity = str(atomic.get("granularity", "coarse"))
    if atomic_mode == "off":
        prompt_ver = ""
    elif extractor_kind == "llm":
        prompt_ver = prompt_version(granularity)
    else:
        prompt_ver = f"{RULE_VERSION}-{granularity}"
    # G46 (ADDENDUM-02 §3.2): explicit, machine-readable LLM provenance on
    # every row. "none" is the stable empty-value semantics for rows with no
    # LLM in play (mode=off, or rule extraction) — never an empty guess.
    # PROMPT_VERSION mirrors ATOMIC_PROMPT_VERSION; the dedicated columns
    # exist so endpoint/model are first-class fields, not config archaeology.
    if atomic_mode == "off" or extractor_kind != "llm":
        llm_endpoint_field = "none"
        llm_model_field = "none"
        prompt_field = prompt_ver if atomic_mode != "off" else "none"
    else:
        llm_endpoint_field = server_llm_endpoint
        llm_model_field = server_llm_model_id
        prompt_field = prompt_ver
    metrics.update(
        {
            "FACE_RAW_COUNT": _mean(face_raw),
            "FACE_ATOMIC_COUNT": _mean(face_atomic),
            "FACE_UNIQUE_DIA": _mean(face_unique_dia),
            "FACE_REDUNDANT_SLOTS": _mean(face_redundant),
            "ATOMIC_MODE": atomic_mode,
            "ATOMIC_EXTRACTOR": "none" if atomic_mode == "off" else extractor_kind,
            "ATOMIC_GRANULARITY": "" if atomic_mode == "off" else granularity,
            "ATOMIC_PROMPT_VERSION": prompt_ver,
            "ATOMIC_UNITS": atomic_units_delta,
            "ATOMIC_CACHE_HITS": cache_hits_delta,
            "ATOMIC_CACHE_MISSES": cache_misses_delta,
            "LLM_ENDPOINT": llm_endpoint_field,
            "LLM_MODEL_ID": llm_model_field,
            "PROMPT_VERSION": prompt_field,
            # V2 promotion audit (WORKORDER_AML_V2_R1 §3.2): per-run measured
            # deltas of the server's cumulative promotion counters. Scope:
            # promoted/orphan count SELECTED families (a family occupying a
            # slot thanks to its atomic best outranking its raw best / with no
            # in-pool raw at all); gate_blocked counts pool families the
            # support_gate barred from selection entirely.
            "FAMILY_AGG": family_agg_field,
            "FAMILIES_PROMOTED_BY_ATOMIC": promotions_delta,
            "FAMILIES_GATE_BLOCKED": gate_blocked_delta,
            "ORPHAN_PROMOTIONS": orphan_delta,
            "EXTRACTION_CACHE_FP": (
                fingerprint_of_db(
                    Path(str(config.extraction_cache_cfg.get("dir", "06_results/atomic_cache")))
                    / "extraction_cache.sqlite3",
                    granularity=granularity,
                )
                if extractor_kind == "llm" and atomic_mode != "off"
                else "n/a"
            ),
            "SPEND_TOTAL_USD": _spend_total(config),
        }
    )
    record = build_ledger_record(metrics)

    if failures:
        (detail_dir / "failures.txt").write_text("\n".join(failures[:200]), encoding="utf-8")

    if write_ledger:
        append_ledger_jsonl(results_root / "ledger" / "ledger.jsonl", record)
        append_ledger_csv(results_root / "ledger" / "ledger.csv", record)
    return record
