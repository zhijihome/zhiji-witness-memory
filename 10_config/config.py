"""AML V0 configuration system.

The three return-surface knobs (RETURN_SIZE / ORDERING_POLICY / NEIGHBOR_POLICY,
Contract §5 AMENDMENT-01) are first-class config values here; they must never be
hardcoded at call sites and must be echoed into every ledger record.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RETURN_SIZE_TOP_K = "top_k"
ORDERING_POLICIES = ("relevance_desc", "temporal_asc")
NEIGHBOR_OFF = "off"

# Top-level config sections that never change retrieval or scoring behavior:
# where the service listens, where the db/journal live. Excluded from
# behavior_fingerprint() so a bench client and the live service can agree
# despite legitimately different db_path/port/ops destinations (G24).
# `extraction_cache` (V1): the cache *location* is infra like db_path; its
# *content* — the thing that can change behavior — is pinned separately via
# the EXTRACTION_CACHE_FP ledger column (workorder A2 / G36).
BEHAVIOR_EXCLUDED_KEYS = ("server", "storage", "ops_log", "extraction_cache", "competition")

ATOMIC_MODES = ("off", "both", "atomic_only", "atomic_index_raw_return")
ATOMIC_EXTRACTORS = ("rule", "llm")
ATOMIC_GRANULARITIES = ("coarse", "fine")
# A7 (ADDENDUM-01): packing-level provenance-family dedup. `off` = the
# pre-A7 behavior (raw unit and its own atomics may both occupy slots);
# keep_raw / keep_atomic pick the family representative by kind. Packing
# only — the store is never touched (G37).
ATOMIC_DEDUP_POLICIES = ("off", "keep_raw", "keep_atomic")
# V2 (WORKORDER_AML_V2_R1 §3.1): explicit family-aggregation discipline for
# the A7 dedup walk. `max` = the V1 implicit behavior (family rank = its best
# member's rank) and must stay bit-identical (G48); raw_only / support_gate /
# additive make the promotion policy explicit, with support_gate implementing
# evidence-bound promotion (an atomic fact can only lift a family whose source
# raw unit is itself visible in the fusion candidate pool).
ATOMIC_FAMILY_AGGS = ("max", "raw_only", "support_gate", "additive")

# EXP1 (ZT-20260919-13): governance state delta knobs. Default off — and the
# section is intentionally ABSENT from defaults.json so a governance-off run
# has a byte-identical behavior fingerprint to the frozen V1 A1-kr baseline.
# mode=on structurally requires the atomic facts layer (assertions are derived
# from atomic facts), and requires a real search policy — silently-ignored
# governance knobs are a hard config error, never a warning.
GOVERNANCE_MODES = ("off", "on")
GOVERNANCE_SEARCH_POLICIES = ("filter_suppressed", "prefer_current")
GOVERNANCE_FACT_KEYERS = ("rule-gk1",)

# EXP-U1 (2026-10-13): optional cross-encoder rerank stage. It re-orders the
# surface that ORDERING_POLICY already selected — the selection set itself is
# never changed (same select/order decoupling as H1/G30). Opt-in section:
# absent or "off" keeps the byte-identical behavior fingerprint of the frozen
# baseline.
# laya_rrf (EXP-L1): same two-stage RRF merge as ce_rrf, but relevance
# scores come from a remote Laya score server over HTTP instead of the
# local cross-encoder — identical merge semantics, swapped score source.
RERANK_MODES = ("off", "ce_rrf", "laya_rrf")
RERANK_DEVICES = ("auto", "cpu", "mps", "cuda")
# scope: "surface" re-orders the already-selected top-N (U1 semantics);
# "pool" RRF-rescores the top `pool_size` fused candidates BEFORE selection —
# it can change WHICH items are selected, not just their order (U4 probe).
RERANK_SCOPES = ("surface", "pool", "both")

# G44 (ADDENDUM-02): bench coordinates and the official-endpoint hard gate.
# OFFICIAL_* coordinates carry the contract's data-handling obligations
# (evaluation content must not be shared with third parties), so the LLM
# endpoint MUST be the official OpenAI endpoint — enforced as a hard config
# error, never a warning and never a silent fallback.
COORDINATES = ("LOCAL", "OFFICIAL_SMOKE", "OFFICIAL_FULL")
OFFICIAL_COORDINATES = ("OFFICIAL_SMOKE", "OFFICIAL_FULL")
OFFICIAL_ENDPOINT_HOST = "api.openai.com"


def canonical_endpoint_identity(base_url: str) -> str:
    """ADDENDUM-02 §3.1: endpoint identity = the URL's hostname, normalized.

    Only the hostname participates — never the scheme, path, query, userinfo
    or any credential material. Returns the lowercased host without port;
    raises ValueError when no host can be parsed.
    """
    from urllib.parse import urlsplit

    host = urlsplit(base_url.strip()).hostname
    if not host:
        raise ValueError(f"cannot derive endpoint identity from base_url: {base_url!r}")
    return host.lower()


def endpoint_allowed_for_coordinate(coordinate: str, base_url: str) -> bool:
    """G44 gate: OFFICIAL_* coordinates require the official endpoint host.
    LOCAL (and any non-official coordinate) is unconstrained."""
    if coordinate not in OFFICIAL_COORDINATES:
        return True
    return canonical_endpoint_identity(base_url) == OFFICIAL_ENDPOINT_HOST


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Typed views over the raw config tree


@dataclass(frozen=True)
class SurfaceConfig:
    """Return-surface knobs (Contract §5 AMENDMENT-01)."""

    RETURN_SIZE: str | int  # "top_k" or a positive int
    ORDERING_POLICY: str  # relevance_desc | temporal_asc
    NEIGHBOR_POLICY: str  # "off" | "window:N"

    def return_size_for(self, top_k: int) -> int:
        if self.RETURN_SIZE == RETURN_SIZE_TOP_K:
            return top_k
        return min(int(self.RETURN_SIZE), top_k)  # never exceed top_k

    def neighbor_window(self) -> int:
        if self.NEIGHBOR_POLICY == NEIGHBOR_OFF:
            return 0
        if self.NEIGHBOR_POLICY.startswith("window:"):
            try:
                return int(self.NEIGHBOR_POLICY.split(":", 1)[1])
            except ValueError:
                pass
        raise ConfigError(f"invalid NEIGHBOR_POLICY: {self.NEIGHBOR_POLICY!r}")


@dataclass(frozen=True)
class RetrieverFlags:
    exact: bool
    bm25: bool
    dense: bool

    def on(self) -> list[str]:
        return [r for r, flag in (("exact", self.exact), ("bm25", self.bm25), ("dense", self.dense)) if flag]


@dataclass(frozen=True)
class AMLConfig:
    raw: dict[str, Any] = field(repr=False)

    # -- surface ------------------------------------------------------------
    @property
    def surface(self) -> SurfaceConfig:
        s = self.raw["surface"]
        return SurfaceConfig(
            RETURN_SIZE=s["RETURN_SIZE"],
            ORDERING_POLICY=s["ORDERING_POLICY"],
            NEIGHBOR_POLICY=s["NEIGHBOR_POLICY"],
        )

    # -- retrieval ----------------------------------------------------------
    @property
    def retrievers(self) -> RetrieverFlags:
        r = self.raw["retrievers"]
        return RetrieverFlags(exact=bool(r["exact"]), bm25=bool(r["bm25"]), dense=bool(r["dense"]))

    @property
    def exact_cfg(self) -> dict[str, Any]:
        return self.raw["exact"]

    @property
    def bm25_cfg(self) -> dict[str, Any]:
        return self.raw["bm25"]

    @property
    def dense_cfg(self) -> dict[str, Any]:
        return self.raw["dense"]

    @property
    def fusion_cfg(self) -> dict[str, Any]:
        return self.raw["fusion"]

    @property
    def neighbor_cfg(self) -> dict[str, Any]:
        return self.raw["neighbor"]

    @property
    def temporal_cfg(self) -> dict[str, Any]:
        return self.raw["temporal"]

    @property
    def atomic_cfg(self) -> dict[str, Any]:
        return self.raw["atomic"]

    @property
    def governance_cfg(self) -> dict[str, Any]:
        return self.raw.get("governance", {})

    @property
    def competition_cfg(self) -> dict[str, Any]:
        return self.raw.get("competition", {})

    @property
    def competition_enabled(self) -> bool:
        return bool(self.competition_cfg.get("enabled", False))

    @property
    def rerank_cfg(self) -> dict[str, Any]:
        return self.raw.get("rerank", {})

    @property
    def extraction_cache_cfg(self) -> dict[str, Any]:
        return self.raw.get("extraction_cache", {})

    @property
    def search_cfg(self) -> dict[str, Any]:
        return self.raw["search"]

    # -- infra --------------------------------------------------------------
    @property
    def storage_cfg(self) -> dict[str, Any]:
        return self.raw["storage"]

    @property
    def server_cfg(self) -> dict[str, Any]:
        return self.raw["server"]

    @property
    def ops_cfg(self) -> dict[str, Any]:
        return self.raw["ops_log"]

    @property
    def adapter_cfg(self) -> dict[str, Any]:
        return self.raw["adapter"]

    # -- misc ---------------------------------------------------------------
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def behavior_fingerprint(self) -> str:
        """sha256 over the behavior-relevant subset of the raw config.

        This is the fingerprint both sides of a bench run must agree on
        (G24): the server exposes it via /health, the bench client computes
        the same over its own effective config and aborts on mismatch.
        Infra keys are excluded because they cannot change any retrieval or
        scoring behavior: the service may bind where it likes, use a
        throwaway db, or journal elsewhere, and `bench_default.json` carries
        a `_comment` key on the client side only.
        """
        pruned = {
            k: v
            for k, v in self.raw.items()
            if k not in BEHAVIOR_EXCLUDED_KEYS and k != "_comment"
        }
        return hashlib.sha256(
            json.dumps(pruned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()


# ---------------------------------------------------------------------------
# Loading / merging / validation


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_set_override(item: str) -> dict[str, Any]:
    """Parse `dotted.path=value` into a nested override dict."""
    if "=" not in item:
        raise ConfigError(f"--set expects dotted.path=value, got {item!r}")
    path, _, raw_value = item.partition("=")
    node: Any = _coerce_scalar(raw_value)
    for part in reversed(path.strip(".").split(".")):
        node = {part: node}
        if not part:
            raise ConfigError(f"empty path segment in --set {item!r}")
    return node


def merge_set_overrides(items: list[str]) -> dict[str, Any]:
    """Deep-merge a list of `--set dotted.path=value` items into one override tree.

    A shallow `{**a, **b}` chain silently drops earlier items under the same
    top-level key (e.g. `--set retrievers.bm25=false --set retrievers.dense=false`
    would lose the bm25 override), mislabeling exactly the kind of multi-knob
    ablation runs the experiment ledger exists to describe.
    """
    merged: dict[str, Any] = {}
    for item in items:
        merged = _deep_merge(merged, parse_set_override(item))
    return merged


def _coerce_scalar(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def validate(raw: dict[str, Any]) -> None:
    s = raw.get("surface", {})
    rs = s.get("RETURN_SIZE")
    if rs != RETURN_SIZE_TOP_K:
        if not isinstance(rs, int) or isinstance(rs, bool) or rs < 1:
            raise ConfigError(f"surface.RETURN_SIZE must be 'top_k' or int>=1, got {rs!r}")
    if s.get("ORDERING_POLICY") not in ORDERING_POLICIES:
        raise ConfigError(f"surface.ORDERING_POLICY must be one of {ORDERING_POLICIES}")
    np = s.get("NEIGHBOR_POLICY")
    if np != NEIGHBOR_OFF and not (isinstance(np, str) and np.startswith("window:")):
        raise ConfigError('surface.NEIGHBOR_POLICY must be "off" or "window:N"')
    if isinstance(np, str) and np.startswith("window:"):
        try:
            w = int(np.split(":", 1)[1])
            if w < 1:
                raise ValueError
        except ValueError:
            raise ConfigError(f"surface.NEIGHBOR_POLICY window must be int>=1, got {np!r}") from None
    flags = raw.get("retrievers", {})
    if not any(bool(flags.get(k)) for k in ("exact", "bm25", "dense")):
        raise ConfigError("at least one retriever must be enabled")
    dense = raw.get("dense", {})
    if dense.get("backend") not in ("hashed_ngram", "sentence_transformers", "dashscope"):
        raise ConfigError("dense.backend must be hashed_ngram, sentence_transformers or dashscope")
    if str(dense.get("device", "auto")) not in ("auto", "cpu", "mps", "cuda"):
        raise ConfigError("dense.device must be one of auto|cpu|mps|cuda")
    fusion = raw.get("fusion", {})
    if fusion.get("method") not in ("rrf", "norm_sum"):
        raise ConfigError("fusion.method must be 'rrf' or 'norm_sum'")
    if str(fusion.get("norm", "max")) not in ("max", "minmax"):
        raise ConfigError("fusion.norm must be 'max' or 'minmax'")
    tier = raw.get("adapter", {}).get("tier")
    if tier not in ("NO_LLM", "LOCAL_SURROGATE", "AML_OFFICIAL_MODEL"):
        raise ConfigError(f"adapter.tier must be a known tier, got {tier!r}")
    # G44 (ADDENDUM-02): coordinate must be known; OFFICIAL_* coordinates hard-
    # fail on any non-official LLM endpoint BEFORE the service can start (and
    # the adapter re-checks before every model call).
    coordinate = raw.get("adapter", {}).get("coordinate", "LOCAL")
    if coordinate not in COORDINATES:
        raise ConfigError(f"adapter.coordinate must be one of {COORDINATES}, got {coordinate!r}")
    base_url = str(raw.get("adapter", {}).get("official_base_url", "https://api.openai.com/v1"))
    try:
        allowed = endpoint_allowed_for_coordinate(str(coordinate), base_url)
    except ValueError as exc:
        raise ConfigError(f"adapter.official_base_url: {exc}") from None
    if not allowed:
        raise ConfigError(
            f"adapter.coordinate={coordinate} requires the official endpoint "
            f"host {OFFICIAL_ENDPOINT_HOST!r}, got {base_url!r} — refusing to "
            "start (evaluation content must not be sent to a third-party relay)"
        )

    atomic = raw.get("atomic", {})
    if atomic.get("mode", "off") not in ATOMIC_MODES:
        raise ConfigError(f"atomic.mode must be one of {ATOMIC_MODES}")
    if atomic.get("extractor", "rule") not in ATOMIC_EXTRACTORS:
        raise ConfigError(f"atomic.extractor must be one of {ATOMIC_EXTRACTORS}")
    if atomic.get("granularity", "coarse") not in ATOMIC_GRANULARITIES:
        raise ConfigError(f"atomic.granularity must be one of {ATOMIC_GRANULARITIES}")
    if atomic.get("dedup_policy", "off") not in ATOMIC_DEDUP_POLICIES:
        raise ConfigError(f"atomic.dedup_policy must be one of {ATOMIC_DEDUP_POLICIES}")
    # V2 (WORKORDER_AML_V2_R1 §3.1): family aggregation discipline. `max` is
    # always legal (it is the V1 default). Any other value is only meaningful
    # when the family walk actually runs — dedup enabled and a family-capable
    # atomic mode — so every other combination is a HARD config error, never a
    # silently-ignored knob.
    family_agg = atomic.get("family_agg", "max")
    if family_agg not in ATOMIC_FAMILY_AGGS:
        raise ConfigError(f"atomic.family_agg must be one of {ATOMIC_FAMILY_AGGS}, got {family_agg!r}")
    if family_agg != "max":
        if atomic.get("dedup_policy", "off") == "off":
            raise ConfigError(
                f"atomic.family_agg={family_agg!r} requires atomic.dedup_policy != 'off' "
                "(family aggregation is a property of the provenance-family dedup walk)"
            )
        if atomic.get("mode", "off") not in ("both", "atomic_only"):
            raise ConfigError(
                f"atomic.family_agg={family_agg!r} requires atomic.mode in "
                f"('both', 'atomic_only'), got {atomic.get('mode', 'off')!r}"
            )
    beta = atomic.get("family_support_beta", 0.5)
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or not 0.0 < float(beta) <= 1.0:
        raise ConfigError(f"atomic.family_support_beta must be a number in (0, 1], got {beta!r}")
    for key, low in (("max_per_message", 1), ("min_char_len", 1)):
        val = atomic.get(key, 8)
        if not isinstance(val, int) or isinstance(val, bool) or val < low:
            raise ConfigError(f"atomic.{key} must be int>={low}, got {val!r}")
    conc = atomic.get("llm_concurrency", 8)
    if not isinstance(conc, int) or isinstance(conc, bool) or conc < 1 or conc > 64:
        raise ConfigError(f"atomic.llm_concurrency must be int in [1, 64], got {conc!r}")
    # LLM extraction needs the sanctioned competition model tier: the surrogate
    # is not an LLM and would poison the science; NO_LLM forbids calls outright.
    if atomic.get("extractor") == "llm" and tier != "AML_OFFICIAL_MODEL":
        raise ConfigError(
            "atomic.extractor=llm requires adapter.tier=AML_OFFICIAL_MODEL "
            f"(got {tier!r}); the rule extractor is the NO_LLM path"
        )
    cap = raw.get("adapter", {}).get("budget_usd_cap", 30.0)
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
        raise ConfigError(f"adapter.budget_usd_cap must be a positive number, got {cap!r}")

    # EXP1 governance state delta (ZT-20260919-13): opt-in section, default off.
    governance = raw.get("governance", {})
    gov_mode = governance.get("mode", "off")
    if gov_mode not in GOVERNANCE_MODES:
        raise ConfigError(f"governance.mode must be one of {GOVERNANCE_MODES}, got {gov_mode!r}")
    if gov_mode == "on":
        if atomic.get("mode", "off") not in ("both", "atomic_only", "atomic_index_raw_return"):
            raise ConfigError(
                "governance.mode=on requires atomic.mode in "
                "('both', 'atomic_only', 'atomic_index_raw_return') "
                f"(assertions are derived from atomic facts), got {atomic.get('mode', 'off')!r}"
            )
        policy = governance.get("search_policy", "filter_suppressed")
        if policy not in GOVERNANCE_SEARCH_POLICIES:
            raise ConfigError(
                f"governance.search_policy must be one of {GOVERNANCE_SEARCH_POLICIES}, got {policy!r}"
            )
        keyer = governance.get("fact_keyer", "rule-gk1")
        if keyer not in GOVERNANCE_FACT_KEYERS:
            raise ConfigError(f"governance.fact_keyer must be one of {GOVERNANCE_FACT_KEYERS}, got {keyer!r}")

    # EXP-U1: optional cross-encoder rerank. Same discipline as governance:
    # an on-mode knob must actually take effect or the config is a hard
    # error, never a silently-ignored flag.
    rerank = raw.get("rerank", {})
    rmode = rerank.get("mode", "off")
    if rmode not in RERANK_MODES:
        raise ConfigError(f"rerank.mode must be one of {RERANK_MODES}, got {rmode!r}")
    if rmode != "off":
        rk = rerank.get("rrf_k", 60.0)
        if not isinstance(rk, (int, float)) or isinstance(rk, bool) or float(rk) <= 0:
            raise ConfigError(f"rerank.rrf_k must be a positive number, got {rk!r}")
        if str(rerank.get("device", "auto")) not in RERANK_DEVICES:
            raise ConfigError(f"rerank.device must be one of {RERANK_DEVICES}")
        if rmode == "ce_rrf" and not str(rerank.get("model", "")).strip():
            raise ConfigError("rerank.model is required when rerank.mode == 'ce_rrf'")
        if rmode == "laya_rrf" and not (
            str(rerank.get("score_url", "")).strip() or os.environ.get("LAYA_SCORE_URL", "").strip()
        ):
            raise ConfigError(
                "rerank.score_url or env LAYA_SCORE_URL is required when rerank.mode == 'laya_rrf'"
            )
        if str(rerank.get("scope", "surface")) not in RERANK_SCOPES:
            raise ConfigError(f"rerank.scope must be one of {RERANK_SCOPES}")
        ps = rerank.get("pool_size", 300)
        if str(rerank.get("scope", "surface")) in ("pool", "both"):
            if not isinstance(ps, int) or isinstance(ps, bool) or ps < 1:
                raise ConfigError(f"rerank.pool_size must be int>=1, got {ps!r}")


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AMLConfig:
    """defaults.json ← config file ← explicit overrides (last wins)."""
    defaults_file = Path(__file__).resolve().parent / "defaults.json"
    raw = json.loads(defaults_file.read_text(encoding="utf-8"))
    if path is not None:
        file_raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = _deep_merge(raw, file_raw)
    if overrides:
        raw = _deep_merge(raw, overrides)
    validate(raw)
    return AMLConfig(raw=raw)
