"""AtomicPipeline — orchestrates Add-time extraction (workorder A1~A3).

One instance per service. Flow per Add request (after the raw commit):

  unit → cache lookup (LLM path) → on miss: budget check → extractor →
  finalize (cap/dedupe/backchannel, uniform for both extractor kinds) →
  cache store → derived rows into `atomic_units` (speaker-bound fact text).

玄神经 invariants (G37): this pipeline only READS raw unit content and only
WRITES the derived table. Every derived row keeps provenance
(`source_unit_id` + the `atom-<seq>-<k>` id). No code path here touches the
`memories` table.

A factless source message is recorded as a marker row (fact_index=0,
fact="") so "which sources were already extracted" stays a pure DB question
across restarts; markers are filtered out at index build.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aml_config.config import AMLConfig, canonical_endpoint_identity
from aml_store import AtomicUnitRecord, MemoryUnit
from aml_store.model import atomic_unit_id

from .cache import ExtractionCache, cache_key
from .extractors import LLMExtractor, RuleBasedExtractor, finalize_facts, prompt_version
from .spend import SpendGate

if TYPE_CHECKING:  # pragma: no cover
    from aml_adapter import AMLModelAdapter
    from aml_store.store import CompetitionMemoryStore


def _fact_text(role: str, fact: str) -> str:
    """Speaker-bound derived text — the same "role: content" form raw units
    are indexed with, so lexical/semantic signals stay uniform."""
    return f"{role}: {fact}"


class AtomicPipeline:
    def __init__(self, config: AMLConfig, adapter: AMLModelAdapter, store: CompetitionMemoryStore) -> None:
        self.config = config
        self.store = store
        atomic = config.atomic_cfg
        self.mode = str(atomic["mode"])
        self.extractor_kind = str(atomic["extractor"])
        self.granularity = str(atomic["granularity"])
        self.min_char_len = int(atomic["min_char_len"])
        self.max_per_message = int(atomic["max_per_message"])
        self.llm_concurrency = max(1, int(atomic.get("llm_concurrency", 8)))

        self.spend = SpendGate(
            str(config.adapter_cfg["spend_state_path"]),
            float(config.adapter_cfg["budget_usd_cap"]),
        )
        self.endpoint_identity = ""
        cache_dir = config.raw.get("extraction_cache", {}).get("dir", "06_results/atomic_cache")
        self.cache: ExtractionCache | None = None
        if self.extractor_kind == "llm":
            llm = LLMExtractor(
                adapter,
                self.granularity,
                max_tokens=int(atomic.get("llm_max_tokens", 500)),
                spend=self.spend,
            )
            self.extractor = llm
            self.model_id = llm.adapter.official_model_id
            # G45 (ADDENDUM-02 §3.1): cache identity is endpoint-bound. The
            # identity is the normalized hostname only — never keys or URLs
            # with parameters — so relay and official products can never share
            # a cache entry.
            self.endpoint_identity = canonical_endpoint_identity(llm.adapter.official_base_url)
            self.cache = ExtractionCache(f"{cache_dir}/extraction_cache.sqlite3")
        elif self.extractor_kind == "rule":
            self.extractor = RuleBasedExtractor(self.granularity)
            self.model_id = "rule"
        else:
            raise ValueError(f"unknown atomic.extractor: {self.extractor_kind!r}")

        # measured counters (ops journal / report evidence; never fabricated)
        self.stats: dict[str, int] = {"cache_hits": 0, "cache_misses": 0, "units_seen": 0, "facts": 0}

    # -- derivation ----------------------------------------------------------

    def _cache_key_for(self, unit: MemoryUnit) -> str | None:
        if self.cache is None:
            return None
        return cache_key(
            prompt_version(self.granularity),
            self.model_id,
            self.granularity,
            unit.content_sha256,
            self.endpoint_identity,
        )

    def _extract_one(self, unit: MemoryUnit) -> list[str]:
        """Extract facts for ONE raw unit (G38 input surface: role + content
        only). LLM path: cache → budget check → call → cache store."""
        key = self._cache_key_for(unit)
        if key is not None:
            cached = self.cache.get(key) if self.cache else None
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached
            self.stats["cache_misses"] += 1
        facts = self.extractor.extract(unit.role, unit.content)
        if key is not None and self.cache is not None:
            self.cache.put(
                key,
                facts,
                model_id=self.model_id,
                prompt_version=prompt_version(self.granularity),
                granularity=self.granularity,
                endpoint_identity=self.endpoint_identity,
                created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            )
        return facts

    def derive(self, units: list[MemoryUnit]) -> list[AtomicUnitRecord]:
        """Derived rows for the given raw units (caller persists them)."""
        if not units:
            return []
        self.stats["units_seen"] += len(units)
        if self.extractor_kind == "llm" and len(units) > 1 and self.llm_concurrency > 1:
            with ThreadPoolExecutor(max_workers=min(self.llm_concurrency, len(units))) as pool:
                per_unit = list(pool.map(self._extract_one, units))
        else:
            per_unit = [self._extract_one(unit) for unit in units]

        records: list[AtomicUnitRecord] = []
        for unit, raw_facts in zip(units, per_unit, strict=True):
            facts = finalize_facts(
                raw_facts, min_char_len=self.min_char_len, max_per_message=self.max_per_message
            )
            self.stats["facts"] += len(facts)
            if not facts:
                records.append(self._record(unit, fact_index=0, fact=""))
                continue
            for k, fact in enumerate(facts, start=1):
                records.append(self._record(unit, fact_index=k, fact=_fact_text(unit.role, fact)))
        return records

    def _record(self, unit: MemoryUnit, *, fact_index: int, fact: str) -> AtomicUnitRecord:
        return AtomicUnitRecord(
            atom_id=atomic_unit_id(unit.arrival_seq, fact_index),
            user_id=unit.user_id,
            source_unit_id=unit.id,
            source_content_sha256=unit.content_sha256,
            session_id=unit.session_id,
            msg_index=unit.msg_index,
            fact_index=fact_index,
            fact=fact,
            extractor=self.extractor.version,
            granularity=self.granularity,
            created_at=unit.created_at,
            ts_ms=unit.ts_ms,
        )

    # -- persistence + index feed ---------------------------------------------

    def extract_and_store(self, units: list[MemoryUnit]) -> list[AtomicUnitRecord]:
        """Derive + persist under the store's lock; returns the stored records
        (marker rows included). Idempotent on re-delivery (INSERT OR IGNORE)."""
        records = self.derive(units)
        self.store.insert_atomic_units(records)
        return records

    def pending_units(self, user_id: str, units: list[MemoryUnit]) -> list[MemoryUnit]:
        """Raw units with no derived row yet (e.g. a crash between the raw
        commit and extraction). Self-healed at index build; cache hits make
        the repair free."""
        done = self.store.atomic_source_ids(user_id)
        return [u for u in units if u.id not in done]

    def snapshot_stats(self) -> dict[str, Any]:
        return dict(self.stats)


def facts_digest(records: list[AtomicUnitRecord]) -> str:
    """Short digest of a derived-row batch — log/journal safe (no raw text)."""
    material = "\n".join(f"{r.atom_id}\x1f{r.fact}" for r in records)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
