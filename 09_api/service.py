"""MemoryService — wires store (01) + retrieval engine (02) + adapter (00)
+ the V1 atomic extraction layer (11).

One instance per process. A single reentrant lock serializes add/search so the
in-memory SearchIndex structures never mutate mid-query (V0 runs one uvicorn
worker; README documents this).

V1 Atomic (workorder A1/A4): when `atomic.mode != off`, every Add extracts
derived atomic facts AFTER the raw commit and BEFORE the 200 — synchronous
persistence covers the derived layer too. `mode=off` constructs no pipeline at
all: the atomic layer is fully bypassed (no extractor, no cache, no derived
writes — G42).
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from aml_adapter import AMLModelAdapter
from aml_atomic import AtomicPipeline
from aml_config.config import AMLConfig
from aml_retrieval import DerivedDoc, PackedItem, RetrievalEngine, SearchIndex, SearchOutcome
from aml_store import AddOutcome, AddPayload, AtomicUnitRecord, CompetitionMemoryStore

_INDEX_CACHE_LIMIT = 64


class MemoryService:
    def __init__(self, config: AMLConfig) -> None:
        self.config = config
        storage = config.storage_cfg
        self.store = CompetitionMemoryStore(
            db_path=storage.get("db_path", "06_results/aml.db"),
            journal_mode=storage.get("journal_mode", "WAL"),
            synchronous=storage.get("synchronous", "FULL"),
        )
        self.engine = RetrievalEngine(config)
        self.adapter = AMLModelAdapter.from_config(config.adapter_cfg)
        self.atomic_mode = str(config.atomic_cfg.get("mode", "off"))
        # derived-only search surfaces index postings from atomic docs alone
        self._derived_only_surface = self.atomic_mode in ("atomic_only", "atomic_index_raw_return")
        self.pipeline = AtomicPipeline(config, self.adapter, self.store) if self.atomic_mode != "off" else None
        # EXP1 governance state delta (ZT-20260919-13): constructed only when
        # governance.mode=on; the off path never imports the layer (G42-style
        # full bypass — no derivation, no journal writes, no state).
        self.governance = None
        if str(config.governance_cfg.get("mode", "off")) == "on":
            from aml_governance.engine import GovernanceEngine

            self.governance = GovernanceEngine(config, self.store)
        self._lock = threading.RLock()
        self._search_indexes: dict[str, SearchIndex] = {}

    # -- write path -----------------------------------------------------------

    def add(self, payload: AddPayload) -> AddOutcome:
        with self._lock:
            outcome = self.store.add(payload)  # raw commit (INV5); raw is canonical, never mutated
            new_records: list[AtomicUnitRecord] = []
            if outcome.stored and self.pipeline is not None:
                new_records = self.pipeline.extract_and_store(outcome.stored)  # derived layer only (G37)
            if outcome.stored and self.governance is not None:
                # EXP1: governance assertions + journal ops AFTER the atomic
                # derivation, BEFORE the 200 — synchronous persistence covers
                # the governance layer too; the folded state updates in place,
                # so the live SearchIndex view transitions incrementally.
                self.governance.ingest_atomic(payload.user_id, new_records)
            if outcome.stored:
                si = self._search_indexes.get(payload.user_id)
                if si is None:
                    si = self._build_index(payload.user_id)
                else:
                    embedder = self.engine.index_embedder()
                    si.extend(outcome.stored, embedder, with_postings=not self._derived_only_surface)
                    if new_records:
                        existing = {d.doc_id for d in si.derived_docs}
                        si.extend_derived(
                            self._to_docs(payload.user_id, [r for r in new_records if r.atom_id not in existing]),
                            embedder,
                        )
            return outcome

    def _to_docs(self, user_id: str, records: list[AtomicUnitRecord]) -> list[DerivedDoc]:
        """Convert derived rows into retrieval docs. Provenance is structural:
        every row must resolve to a live raw unit (integrity violation is a
        loud error, never a silently dropped fact)."""
        if not records:
            return []
        ui = self.store.get_user_index(user_id)
        docs: list[DerivedDoc] = []
        for r in records:
            if r.fact == "":
                continue  # factless marker row
            source_idx = ui.by_id.get(r.source_unit_id)
            if source_idx is None:
                raise RuntimeError(
                    f"atomic provenance broken: {r.atom_id} points at missing raw unit {r.source_unit_id}"
                )
            docs.append(
                DerivedDoc(
                    doc_id=r.atom_id,
                    text=r.fact,
                    created_at=r.created_at,
                    ts_ms=r.ts_ms,
                    source_idx=source_idx,
                    source_unit_id=r.source_unit_id,
                )
            )
        return docs

    def _build_index(self, user_id: str) -> SearchIndex:
        ui = self.store.get_user_index(user_id)
        derived_docs: list[DerivedDoc] | None = None
        if self.pipeline is not None:
            pending = self.pipeline.pending_units(user_id, ui.units)
            if pending:
                # self-heal (crash between raw commit and extraction); with the
                # extraction cache this is free for LLM arms
                self.pipeline.extract_and_store(pending)
            derived_docs = self._to_docs(user_id, self.store.atomic_units(user_id))
        si = self.engine.build_index(ui, self.engine.index_embedder(), derived_docs)
        if self.governance is not None:
            # EXP1: idempotent governance self-heal/backfill — atomic rows that
            # never reached the governance journal (crash between the atomic
            # store and the governance ingest, or facts extracted before
            # governance was enabled) are derived here; already-journaled
            # assertion ids are skipped by the engine.
            self.governance.ingest_atomic(user_id, self.store.atomic_units(user_id))
            # attach the live governance view; it wraps the engine's
            # GovernanceState, so later add-time folds are visible to search
            # on this index without any rebuild (incremental transition).
            si.governance = self.governance.view_for(user_id)
        self._search_indexes[user_id] = si
        while len(self._search_indexes) > _INDEX_CACHE_LIMIT:
            self._search_indexes.pop(next(iter(self._search_indexes)))
        return si

    # -- read path ------------------------------------------------------------

    def search(self, user_id: str, query: str, top_k: int) -> SearchOutcome:
        with self._lock:
            si = self._search_indexes.get(user_id)
            if si is None or (
                not self._derived_only_surface and len(si.ui.units) != si.processed
            ):  # defensive; should not happen
                si = self._build_index(user_id)
            outcome = self.engine.search(si, query, top_k)
            date_on = self.engine.config.search_cfg.get("date_prefix")
            speaker_on = self.engine.config.search_cfg.get("speaker_prefix")
            if date_on or speaker_on:
                def _render(it: PackedItem) -> PackedItem:
                    text = it.content
                    if speaker_on and it.role:
                        text = f"{it.role}: {text}"
                    if date_on and it.ts_ms is not None:
                        text = f"[{datetime.fromtimestamp(it.ts_ms / 1000.0, tz=UTC).strftime('%Y-%m-%d')}] {text}"
                    return replace(it, content=text) if text != it.content else it
                outcome = replace(outcome, items=[_render(it) for it in outcome.items])
            return outcome

    # -- lifecycle -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def close(self) -> None:
        self.store.close()
        self.adapter.close()
        if self.pipeline is not None and self.pipeline.cache is not None:
            self.pipeline.cache.close()
