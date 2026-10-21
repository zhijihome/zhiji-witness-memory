"""Retrieval engine (architecture layer 02, core).

Pipeline per search (deterministic end-to-end, G6):

  retrievers (exact / bm25 / dense, each independently toggleable)
    → fusion: "rrf" (rank-based; only integer ranks feed the score, which makes
      the fused score platform-stable) or "norm_sum" (CombMNZ over per-channel
      max-normalized scores — parameter-free, so it cannot be overfit)
    → optional small additive temporal-anchor boost
    → optional neighbor expansion over session-adjacent units
    → SELECT top-N purely by fused relevance with fixed tie-break (-score,
      arrival order), N = return_size_for(top_k)   ← V0.5 (H1): selection and
      ordering are decoupled — ORDERING_POLICY can only re-rank the selected
      surface, never change which items are on it (G30)
    → ORDER the selected N items by ORDERING_POLICY
    → packed items: exactly {id, content, score, created_at}

The three return-surface knobs (RETURN_SIZE / ORDERING_POLICY / NEIGHBOR_POLICY)
come from config; the engine records what it applied in `SearchOutcome.trace`
so every layer above (API ops log, bench ledger) can echo them verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aml_config.config import ConfigError
from aml_store.model import MemoryUnit, speaker_label
from aml_store.store import UserIndex

from .bm25 import BM25Retriever
from .dense import DenseRetriever, build_embedder, embed_content_for_index, resolve_embedding_device
from .exact import ExactRetriever
from .temporal import TimeAnchor, extract_anchors, query_anchors, unit_anchor_from_timestamp
from .tokenize import content_tokens, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from aml_config.config import AMLConfig

# V1 Atomic: derived docs occupy a disjoint doc-idx range far above any
# realistic raw index, so one postings space can hold both segments (same-pool
# retrieval for `both` mode) without colliding when new raw units arrive.
DERIVED_BASE = 1 << 22

ATOMIC_RETURN_MODES = ("atomic_only", "atomic_index_raw_return")  # derived-only search surface


def _index_text(unit: MemoryUnit) -> str:
    """Raw-unit index text: "speaker: content" when the role carries a real
    participant name, bare content under generic chat roles."""
    label = speaker_label(unit.role)
    return f"{label}: {unit.content}" if label else unit.content


@dataclass(frozen=True)
class DerivedDoc:
    """One derived (atomic) retrieval document. `text` is the speaker-bound
    fact; provenance is structural: doc_id embeds the source raw unit's
    arrival seq, and source_idx/source_unit_id point back into the raw
    UserIndex. Derived docs are never part of the raw canonical store."""

    doc_id: str
    text: str
    created_at: str
    ts_ms: int | None
    source_idx: int
    source_unit_id: str


@dataclass(frozen=True)
class PackedItem:
    idx: int
    via: str  # "direct" | "neighbor" | "atomic"
    id: str
    content: str
    score: float
    created_at: str
    ts_ms: int | None = None
    role: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """Exactly the four contract-visible fields (Contract §5)."""
        return {"id": self.id, "content": self.content, "score": round(self.score, 6), "created_at": self.created_at}


@dataclass(frozen=True)
class SearchOutcome:
    items: list[PackedItem]
    trace: dict[str, Any] = field(default_factory=dict)


def fuse_rrf(
    ranked: dict[str, list[int]], rrf_k: float = 60.0, weights: dict[str, float] | None = None
) -> dict[int, float]:
    """Classic reciprocal-rank fusion — the V0 semantics, unchanged bit for bit
    (integer ranks only, channel iteration in insertion order)."""
    weights = weights or {}
    fused: dict[int, float] = {}
    for name, order in ranked.items():
        w = float(weights.get(name, 1.0))
        for pos, idx in enumerate(order, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (rrf_k + pos)
    return fused


def fuse_norm_sum(
    channel_scores: dict[str, dict[int, float]], norm: str = "max"
) -> dict[int, float]:
    """CombMNZ fusion (Fox & Shaw 1994): per-channel normalized scores, summed,
    times the number of channels that retrieved the document. The normalization
    is over the channel's own hit list for this query, so the fusion carries no
    tunable constant at all — that is the point (H3 anti-overfitting discipline).

    norm="max": score / channel max (default; rank-preserving within channel).
    norm="minmax": (score - min) / (max - min); a single-hit channel maps to 1.
    """
    contrib: dict[int, float] = {}
    count: dict[int, int] = {}
    for scores in channel_scores.values():
        if not scores:
            continue
        s_max = max(scores.values())
        s_min = min(scores.values())
        for idx, s in scores.items():
            if norm == "minmax":
                v = 1.0 if s_max <= s_min else (s - s_min) / (s_max - s_min)
            elif norm == "max":
                v = s / s_max
            else:
                raise ConfigError(f"unknown fusion.norm: {norm!r}")
            contrib[idx] = contrib.get(idx, 0.0) + v
            count[idx] = count.get(idx, 0) + 1
    return {idx: count[idx] * contrib[idx] for idx in contrib}


class SearchIndex:
    """Per-user retrieval structures, append-only, kept in sync with a UserIndex.

    V1: `doc_len` and `anchors` are idx-keyed dicts because the postings space
    now spans two segments — raw units at idx < DERIVED_BASE, derived atomic
    docs at idx >= DERIVED_BASE. Values and float math are unchanged vs the
    V0.5 list form (same corpus → same numbers, G41)."""

    __slots__ = (
        "ui",
        "bm25_postings",
        "doc_len",
        "total_len",
        "dense_postings",
        "anchors",
        "temporal_enabled",
        "stemming",
        "_processed",
        "derived_docs",
        "derived_by_source",
        # EXP1 governance state delta (ZT-20260919-13): optional GovernanceView
        # attached by the service when governance.mode=on. Always None on the
        # V1 path — the engine's behavior is bit-identical whether this slot
        # is None or simply never read.
        "governance",
    )

    def __init__(
        self, user_index: UserIndex, temporal_enabled: bool = True, stemming: bool = True
    ) -> None:
        self.ui = user_index
        self.bm25_postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: dict[int, int] = {}
        self.total_len = 0
        self.dense_postings: dict[int, list[tuple[int, float]]] = {}
        self.anchors: dict[int, list[TimeAnchor]] = {}
        self.temporal_enabled = temporal_enabled
        self.stemming = stemming
        self._processed = 0
        self.derived_docs: list[DerivedDoc] = []
        self.derived_by_source: dict[int, list[int]] = {}
        self.governance: Any | None = None  # EXP1: GovernanceView when enabled

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def processed(self) -> int:
        """Number of units whose postings/anchors have been built."""
        return self._processed

    @property
    def avgdl(self) -> float:
        return self.total_len / len(self.doc_len) if self.doc_len else 0.0

    # -- incremental maintenance -------------------------------------------

    def extend(self, units: list[MemoryUnit], embedder: Any, *, with_postings: bool = True) -> None:
        """embedder may be None when the dense retriever is disabled — dense
        postings are then left empty and the write path skips embedding.
        with_postings=False maintains only the raw UserIndex view (the derived-
        only search modes index postings from atomic docs alone; the raw units
        stay canonical in the store and are never deleted)."""
        embed_many = getattr(embedder, "embed_many", None)
        if embedder is not None and with_postings and embed_many is not None:
            # EXP-EMB-V4: batch-warm an API backend's cache; the per-unit
            # `embed` below then hits cache and posting order is unchanged.
            embed_many([_index_text(unit) for unit in units])
        for unit in units:
            if unit.user_id != self.ui.user_id:
                raise ValueError("cross-user extend blocked")
            already_present = unit.id in self.ui.by_id
            if not already_present:
                if unit.arrival_seq != len(self.ui.units) + 1:
                    # units must arrive in arrival order; rebuild is the safe path
                    raise RuntimeError(f"SearchIndex out of sync at seq {unit.arrival_seq}")
                self.ui.append(unit)
            idx = self.ui.by_id[unit.id]
            if not with_postings or idx < self._processed:
                continue  # postings already built for this unit

            # BM25 sees "speaker: content" when the role carries a real
            # participant name — a first-class lexical signal in
            # two-participant conversations (dense embeds the same text).
            tf: dict[str, int] = {}
            for t in content_tokens(_index_text(unit), self.stemming):
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self.bm25_postings.setdefault(t, []).append((idx, f))
            self.doc_len[idx] = sum(tf.values())
            self.total_len += sum(tf.values())

            if embedder is not None:
                vec = embed_content_for_index(embedder, unit.role, unit.content)
                for bucket, val in vec.items():
                    self.dense_postings.setdefault(bucket, []).append((idx, val))

            if self.temporal_enabled:
                unit_anchors = extract_anchors(unit.content, unit.ts_ms)
                if unit.ts_ms is not None:
                    unit_anchors.append(unit_anchor_from_timestamp(unit.ts_ms))
                self.anchors[idx] = unit_anchors
            else:
                self.anchors[idx] = []

            self._processed = idx + 1

    def extend_derived(self, docs: list[DerivedDoc], embedder: Any) -> None:
        """Append derived (atomic) docs to the SAME postings pool: BM25 corpus
        statistics and dense space span raw + derived docs together (the
        `both`-mode same-pool semantics). Derived docs inherit temporal anchors
        from their own text + the source unit's timestamp."""
        embed_many = getattr(embedder, "embed_many", None)
        if embedder is not None and embed_many is not None:
            embed_many([doc.text for doc in docs])
        for doc in docs:
            idx = DERIVED_BASE + len(self.derived_docs)
            self.derived_docs.append(doc)

            tf: dict[str, int] = {}
            for t in content_tokens(doc.text, self.stemming):
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self.bm25_postings.setdefault(t, []).append((idx, f))
            self.doc_len[idx] = sum(tf.values())
            self.total_len += sum(tf.values())

            if embedder is not None:
                vec = embedder.embed(doc.text)  # speaker already bound in the fact text
                for bucket, val in vec.items():
                    self.dense_postings.setdefault(bucket, []).append((idx, val))

            if self.temporal_enabled:
                doc_anchors = extract_anchors(doc.text, doc.ts_ms)
                if doc.ts_ms is not None:
                    doc_anchors.append(unit_anchor_from_timestamp(doc.ts_ms))
                self.anchors[idx] = doc_anchors
            else:
                self.anchors[idx] = []

            self.derived_by_source.setdefault(doc.source_idx, []).append(idx)

    @classmethod
    def build(
        cls,
        user_index: UserIndex,
        config: AMLConfig,
        embedder: Any,
        derived_docs: list[DerivedDoc] | None = None,
    ) -> SearchIndex:
        mode = str(config.atomic_cfg.get("mode", "off"))
        derived_only_surface = mode in ATOMIC_RETURN_MODES
        index = cls(
            user_index,
            temporal_enabled=bool(config.temporal_cfg.get("extract_anchors", True)),
            stemming=bool(config.raw.get("text", {}).get("stemming", True)),
        )
        index.extend(user_index.units, embedder, with_postings=not derived_only_surface)
        if mode != "off" and derived_docs:
            index.extend_derived([d for d in derived_docs if d.text], embedder)
        return index


def _select_by_family_key(
    families: dict[int, list[int]],
    results: dict[int, float],
    family_agg: str,
    beta: float,
    family_rep,
    return_size: int,
) -> list[tuple[int, float]]:
    """V2 explicit family-aggregation selection (WORKORDER_AML_V2_R1 §3.1).

    Each pooled family gets one explicit key from its best raw/atomic pooled
    scores; families are selected best-key-first and represented by the dedup
    policy's preferred member. Key per discipline (raw_best / atomic_best =
    the member's pooled score; None = kind absent from the pool):

      raw_only     key = raw_best                    (raw absent → not selectable)
      support_gate key = max(raw_best, atomic_best)  (raw absent → NOT selectable:
                   evidence-bound promotion — an atomic fact may only lift a
                   family whose source raw evidence is itself in the pool)
      additive     key = raw_best + beta * atomic_best (raw absent →
                   beta * atomic_best: discounted entry, audited as orphan)
      max          key = max(raw_best, atomic_best)  (raw absent → atomic_best;
                   the V1 implicit rule, kept callable here for the audit /
                   determinism harnesses — the engine's default path walks the
                   identical order in its original bit-exact form)

    Ties break by the family's minimum member idx (the raw idx when present,
    raw idxs < DERIVED_BASE ≤ atomic idxs), compatible with the V0
    (-score, idx) total order. Deterministic, no LLM, no benchmark signals.
    """
    ranked: list[tuple[float, int, int]] = []  # (-key, family_min_idx, fam)
    for fam, members in families.items():
        raw_best = max((results[i] for i in members if i < DERIVED_BASE), default=None)
        atomic_best = max((results[i] for i in members if i >= DERIVED_BASE), default=None)
        if family_agg == "raw_only":
            key = raw_best
        elif family_agg in ("max", "support_gate"):
            if raw_best is None:
                if family_agg == "support_gate":
                    key = None  # gate: no in-pool raw evidence, no promotion
                else:
                    key = atomic_best
            elif atomic_best is None:
                key = raw_best
            else:
                key = max(raw_best, atomic_best)
        elif family_agg == "additive":
            if raw_best is None:
                key = None if atomic_best is None else beta * atomic_best
            elif atomic_best is None:
                key = raw_best
            else:
                key = raw_best + beta * atomic_best
        else:
            raise ConfigError(f"unknown atomic.family_agg: {family_agg!r}")
        if key is None:
            continue
        ranked.append((-key, min(members), fam))
    ranked.sort()
    selected: list[tuple[int, float]] = []
    for _neg_key, _min_idx, fam in ranked[:return_size]:
        rep = family_rep(families[fam])
        selected.append((rep, results[rep]))
    return selected


class RetrievalEngine:
    def __init__(self, config: AMLConfig) -> None:
        self.config = config
        self.stemming = bool(config.raw.get("text", {}).get("stemming", True))
        self._embedder: Any = None  # lazy: heavy backends load on first use
        self._reranker: Any = None  # lazy: cross-encoder loads only when rerank.mode != off
        self.exact = ExactRetriever(min_coverage=float(config.exact_cfg.get("min_coverage", 0.25)))
        self.bm25 = BM25Retriever(
            k1=float(config.bm25_cfg.get("k1", 1.5)),
            b=float(config.bm25_cfg.get("b", 0.75)),
            stemming=self.stemming,
        )
        self.dense = DenseRetriever(None, min_score=float(config.dense_cfg.get("min_score", 0.05)))

    @property
    def embedder(self) -> Any:
        """Built on first access so disabled-dense deployments never load the
        embedding backend (torch et al.) at all."""
        if self._embedder is None:
            self._embedder = build_embedder(self.config.dense_cfg)
            self.dense.embedder = self._embedder
        return self._embedder

    def index_embedder(self) -> Any:
        """The embedder for indexing, or None when the dense retriever is off
        (skips embedding work entirely on the write path)."""
        return self.embedder if self.config.retrievers.dense else None

    @property
    def ce_reranker(self) -> Any:
        """EXP-U1: lazy cross-encoder, built on first rerank use only — a
        rerank-off deployment never imports torch a second time for this."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            r = self.config.rerank_cfg
            device = resolve_embedding_device(str(r.get("device", "auto")))
            self._reranker = CrossEncoder(str(r["model"]), device=device)
        return self._reranker

    # -- public API ---------------------------------------------------------

    def build_index(
        self,
        user_index: UserIndex,
        embedder: Any | None = None,
        derived_docs: list[DerivedDoc] | None = None,
    ) -> SearchIndex:
        return SearchIndex.build(
            user_index,
            self.config,
            self.index_embedder() if embedder is None else embedder,
            derived_docs,
        )

    def search(self, index: SearchIndex, query: str, top_k: int) -> SearchOutcome:
        cfg = self.config
        flags = cfg.retrievers
        surface = cfg.surface
        fusion = cfg.fusion_cfg
        method = str(fusion.get("method", "rrf"))
        rrf_k = float(fusion.get("rrf_k", 60))
        weights: dict[str, float] = fusion.get("weights", {})
        atomic_mode = str(cfg.atomic_cfg.get("mode", "off"))

        # 1) run enabled retrievers → per-channel score dicts + ranked lists.
        # The exact channel is raw-native (normalized substring scan over raw
        # units), so it stays off the derived-only search surfaces.
        channel_scores: dict[str, dict[int, float]] = {}
        ranked: dict[str, list[int]] = {}
        raw_hits: dict[str, int] = {}
        exact_eligible = atomic_mode not in ATOMIC_RETURN_MODES
        if flags.exact and exact_eligible:
            scores = self.exact.score(index, query)
            channel_scores["exact"] = scores
            ranked["exact"] = [idx for idx, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
            raw_hits["exact"] = len(scores)
        if flags.bm25:
            scores = self.bm25.score(index, query)
            channel_scores["bm25"] = scores
            ranked["bm25"] = [idx for idx, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
            raw_hits["bm25"] = len(scores)
        if flags.dense:
            q_vec = self.embedder.embed(query)  # property lazy-builds and wires dense
            scores = self.dense.score(index, q_vec)
            channel_scores["dense"] = scores
            ranked["dense"] = [idx for idx, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
            raw_hits["dense"] = len(scores)

        # 2) fusion (+ temporal boost)
        if method == "rrf":
            fused = fuse_rrf(ranked, rrf_k=rrf_k, weights=weights)
        elif method == "norm_sum":
            fused = fuse_norm_sum(channel_scores, norm=str(fusion.get("norm", "max")))
        else:
            raise ConfigError(f"unknown fusion.method: {method!r}")

        temporal_boosted = 0
        if bool(cfg.temporal_cfg.get("boost_delta", 0)) and fused:
            delta = float(cfg.temporal_cfg["boost_delta"])
            q_anchors = query_anchors(query)
            if q_anchors:
                for idx in list(fused.keys()):
                    unit_anchors = index.anchors.get(idx, [])
                    if any(qa.overlaps(ua) for qa in q_anchors for ua in unit_anchors):
                        fused[idx] += delta
                        temporal_boosted += 1

        base_ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

        # 3) neighbor expansion (NEIGHBOR_POLICY knob). Raw seeds expand to
        # session-adjacent raw units exactly as V0.5; derived seeds expand to
        # the atomic docs of the source's session-adjacent raw units — the
        # derived-space analog, so a derived-only surface stays derived-only.
        window = surface.neighbor_window()
        neighbor_added = 0
        results: dict[int, float] = {idx: s for idx, s in base_ranked}
        via: dict[int, str] = {idx: "direct" for idx in results}
        if window > 0 and base_ranked:
            seed_top = int(cfg.neighbor_cfg.get("seed_top", 25))
            decay = float(cfg.neighbor_cfg.get("decay", 0.5))
            for seed_idx, seed_score in base_ranked[:seed_top]:
                if seed_idx >= DERIVED_BASE:
                    source_idx = index.derived_docs[seed_idx - DERIVED_BASE].source_idx
                    for n_idx, d in index.ui.neighbors(source_idx, window):
                        for a_idx in index.derived_by_source.get(n_idx, ()):
                            cand = seed_score * (decay**d)
                            if cand > results.get(a_idx, float("-inf")):
                                if a_idx not in results:
                                    neighbor_added += 1
                                results[a_idx] = cand
                                via[a_idx] = "neighbor"
                else:
                    for n_idx, d in index.ui.neighbors(seed_idx, window):
                        cand = seed_score * (decay**d)
                        if cand > results.get(n_idx, float("-inf")):
                            if n_idx not in results:
                                neighbor_added += 1
                            results[n_idx] = cand
                            via[n_idx] = "neighbor"

        # 3.5) EXP1 governance state delta (ZT-20260919-13): read-surface
        # application of governance state, gated on governance.mode=on AND a
        # view actually attached to this index. Off/absent → this block is
        # dead code and V1 behavior is bit-identical. Selection-layer only
        # (same discipline as the A7 family walk): governance-dead candidates
        # are dropped before selection, the ranked remainder refills the
        # slots, nothing is rescoring and the store is never touched.
        governance_filtered: dict[str, int] = {}
        gov_view = getattr(index, "governance", None)
        if gov_view is not None and str(cfg.governance_cfg.get("mode", "off")) == "on":
            from aml_governance.filter import apply_governance_pool_filter

            results, governance_filtered = apply_governance_pool_filter(
                results, index, gov_view, gov_view.search_policy, DERIVED_BASE
            )

        # 4) SELECT top-N purely by fused relevance (fixed tie-break), then
        # 5) ORDER the selected surface by ORDERING_POLICY. Decoupled in V0.5
        # (H1/G30): the policy re-ranks the selected N items, it never changes
        # which items are selected. Under the default relevance_desc this is
        # exactly the V0 behavior, bit for bit (G29).
        #
        # A7 (ADDENDUM-01): atomic.dedup_policy=keep_raw/keep_atomic dedups by
        # PROVENANCE FAMILY at selection time — a raw unit and its own atomic
        # facts share one slot and the freed slot is refilled by the next
        # distinct family. Packing-layer only; the store is never touched
        # (G37). Deterministic: candidates are in fixed (score desc, idx asc)
        # order and the family representative is the preferred-kind member
        # with the best score.
        #
        # V2 (WORKORDER_AML_V2_R1 §3.1): atomic.family_agg makes the family's
        # rank discipline explicit. `max` (default) is the V1 walk, bit for
        # bit (G48): a family claims its slot at the first appearance of any
        # member in the candidate walk, i.e. family rank = max(member scores)
        # — the implicit "atomic promotion". raw_only / support_gate / additive
        # replace that implicit rule with an explicit family key, where
        # support_gate implements evidence-bound promotion: an atomic fact can
        # only lift a family whose source raw unit is itself visible in the
        # fusion candidate pool (`results`). Selection-layer only — packing,
        # ordering policies and the store are unchanged.
        return_size = surface.return_size_for(top_k)
        candidates = sorted(results.items(), key=lambda kv: (-kv[1], kv[0]))

        # EXP-U4 rerank scope=pool: RRF-fuse fused-score rank with CE rank over
        # the top `pool_size` candidates BEFORE selection — this CAN change the
        # selected set (that's the point); family walk/ordering stay unchanged.
        rerank_mode = str(cfg.rerank_cfg.get("mode", "off"))
        rerank_scope = str(cfg.rerank_cfg.get("scope", "surface"))
        if rerank_mode in ("ce_rrf", "laya_rrf") and rerank_scope in ("pool", "both") and len(candidates) > 1:
            pool_n = min(int(cfg.rerank_cfg.get("pool_size", 300)), len(candidates))
            head = self._ce_rrf_rerank(index, query, candidates[:pool_n], cfg.rerank_cfg)
            candidates = head + candidates[pool_n:]

        dedup_policy = str(cfg.atomic_cfg.get("dedup_policy", "off"))
        family_agg = str(cfg.atomic_cfg.get("family_agg", "max"))
        families_promoted = 0
        families_gate_blocked = 0
        orphan_promotions = 0
        family_walk_active = (
            dedup_policy != "off" and atomic_mode in ("both", "atomic_only") and bool(index.derived_docs)
        )
        if family_walk_active:
            families: dict[int, list[int]] = {}
            for idx, _ in candidates:
                fam = idx if idx < DERIVED_BASE else index.derived_docs[idx - DERIVED_BASE].source_idx
                families.setdefault(fam, []).append(idx)

            def family_rep(members: list[int]) -> int:
                if dedup_policy == "keep_raw":
                    return next((i for i in members if i < DERIVED_BASE), members[0])
                return next((i for i in members if i >= DERIVED_BASE), members[0])  # keep_atomic

            if family_agg == "max":
                # V1 A7 walk, unchanged bit for bit (G48).
                selected: list[tuple[int, float]] = []
                claimed: set[int] = set()
                for idx, _score in candidates:
                    if len(selected) >= return_size:
                        break
                    fam = idx if idx < DERIVED_BASE else index.derived_docs[idx - DERIVED_BASE].source_idx
                    if fam in claimed:
                        continue
                    rep = family_rep(families[fam])
                    selected.append((rep, results[rep]))
                    claimed.add(fam)
            else:
                selected = _select_by_family_key(
                    families,
                    results,
                    family_agg,
                    float(cfg.atomic_cfg.get("family_support_beta", 0.5)),
                    family_rep,
                    return_size,
                )

            # Promotion audit (WORKORDER §3.2) over the SELECTED families:
            # a family is "promoted by atomic" when its atomic best outranks
            # its raw best (or raw is absent from the pool entirely); an
            # "orphan" promotion happened without any in-pool raw support.
            # support_gate structurally forbids orphans (G49): raw-absent
            # families are never selectable. `selected` holds exactly one
            # (rep, score) per claimed family, families at most once.
            for rep, _score in selected:
                fam = rep if rep < DERIVED_BASE else index.derived_docs[rep - DERIVED_BASE].source_idx
                members = families[fam]
                raw_best = max((results[i] for i in members if i < DERIVED_BASE), default=None)
                atomic_best = max((results[i] for i in members if i >= DERIVED_BASE), default=None)
                if atomic_best is not None and (raw_best is None or atomic_best > raw_best):
                    families_promoted += 1
                    if raw_best is None:
                        orphan_promotions += 1
            if family_agg == "support_gate":
                # pool-wide gate accounting: raw-absent families with atomic
                # members were barred from selection entirely (blocked
                # promotions).
                families_gate_blocked = sum(
                    1 for members in families.values() if not any(i < DERIVED_BASE for i in members)
                )
        else:
            selected = candidates[:return_size]
        ordering = surface.ORDERING_POLICY
        if ordering == "temporal_asc":
            def sort_key(item: tuple[int, float]) -> tuple[Any, ...]:
                idx, _ = item
                return (self._doc_created_at(index, idx), -results[idx], idx)

            ordered = sorted(selected, key=sort_key)
        else:  # relevance_desc — `selected` already carries this exact order
            ordered = selected

        # 5.5) EXP-U1 rerank (opt-in): RRF-fuse the selected surface's own
        # rank with a cross-encoder relevance rank. Re-orders the selected N
        # items only — the selection set, packing and store are untouched.
        if rerank_mode in ("ce_rrf", "laya_rrf") and rerank_scope in ("surface", "both") and len(ordered) > 1:
            ordered = self._ce_rrf_rerank(index, query, ordered, cfg.rerank_cfg)

        # 6) pack per atomic mode. atomic_index_raw_return maps each derived
        # hit back to its provenance raw unit (best hit wins on duplicates —
        # `ordered` is score-sorted, so first occurrence is the best).
        items: list[PackedItem] = []
        seen_raw_source: set[str] = set()
        n_derived_returned = 0
        for idx, score in ordered:
            via_kind = via.get(idx, "direct")
            if idx < DERIVED_BASE:
                unit = index.ui.units[idx]
                items.append(
                    PackedItem(
                        idx=idx, via=via_kind, id=unit.id, content=unit.content,
                        score=score, created_at=unit.created_at, ts_ms=unit.ts_ms,
                        role=unit.role,
                    )
                )
                continue
            doc = index.derived_docs[idx - DERIVED_BASE]
            n_derived_returned += 1
            if atomic_mode == "atomic_index_raw_return":
                if doc.source_unit_id in seen_raw_source:
                    continue
                seen_raw_source.add(doc.source_unit_id)
                unit = index.ui.units[doc.source_idx]
                items.append(
                    PackedItem(
                        idx=idx, via="atomic", id=unit.id, content=unit.content,
                        score=score, created_at=unit.created_at, ts_ms=unit.ts_ms,
                        role=unit.role,
                    )
                )
            else:
                items.append(
                    PackedItem(
                        idx=idx, via="atomic", id=doc.doc_id, content=doc.text,
                        score=score, created_at=doc.created_at,
                        ts_ms=getattr(doc, "ts_ms", None),
                    )
                )

        trace = {
            "retrievers_on": sorted(ranked.keys()),
            "raw_hits": raw_hits,
            "fusion_method": method,
            "fused_candidates": len(fused),
            "temporal_boosted": temporal_boosted,
            "neighbor_window": window,
            "neighbor_added": neighbor_added,
            "select_policy": "relevance_top_n",
            "ordering_policy": surface.ORDERING_POLICY,
            "rerank_mode": rerank_mode,
            "rerank_scope": rerank_scope,
            "neighbor_policy": surface.NEIGHBOR_POLICY,
            "return_size_effective": return_size,
            "top_k": top_k,
            "query_token_count": len(tokenize(query)),
            "atomic_mode": atomic_mode,
            "n_derived_docs": len(index.derived_docs),
            "returned_derived": n_derived_returned,
            "returned_raw": len(items) - n_derived_returned,
            # V2 promotion audit (WORKORDER §3.2): the aggregation actually
            # applied this query ("off" = no family walk ran) plus the three
            # promotion counters; support_gate must keep orphan_promotions
            # at exactly 0 (G49).
            "family_agg": family_agg if family_walk_active else "off",
            "families_promoted_by_atomic": families_promoted,
            "families_gate_blocked": families_gate_blocked,
            "orphan_promotions": orphan_promotions,
            # EXP1 governance counters (zeros when the layer is off/absent)
            "governance_mode": str(cfg.governance_cfg.get("mode", "off")),
            **governance_filtered,
        }
        return SearchOutcome(items=items, trace=trace)

    def _ce_rrf_rerank(
        self,
        index: SearchIndex,
        query: str,
        ordered: list[tuple[int, float]],
        rerank: dict[str, Any],
    ) -> list[tuple[int, float]]:
        """EXP-U1: score every selected doc with the cross-encoder, then fuse
        (retrieval rank, CE rank) by classic RRF — same semantics verified in
        the offline probe: 1/(k+pos+1) summed over both rank lists."""
        k = float(rerank.get("rrf_k", 60.0))
        texts = [self._doc_text(index, idx) for idx, _ in ordered]
        if str(rerank.get("mode", "ce_rrf")) == "laya_rrf":
            ce_scores = self._laya_scores(query, texts, rerank)
        else:
            ce_scores = self.ce_reranker.predict([(query, t) for t in texts])
        ce_pos = {i: p for p, i in enumerate(sorted(range(len(ordered)), key=lambda i: -ce_scores[i]))}
        fused_order = sorted(
            range(len(ordered)),
            key=lambda i: (-(1.0 / (k + i + 1) + 1.0 / (k + ce_pos[i] + 1)), i),
        )
        return [ordered[i] for i in fused_order]

    @staticmethod
    def _laya_scores(query: str, texts: list[str], rerank: dict[str, Any]) -> list[float]:
        """EXP-L1: remote Laya score server — POST (query, text) pairs and get
        rubric-expectation scores back (0..3, monotonic in relevance, drop-in
        for a cross-encoder ranker). Fail-loud: transport/protocol errors raise,
        there is no silent fallback to another scorer."""
        import json as _json
        import os
        import urllib.request

        url = str(rerank.get("score_url", "")).strip() or os.environ.get("LAYA_SCORE_URL", "").strip()
        if not url:
            raise ConfigError("laya_rrf requires rerank.score_url or env LAYA_SCORE_URL")
        body = _json.dumps({"pairs": [[query, t] for t in texts]}).encode("utf-8")
        timeout_s = float(rerank.get("score_timeout_s", 120.0))
        attempts = int(rerank.get("score_retries", 4))
        payload = None
        last_exc = None
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(
                    url.rstrip("/") + "/score",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    payload = _json.loads(resp.read().decode("utf-8"))
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 — transient LAN/timeout blips
                last_exc = exc
                if attempt + 1 < attempts:
                    import time as _time
                    _time.sleep(min(2.0 ** attempt, 8.0))
        if payload is None:
            raise ConfigError(f"laya score server unreachable after {attempts} attempts: {last_exc}")
        scores = payload["scores"]
        if len(scores) != len(texts):
            raise ConfigError(f"laya score server returned {len(scores)} scores for {len(texts)} pairs")
        return [float(s) for s in scores]

    @staticmethod
    def _doc_text(index: SearchIndex, idx: int) -> str:
        if idx < DERIVED_BASE:
            return index.ui.units[idx].content
        return index.derived_docs[idx - DERIVED_BASE].text

    @staticmethod
    def _doc_created_at(index: SearchIndex, idx: int) -> str:
        if idx < DERIVED_BASE:
            return index.ui.units[idx].created_at
        return index.derived_docs[idx - DERIVED_BASE].created_at
