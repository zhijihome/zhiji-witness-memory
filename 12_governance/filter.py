"""Governance search-pool filter (EXP1).

Read-surface application of governance state. Runs on the fused candidate
pool AFTER retrievers/fusion/neighbors and BEFORE selection — the same
selection-layer discipline as the A7 family dedup walk: the filter only
removes governance-dead candidates (the ranked remainder refills the freed
slots), it never rescoring anything and never touches the store (INV2/G37).

Scope decisions (documented in the EXP1 report):
  * Only derived (atomic) candidates carry governance semantics — assertions
    are derived FROM atomic facts. Raw candidates are the canonical evidence
    layer (INV1) and stay governance-neutral in EXP1.
  * `filter_suppressed`: drop atomic candidates whose assertion or fact key is
    suppressed (deletion / forgetting suppression).
  * `prefer_current`: additionally drop a superseded/conflicted atomic
    candidate when its fact key has a live current representative IN THE POOL
    (current preferred; historical stays retrievable when current is absent).

Deterministic: pool iteration order is the candidate dict order (sorted
upstream); filtering never depends on anything else.
"""

from __future__ import annotations

from typing import Any


def apply_governance_pool_filter(
    results: dict[int, float],
    index: Any,
    view: Any,
    policy: str,
    derived_base: int,
) -> tuple[dict[int, float], dict[str, int]]:
    """Return (filtered_results, counters). `view` is a GovernanceView,
    `derived_base` the engine's DERIVED_BASE (passed in — no import cycle)."""
    counters = {"governance_filtered_suppressed": 0, "governance_filtered_stale": 0}
    if view is None or policy == "off":
        return results, counters

    keep: dict[int, bool] = {}
    for idx in results:
        if idx < derived_base:
            keep[idx] = True  # raw evidence layer: governance-neutral in EXP1
            continue
        doc = index.derived_docs[idx - derived_base]
        # derived doc_id IS the source atom id (service._to_docs doc_id=r.atom_id)
        if view.is_suppressed_atom(doc.doc_id):
            keep[idx] = False
            counters["governance_filtered_suppressed"] += 1
            continue
        if policy == "prefer_current" and view.is_stale_atom(doc.doc_id):
            keep[idx] = False
            counters["governance_filtered_stale"] += 1
            continue
        keep[idx] = True

    filtered = {idx: score for idx, score in results.items() if keep.get(idx, True)}
    return filtered, counters
