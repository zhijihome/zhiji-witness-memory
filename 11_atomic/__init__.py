"""aml_atomic — V1 Atomic extraction layer (architecture layer 11).

Add-time decomposition of raw messages into derived atomic facts.

玄神经 invariants enforced structurally (workorder A1 / G37):
  * raw evidence is canonical — this layer only ever READS raw unit content
    and writes to the separate derived table; there is no writeback path;
  * derived ≠ canonical — every produced row is marked derived by living in
    the `atomic_units` table and carries provenance (source unit id);
  * provenance — the atom id itself embeds the source arrival seq
    (`atom-<seq>-<k>`), and each row stores `source_unit_id`;
  * no-writeback — extraction results never mutate raw units.

Determinism (workorder A2 / G36): the LLM path runs at temperature 0 behind a
content-hash extraction cache, so identical inputs reuse the identical stored
extraction and bench reruns make zero LLM calls.
"""

from .cache import ExtractionCache, cache_fingerprint, cache_key
from .extractors import (
    PROMPT_VERSION_BASE,
    RULE_VERSION,
    ExtractionError,
    Extractor,
    LLMExtractor,
    RuleBasedExtractor,
)
from .pipeline import AtomicPipeline
from .spend import BudgetExceededError, SpendGate

__all__ = [
    "AtomicPipeline",
    "BudgetExceededError",
    "ExtractionCache",
    "ExtractionError",
    "Extractor",
    "LLMExtractor",
    "PROMPT_VERSION_BASE",
    "RULE_VERSION",
    "RuleBasedExtractor",
    "SpendGate",
    "cache_fingerprint",
    "cache_key",
]
