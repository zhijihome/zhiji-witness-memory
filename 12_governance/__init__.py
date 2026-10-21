"""aml_governance — governance state delta (EXP1 variant, ZT-20260919-13).

Seven-dimension governance state semantics over the frozen V1 architecture,
as an isolated delta: fact_key / assertion identity, current vs historical,
update / supersede, unresolved conflict, deletion / forgetting suppression,
provenance, and incremental Add/Search state transitions. Rule-based only
(zero model calls), insert-only storage, default-off at the config gate
(governance.mode=off → the layer is fully bypassed and V1 behavior is
bit-identical).
"""

from .engine import GovernanceEngine, GovernanceView
from .factkey import FACT_KEYER_VERSION, derive_fact_key, has_update_marker
from .filter import apply_governance_pool_filter
from .state import (
    STATUS_CONFLICT_OPEN,
    STATUS_CURRENT,
    STATUS_HISTORICAL,
    GovernanceState,
    apply_op,
    fold_ops,
    plan_assert,
)

__all__ = [
    "GovernanceEngine",
    "GovernanceView",
    "GovernanceState",
    "FACT_KEYER_VERSION",
    "derive_fact_key",
    "has_update_marker",
    "apply_governance_pool_filter",
    "STATUS_CURRENT",
    "STATUS_HISTORICAL",
    "STATUS_CONFLICT_OPEN",
    "apply_op",
    "fold_ops",
    "plan_assert",
]
