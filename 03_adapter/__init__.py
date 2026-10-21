"""aml_adapter — the three-tier LLM gateway (architecture layer 00)."""

from .adapter import (
    DEFAULT_OFFICIAL_MODEL_ID,
    TIERS,
    AMLModelAdapter,
    CompletionResult,
    LLMUnavailableError,
    OfficialEndpointViolationError,
    Usage,
)

__all__ = [
    "DEFAULT_OFFICIAL_MODEL_ID",
    "TIERS",
    "AMLModelAdapter",
    "CompletionResult",
    "LLMUnavailableError",
    "OfficialEndpointViolationError",
    "Usage",
]
