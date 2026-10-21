"""Atomic fact extractors (workorder A1 raw-message → atomic facts).

Two implementations behind one interface:

  * RuleBasedExtractor — pure-deterministic segmentation (sentences, optionally
    clauses). Zero LLM calls; the NO_LLM path required since V0 (G39) and the
    scientific control arm A5 that separates "atomicity helps" from "the LLM
    extracts well".
  * LLMExtractor — the competition model (Contract §3: Add-side LLM use is
    sanctioned; Search-side stays untouched, U9) at temperature 0.

Anti-overfitting discipline (workorder §2.1): both extractors are generic text
processing over exactly one message (role + content, nothing else — G38). No
prompt or rule references any benchmark, dataset, or conversation-corpus
feature; the prompts never name a dataset or question style.

Extractor outputs are BARE fact strings; the caller binds the speaker prefix
(`"{role}: {fact}"`) so the derived text form matches the raw indexing form
uniformly for both extractors.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Protocol

from aml_adapter import AMLModelAdapter

PROMPT_VERSION_BASE = "atomic-extract-llm-v1"
RULE_VERSION = "atomic-extract-rule-v1"

GRANULARITIES = ("coarse", "fine")


class ExtractionError(RuntimeError):
    """Raised when the LLM extractor cannot produce a parseable fact list."""


class Extractor(Protocol):
    version: str

    def extract(self, role: str, content: str) -> list[str]:
        """Return bare atomic facts for one message. Must be deterministic."""
        ...


# ---------------------------------------------------------------------------
# shared post-processing (deterministic; identical semantics for both paths)

_BACKCHANNEL = {
    "ok",
    "okay",
    "yes",
    "yeah",
    "yep",
    "no",
    "nope",
    "uh",
    "uh-huh",
    "um",
    "hmm",
    "hm",
    "haha",
    "hahaha",
    "lol",
    "thanks",
    "thank you",
    "thankyou",
    "sure",
    "right",
    "wow",
    "hello",
    "hi",
    "hey",
    "bye",
    "goodbye",
    "good night",
    "goodnight",
    "good morning",
    "see you",
    "talk later",
}


def _norm_fact(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def finalize_facts(facts: list[str], *, min_char_len: int, max_per_message: int) -> list[str]:
    """Deterministic cleanup applied to BOTH extractor outputs: strip, drop
    empties/backchannel/ultra-short fragments, dedupe within the message
    (normalized), and cap the count — first-wins in extraction order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in facts:
        fact = " ".join(raw.split()).strip(" \t-–—*•·,;:.")
        if len(fact) < min_char_len:
            continue
        key = _norm_fact(fact)
        if not key or key in _BACKCHANNEL or key in seen:
            continue
        seen.add(key)
        out.append(fact)
        if len(out) >= max_per_message:
            break
    return out


# ---------------------------------------------------------------------------
# A3 — NO_LLM rule-based extractor


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。])\s+|\n+|\s*\|\s*")
_CLAUSE_SPLIT = re.compile(
    r",?\s+(?:but|because|although|though|however|after that|and then|then|later|so that)\s+",
    re.IGNORECASE,
)


class RuleBasedExtractor:
    """Segment a message into sentence-shaped atomic units.

    granularity="coarse": one fact per sentence.
    granularity="fine": sentences are further split at a conservative set of
    clause connectives when both halves clear min_char_len.
    Speaker bindings and temporal expressions are preserved verbatim in the
    fact text (指代/时间保留 per workorder A3); pronouns are never resolved.

    Returns raw candidates; the pipeline applies `finalize_facts` uniformly to
    both extractor kinds AFTER the cache lookup, so the cap/min-length
    constants never invalidate cached extractions.
    """

    def __init__(self, granularity: str = "coarse") -> None:
        if granularity not in GRANULARITIES:
            raise ValueError(f"unknown atomic granularity: {granularity!r}")
        self.granularity = granularity
        self.version = f"{RULE_VERSION}-{granularity}"

    def extract(self, role: str, content: str) -> list[str]:
        text = " ".join(content.split())
        if not text:
            return []
        parts = [p for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
        if self.granularity == "fine":
            clauses: list[str] = []
            for part in parts:
                pieces = _CLAUSE_SPLIT.split(part.strip())
                if len(pieces) > 1:
                    clauses.extend(p for p in pieces if p and p.strip())
                else:
                    clauses.append(part)
            parts = clauses
        return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# A1/A2 — LLM extractor (Add-side; Contract §3 sanctions gpt-4o-mini on Add)

_GRANULARITY_INSTRUCTION = {
    "fine": (
        "Prefer splitting: each distinct fact gets its own item — one item per"
        " event, plan, preference, attribute, or piece of information."
    ),
    "coarse": (
        "Prefer grouping: keep details that belong to the same event or topic"
        " together in one item; do not over-split."
    ),
}

_PROMPT_TEMPLATE = """Extract atomic memory facts from the message below.

Message from {role}:
\"\"\"
{content}
\"\"\"

Rules:
- Each fact must be a single self-contained statement: understandable on its own, without this or any other message.
- Copy every name, date, time, place, number and quantity exactly as written; never translate or reformat them.
- Do not resolve pronouns; do not add outside knowledge. Keep unclear statements as written, or skip them.
- Never invent facts. If the message contains no facts, return an empty list.
- {granularity_instruction}
Output JSON exactly in this shape: {{"facts": ["...", "..."]}}"""

_PROMPT_SYSTEM = "You are a memory extraction engine. You reply with JSON only."


def build_prompt(role: str, content: str, granularity: str) -> str:
    return _PROMPT_TEMPLATE.format(
        role=role, content=content, granularity_instruction=_GRANULARITY_INSTRUCTION[granularity]
    )


def prompt_version(granularity: str) -> str:
    """The prompt and its version pin the granularity: each granularity is its
    own prompt (and therefore its own cache namespace and ledger value)."""
    return f"{PROMPT_VERSION_BASE}-{granularity}"


class LLMExtractor:
    """Extracts atomic facts with the competition model at temperature 0.

    Determinism contract (workorder A2 / G36): temperature 0 + fixed seed on
    every call, and every result is stored in the content-hash extraction
    cache by the caller's pipeline, so identical inputs reuse the identical
    stored output and bench reruns make zero LLM calls.

    Money contract (workorder A5 / G40): the spend gate is checked BEFORE each
    call and the completed call's measured cost is persisted AFTER it — even
    if response parsing later fails, the money was spent and is accounted.
    """

    def __init__(
        self,
        adapter: AMLModelAdapter,
        granularity: str = "fine",
        *,
        max_tokens: int = 500,
        spend: Any = None,  # SpendGate | None
    ) -> None:
        if granularity not in GRANULARITIES:
            raise ValueError(f"unknown atomic granularity: {granularity!r}")
        self.adapter = adapter
        self.granularity = granularity
        self.max_tokens = max_tokens
        self.spend = spend
        self.version = f"{prompt_version(granularity)}@{adapter.official_model_id}"

    def extract(self, role: str, content: str) -> list[str]:
        if self.spend is not None:
            self.spend.check()  # loud hard failure at/over the cap, before any call
        prompt = build_prompt(role, content, self.granularity)
        result = self.adapter.complete(
            prompt,
            system=_PROMPT_SYSTEM,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        if self.spend is not None:
            self.spend.record(
                cost_usd=result.cost_usd, tokens_in=result.tokens_in, tokens_out=result.tokens_out
            )
        return self._parse(result.text)

    def _parse(self, text: str) -> list[str]:
        try:
            payload = json.loads(text)
            facts = payload["facts"]
            if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
                raise ValueError("facts must be a list of strings")
        except Exception as exc:
            raise ExtractionError(f"atomic extraction returned unparseable output: {exc}") from None
        return [f for f in (raw.strip() for raw in facts) if f]
