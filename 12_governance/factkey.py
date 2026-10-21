"""Deterministic fact-key derivation for governance assertions (EXP1).

Governance state machine dimension 1 (fact_key / assertion identity): two
assertions can only interact (update, supersede, conflict) if they are keyed
to the same underlying fact subject. The keyer below is a pure-text, rule-based
classifier — zero model calls, zero benchmark signals, deterministic byte for
byte — so the whole governance layer stays inside the NO_LLM / offline test
contract of the EXP1 variant.

Identity model per fact text:

    "<skeleton><marker><value>"   e.g. "alice's favorite pizza is margherita"
                                   skeleton ~ "alice's favorite pizza is"
                                   value    ~ "margherita"

  * fact_key  = sha256(normalized skeleton) — the subject+relation identity;
    two assertions share a fact_key iff they speak about the same fact subject
    with the same relation verb.
  * value_hash = sha256(normalized value) — the asserted fill; equal
    fact_key + equal value_hash = the same fact asserted again (identity
    no-op); equal fact_key + different value_hash = a governance event
    (update / supersede / conflict, decided by the state machine).
  * No relation marker found → the whole normalized text is both key and
    value (whole-fact identity; only byte-equal repeats interact, i.e. none).

The keyer is a pluggable component (`governance.fact_keyer`); `rule-gk1` is
the only implementation in EXP1. Its splits are heuristic by design — the
experiment measures governance *state semantics* over an isolated corpus, not
keying quality, and every split decision is recorded in assertion rows so a
future keyer can be audited against this one.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

FACT_KEYER_VERSION = "gov-factkey-rule-gk1"

# Relation-boundary markers. Lazy leftmost match: the skeleton is everything
# up to and including the FIRST marker word, the value is the remainder.
# Adjectives like "favorite" are deliberately NOT markers: "my favorite pizza
# is X" must key on the full subject phrase, not its leading adjective.
_MARKER = (
    "is|are|was|were|am|has|have|had|be|been|being"
    "|like|likes|liked|love|loves|loved|hate|hates|hated"
    "|prefer|prefers|preferred|enjoy|enjoys|enjoyed"
    "|own|owns|owned|drive|drives|drove|ride|rides|rode"
    "|work|works|worked|live|lives|lived|study|studies|studied"
    "|teach|teaches|taught|speak|speaks|spoke|play|plays|played"
    "|watch|watches|watched|read|reads|eat|eats|ate|drink|drinks|drank"
    "|move|moves|moved|adopt|adopts|adopted|start|starts|started"
    "|sell|sells|sold|buy|buys|bought|quit|retired|graduated"
)
_RE_MARKER = re.compile(rf"^(.*?\b(?:{_MARKER})\b)(.*)$", re.IGNORECASE | re.DOTALL)

# Explicit update vocabulary: a newer assertion carrying one of these is
# treated as a correction/update even without message timestamps (the
# update/supersede transition's second evidence channel).
_RE_UPDATE_MARKER = re.compile(
    r"\b("
    r"moved to|changed|switched to|no longer|not anymore|actually|instead|"
    r"now|these days|currently|updated|upgraded|got married|got divorced|"
    r"broke up|got engaged|sold|bought|quit|retired|got promoted|transferred|"
    r"used to|renamed|replaced|new job|new home"
    r")\b",
    re.IGNORECASE,
)

# Discourse markers carry update EVIDENCE but not IDENTITY: they are stripped
# before keying so "actually my car is a toyota" lands on the same fact_key as
# "my car is a ford". Content verbs (sold/bought/quit/...) stay in the text —
# they evidence updates but are part of the fact's subject+relation.
_RE_DISCOURSE_STRIP = re.compile(
    r"\b(moved to|changed|switched to|no longer|not anymore|actually|instead|"
    r"now|these days|currently|updated|upgraded|used to)\b,?",
    re.IGNORECASE,
)


def normalize_gov_text(text: str) -> str:
    """NFKC + casefold + whitespace collapse — same normalization family as
    the store's exact-match path (01_store.store.normalize_text)."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def has_update_marker(text: str) -> bool:
    return bool(_RE_UPDATE_MARKER.search(text))


def derive_fact_key(fact_text: str) -> tuple[str, str, str, bool]:
    """Split a speaker-bound fact text into governance identity parts.

    Returns (fact_key, value_hash, normalized_text, update_marker).
    fact_key/value_hash are hex sha256 digests of the normalized skeleton and
    value remainder respectively; discourse markers are stripped before the
    split (they evidence updates, they do not identify facts).
    """
    norm = normalize_gov_text(fact_text)
    marker = has_update_marker(norm)
    keyable = normalize_gov_text(_RE_DISCOURSE_STRIP.sub(" ", norm))
    m = _RE_MARKER.match(keyable)
    if m is None:
        # whole-fact identity: key == value == full normalized text
        return (_sha256("raw:" + keyable), _sha256("val:" + keyable), norm, marker)
    skeleton = normalize_gov_text(m.group(1))
    value = normalize_gov_text(m.group(2))
    return (_sha256("skel:" + skeleton), _sha256("val:" + value), norm, marker)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
