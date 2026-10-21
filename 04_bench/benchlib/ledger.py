"""Experiment Ledger (workorder §5) — machine-readable, schema fixed at V0.

Mandated field order (JSONL keys / CSV columns); extra fields are appended
after FAILURE_CASES and documented in 04_bench/LEDGER_SCHEMA.md.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

LEDGER_FIELDS = [
    "VERSION",
    "COMMIT",
    "CONFIG",
    "MODEL",
    "COORDINATE",
    "DATASET",
    "RETRIEVERS_ON",
    "RETURN_SIZE",
    "ORDERING_POLICY",
    "NEIGHBOR_POLICY",
    "OVERALL",
    "FACT",
    "TEMPORAL",
    "GOVERNANCE",
    "MULTI_HOP",
    "PERSONALIZATION",
    "RULE",
    "SAFETY",
    "LATENCY_P50",
    "LATENCY_P95",
    "MODEL_CALLS",
    "TOKENS",
    "COST",
    "FAILURE_CASES",
]

EXTRA_FIELDS = [
    "SERVER_CONFIG",
    "RUN_ID",
    "TS",
    "LABEL",
    "N_CONVS",
    "N_QUESTIONS",
    "N_ADDS",
    "N_SEARCHES",
    "ADD_LATENCY_P50",
    "ADD_LATENCY_P95",
    "EM_OVERALL",
    "MEMORY_ONLY_OVERALL",
    "EVIDENCE_RECALL_AT_10",
    "EVIDENCE_RECALL_AT_100",
    "LOCOMO_CAT1_F1",
    "LOCOMO_CAT2_F1",
    "LOCOMO_CAT3_F1",
    "LOCOMO_CAT4_F1",
    "LOCOMO_CAT5_F1",
    "SURROGATE_ANSWERER",
    "SERVER_TIER",
    "DENSE_DEVICE",
    "DENSE_MODEL",
    "TOP_K",
    "DETAIL_PATH",
    # A8 return-face composition (ADDENDUM-01): means over questions
    "FACE_RAW_COUNT",
    "FACE_ATOMIC_COUNT",
    "FACE_UNIQUE_DIA",
    "FACE_REDUNDANT_SLOTS",
    # V1 Atomic (workorder A2/A4/A5, gates G36/G40/G43)
    "ATOMIC_MODE",
    "ATOMIC_EXTRACTOR",
    "ATOMIC_GRANULARITY",
    "ATOMIC_PROMPT_VERSION",
    "ATOMIC_UNITS",
    "ATOMIC_CACHE_HITS",
    "ATOMIC_CACHE_MISSES",
    "EXTRACTION_CACHE_FP",
    "SPEND_TOTAL_USD",
    # G46 (ADDENDUM-02 §3.2): explicit machine-readable LLM provenance —
    # endpoint identity (normalized hostname, no credentials), the requested
    # model id, the extractor prompt version. "none" = no LLM in play.
    "LLM_ENDPOINT",
    "LLM_MODEL_ID",
    "PROMPT_VERSION",
    # V2 (WORKORDER_AML_V2_R1 §3.2): family-aggregation discipline actually
    # applied ("off" when no family walk ran; "unknown" = pre-V2 server) and
    # the promotion-audit deltas. Schema extension follows the G46 precedent:
    # historical rows keep empty cells in the CSV projection; the JSONL ledger
    # stays byte-append-only.
    "FAMILY_AGG",
    "FAMILIES_PROMOTED_BY_ATOMIC",
    "FAMILIES_GATE_BLOCKED",
    "ORPHAN_PROMOTIONS",
]


def build_ledger_record(metrics: dict[str, Any]) -> dict[str, Any]:
    """Order the mandated fields first; keep unknown keys after the known ones."""
    record: dict[str, Any] = {}
    for f in LEDGER_FIELDS + EXTRA_FIELDS:
        if f in metrics:
            record[f] = metrics[f]
    for k, v in metrics.items():
        if k not in record:
            record[k] = v
    missing = [f for f in LEDGER_FIELDS if f not in record]
    if missing:
        raise ValueError(f"ledger record missing mandated fields: {missing}")
    return record


def append_ledger_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


# ---------------------------------------------------------------------------
# H7 (G34): SERVER_CONFIG relation accounting.
#
# The three states between SERVER_CONFIG and CONFIG must be distinguished
# EXPLICITLY. Folding "key missing" into "value equal" (e.g. via
# `record.get("SERVER_CONFIG")`) is exactly the accounting bug the V0 Judge
# flagged in the PI review: ledger rows 1–3 carry NO SERVER_CONFIG key at all
# (they predate the R1.1 fingerprint scheme) — that is not the same thing as a
# null value, and neither is the same as a strict match.


REL_STRICT_MATCH = "strict_match"
REL_NO_KEY = "no_server_config_key"
REL_NULL_VALUE = "null_value"
REL_VALUE_MISMATCH = "value_mismatch"


def server_config_relation(record: dict[str, Any]) -> str:
    """Classify one ledger record's SERVER_CONFIG vs CONFIG relation.

    Never uses `.get()` for the key test: a missing key and a null value are
    different states and must never fold into each other (G34).
    """
    if "SERVER_CONFIG" not in record:
        return REL_NO_KEY
    if record["SERVER_CONFIG"] is None:
        return REL_NULL_VALUE
    return REL_STRICT_MATCH if record["SERVER_CONFIG"] == record["CONFIG"] else REL_VALUE_MISMATCH


def audit_ledger_server_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Whole-ledger accounting: per-relation counts plus the row numbers of
    every non-strict row (rows 1–3 of the V0 ledger are the documented legacy
    `no_server_config_key` case)."""
    counts = {REL_STRICT_MATCH: 0, REL_NO_KEY: 0, REL_NULL_VALUE: 0, REL_VALUE_MISMATCH: 0}
    non_strict_rows: dict[str, list[int]] = {}
    for i, rec in enumerate(records, start=1):
        rel = server_config_relation(rec)
        counts[rel] += 1
        if rel != REL_STRICT_MATCH:
            non_strict_rows.setdefault(rel, []).append(i)
    return {"total_rows": len(records), "counts": counts, "non_strict_rows": non_strict_rows}


# ---------------------------------------------------------------------------
# G46 (ADDENDUM-02 §3.2): LLM provenance accounting.
#
# Three states per row, again NEVER folded together: rows written before the
# endpoint-identity round carry none of the new fields (documented legacy —
# preserved as-is, never rewritten); rows that do carry them must be
# well-formed; and NO row may ever contain key-shaped material.

_PROV_FIELDS = ("LLM_ENDPOINT", "LLM_MODEL_ID", "PROMPT_VERSION")
_HOSTNAME_RE = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")
_KEY_MATERIAL_RE = re.compile(r"sk-[A-Za-z0-9]{10,}")

PROV_OK = "provenance_ok"
PROV_LEGACY = "provenance_legacy_missing"
PROV_NULL = "provenance_null_value"
PROV_MALFORMED = "provenance_malformed"
PROV_KEY_LEAK = "provenance_key_material"


def llm_provenance_relation(record: dict[str, Any]) -> str:
    """Classify one row's LLM provenance fields (G46).

    key_material beats everything: no row may ever carry key-shaped material.
    legacy_missing = none of the three fields present (pre-reforge rows,
    preserved verbatim). Otherwise every field must be a non-null, well-formed
    value: LLM_ENDPOINT is either "none" or a bare hostname (never a URL,
    never userinfo/query/credentials), and the model/prompt ids are non-empty
    strings.
    """
    if _row_has_key_material(record):
        return PROV_KEY_LEAK
    present = [f for f in _PROV_FIELDS if f in record]
    if not present:
        return PROV_LEGACY
    if any(record[f] is None for f in present):
        return PROV_NULL
    endpoint = record.get("LLM_ENDPOINT", "none")
    if endpoint != "none":
        if not isinstance(endpoint, str) or not _HOSTNAME_RE.match(endpoint):
            return PROV_MALFORMED
    for f in ("LLM_MODEL_ID", "PROMPT_VERSION"):
        value = record.get(f)
        if not isinstance(value, str) or (value == "" and f == "LLM_MODEL_ID"):
            return PROV_MALFORMED
    return PROV_OK


def _row_has_key_material(record: dict[str, Any]) -> bool:
    return any(isinstance(v, str) and _KEY_MATERIAL_RE.search(v) for v in record.values())


def audit_ledger_key_hygiene(records: list[dict[str, Any]]) -> list[int]:
    """Row numbers containing key-shaped material (G40/G46 leak tripwire).
    Scans every value of every row — a key must never reach the ledger."""
    return [i for i, rec in enumerate(records, start=1) if _row_has_key_material(rec)]


def audit_ledger_llm_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Whole-ledger G46 accounting: per-relation counts and the row numbers of
    every non-ok row (key-material rows included)."""
    counts = {PROV_OK: 0, PROV_LEGACY: 0, PROV_NULL: 0, PROV_MALFORMED: 0, PROV_KEY_LEAK: 0}
    non_ok_rows: dict[str, list[int]] = {}
    for i, rec in enumerate(records, start=1):
        rel = llm_provenance_relation(rec)
        counts[rel] += 1
        if rel != PROV_OK:
            non_ok_rows.setdefault(rel, []).append(i)
    return {"total_rows": len(records), "counts": counts, "non_ok_rows": non_ok_rows}


def append_ledger_csv(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    existing_columns: list[str] = []
    if path.exists():
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_columns = list(reader.fieldnames or [])
            existing_rows = [{k: v for k, v in row.items() if k is not None} for row in reader]
    columns = list(dict.fromkeys(existing_columns + list(record.keys())))
    if existing_columns and columns == existing_columns:
        # plain append — schema unchanged
        with open(path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writerow({k: ("" if v is None else v) for k, v in record.items()})
        return
    # schema evolution (e.g. SERVER_CONFIG added in R1.1): rewrite the CSV
    # projection with the union of columns; old rows keep their values and get
    # empty cells for new columns. The JSONL ledger remains append-only.
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in columns})
        writer.writerow({k: ("" if record.get(k) is None else record.get(k, "")) for k in columns})
