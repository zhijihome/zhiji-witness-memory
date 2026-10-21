# Experiment Ledger Schema (V0, frozen)

One JSON object per line in `06_results/ledger/ledger.jsonl`; the same columns
in `06_results/ledger/ledger.csv` (nulls as empty cells). Field order is fixed;
mandated fields first (workorder §5), documented extras after `FAILURE_CASES`.

## Mandated fields

| Field | Type | Meaning |
|---|---|---|
| VERSION | str | service/bench version (`0.1.0` at V0) |
| COMMIT | str | git HEAD at run time (`unknown` when detached, e.g. Docker without build args) |
| CONFIG | str | sha256 of the behavior-relevant config subset (sorted keys) — **since R1.1**; see "Config fingerprint" below |
| MODEL | str | adapter tier + retriever backends summary |
| COORDINATE | str | `LOCAL` at V0 (never an AML score); `OFFICIAL_SMOKE` / `OFFICIAL_FULL` reserved |
| DATASET | str | dataset file + provenance note |
| RETRIEVERS_ON | str | comma list of enabled retrievers (exact/bm25/dense) |
| RETURN_SIZE | str/int | the effective return-surface knob (Contract §5 AMENDMENT-01) |
| ORDERING_POLICY | str | `relevance_desc` or `temporal_asc` |
| NEIGHBOR_POLICY | str | `off` or `window:N` |
| OVERALL | float/null | micro-mean token-F1 over all evaluated questions |
| FACT | float/null | LOCOMO cat-1 (single-hop) mean F1, mapped to FACT |
| TEMPORAL | float/null | LOCOMO cat-3 (temporal) mean F1 |
| GOVERNANCE | null | no LOCAL probe at V0 — honestly null, never fabricated |
| MULTI_HOP | float/null | LOCOMO cat-2 (multi-hop) mean F1 |
| PERSONALIZATION | null | no LOCAL probe at V0 |
| RULE | null | no LOCAL probe at V0 |
| SAFETY | null | no LOCAL probe at V0 |
| LATENCY_P50 | float | search latency p50, ms (runner-side wall clock) |
| LATENCY_P95 | float | search latency p95, ms |
| MODEL_CALLS | int | LLM calls made by the service during the run — measured as the delta of the server's cumulative `/health` usage counters (0 in NO_LLM is a real zero, not a constant; V0.5 O2) |
| TOKENS | int | LLM tokens (measured delta, 0 in NO_LLM) |
| COST | float | USD (measured delta, 0.0 in NO_LLM) |
| FAILURE_CASES | int | count of non-200 operations; details in `runs/<id>/failures.txt` |

## Documented extras

SERVER_CONFIG, RUN_ID, TS, LABEL, N_CONVS, N_QUESTIONS, N_ADDS, N_SEARCHES, ADD_LATENCY_P50,
ADD_LATENCY_P95, EM_OVERALL, MEMORY_ONLY_OVERALL (mean over LOCOMO cats 1–3),
EVIDENCE_RECALL_AT_10 / EVIDENCE_RECALL_AT_100 (gold dia_id coverage of the
returned surface), LOCOMO_CAT1..5_F1 (raw category breakdown), 
SURROGATE_ANSWERER, SERVER_TIER, DENSE_DEVICE (V0.5: resolved dense inference
device — `none` for the stdlib hashed backend, `off` when dense is disabled,
otherwise `cpu`/`mps`/`cuda`; latency and low-order floats are only comparable
within the same device class), DENSE_MODEL (V0.5: model@revision for the
sentence_transformers backend, empty otherwise), TOP_K, DETAIL_PATH.

## Mapping honesty

LOCOMO categories 4 (open-domain knowledge) and 5 (adversarial) have **no AML
dimension equivalent**; they appear only in the `LOCOMO_CAT*_F1` extras and in
OVERALL, and are `null` in the AML-dim columns. GOVERNANCE / PERSONALIZATION /
RULE / SAFETY are null at V0 because no faithful LOCAL proxy probe exists —
they must never be backfilled with fabricated numbers.

All LOCAL numbers come from the **fixed surrogate answerer** (deterministic,
no LLM, never sees gold) — they rank configurations for ablation only and are
**not AML scores** (two coordinate systems, Architecture §6).

## Config fingerprint (R1.1 — workorder G23~G25)

Retrieval behavior is decided by the **server** (`serve.py --set ...`); the
ledger label comes from the **client** (`run_bench.py --set ...`). To keep the
label honest, every record carries two fingerprints of the behavior-relevant
config subset (`AMLConfig.behavior_fingerprint()`):

| Field | Derivation |
|---|---|
| CONFIG | client-side behavior fingerprint |
| SERVER_CONFIG | server-side behavior fingerprint, as reported by `GET /health` at run start |

Excluded from the fingerprint (they cannot change any retrieval or scoring
behavior): `server` (bind host/port), `storage` (throwaway db paths,
journal/durability settings), `ops_log` (journal destination), and the
`_comment` key that `04_bench/configs/bench_default.json` carries on the
client side only. Everything else — surface knobs, retriever flags and
hyperparameters, dense backend, fusion, neighbor, temporal, text, adapter —
is included.

The runner aborts (exit 3, nothing written to the ledger or run details)
when the two fingerprints disagree (G24); under normal runs
`SERVER_CONFIG == CONFIG` (G25).

History note: ledger rows 1–3 (R1 foundation, through commit `224ce55`)
predate this scheme; their `CONFIG` was the sha256 of the **full** client
config JSON and they carry no `SERVER_CONFIG`. Those rows are append-only
history and are not rewritten.

`ledger.csv` is a projection of the JSONL. When a new field appears, its
header is evolved to the union of columns; old rows keep their values with
empty cells for new columns. The JSONL ledger itself stays append-only.

## V0.5 — SERVER_CONFIG relation accounting (G34)

Any audit of this ledger must distinguish **explicitly**:

| relation | meaning |
|---|---|
| `strict_match` | `SERVER_CONFIG` present and equal to `CONFIG` |
| `no_server_config_key` | key **absent** (legacy rows 1–3 only — pre-R1.1 scheme) |
| `null_value` | key present but null (integrity violation) |
| `value_mismatch` | key present, different value (integrity violation) |

Folding the missing key into a null comparison (e.g. `record.get("SERVER_CONFIG")`)
is prohibited — that is exactly the accounting bug the V0 Judge flagged.
Audit CLI: `python 04_bench/ledger_check.py [path]`; the V0 25-row ledger
accounts as 22 strict + 3 documented legacy no-key rows.

## V1 Atomic columns (workorder AML-V1 §1)

| Field | Meaning |
|---|---|
| ATOMIC_MODE | `off` / `both` / `atomic_only` / `atomic_index_raw_return` (workorder A4) |
| ATOMIC_EXTRACTOR | `none` (mode off) / `rule` / `llm` |
| ATOMIC_GRANULARITY | `coarse` / `fine`; empty when off |
| ATOMIC_PROMPT_VERSION | extractor+prompt version pin: `atomic-extract-llm-v1-<gran>` or `atomic-extract-rule-v1-<gran>`; G43 |
| ATOMIC_UNITS | derived rows created during this run (measured /health delta, like the O2 usage counters) |
| ATOMIC_CACHE_HITS / ATOMIC_CACHE_MISSES | extraction-cache counters for the run (LLM arms; rule/off are 0) |
| EXTRACTION_CACHE_FP | sha256 of the canonical extraction-cache export at run time (G36, `python -m aml_atomic.cli fingerprint`); `n/a` when no LLM cache is in play, `missing` when the cache db does not exist. **From the endpoint-identity round (G45) on, LLM rows record the fingerprint of their own granularity's entries only** — a coarse run is never again labeled with a fine+coarse mixed fingerprint (the pre-reforge coarse rows 66/67 carry the mixed value; that historical quirk is preserved, not rewritten). |
| SPEND_TOTAL_USD | whole-round cumulative API spend from the persisted gate state at run end (A5/G40); null = state unreadable, never a fabricated zero |

## Endpoint-identity round columns (ADDENDUM-02 §3.2 / G44–G46, reforge R1)

| Field | Meaning |
|---|---|
| LLM_ENDPOINT | endpoint identity of the endpoint the server actually called — the normalized HOSTNAME only (e.g. `api.openai.com`), never a full URL, never path/query/userinfo/credentials. `none` = no LLM in play (mode off or rule extractor). Sourced from the server's /health (client config is fingerprint-pinned equal via G24). |
| LLM_MODEL_ID | the model id actually sent in the requests. `none` when no LLM in play. |
| PROMPT_VERSION | the extractor prompt version in effect (`atomic-extract-llm-v1-<gran>` / `atomic-extract-rule-v1-<gran>`); `none` when mode off. Mirrors ATOMIC_PROMPT_VERSION; the dedicated column exists so provenance is machine-readable without config archaeology. |
| COORDINATE | `LOCAL` / `OFFICIAL_SMOKE` / `OFFICIAL_FULL` (server-reported, G44). OFFICIAL_* coordinates hard-fail at config validation and again before every model call unless the endpoint host is `api.openai.com`. |

Legacy rows written before this round carry none of these fields — that is the
documented `provenance_legacy_missing` state (`ledger_check.py`); they are
preserved verbatim, never rewritten.

## Extraction cache schema v2 (G45)

Cache key = `sha256(prompt_version | model_id | granularity | source_content_sha256
| endpoint_identity)`; the identity is the normalized hostname of the LLM
endpoint (never keys, never URLs with parameters). Same inputs on the same
endpoint hit the same entry deterministically; the same content via a
different endpoint lands on a different key, so relay and official products
cannot mix. Entries carry an `endpoint_identity` column and v2 frozen
artifacts include it, making every artifact self-describing about its
endpoint. Legacy v1 entries (keys without endpoint identity) are never
deleted; their old keys are unhittable under v2 lookups. Legacy-artifact
fingerprints are unchanged when read from a v1-schema db (the field is simply
absent from their canonical rows), so pre-existing EXTRACTION_CACHE_FP values
remain independently recomputable.

## Fingerprint semantics — lazy keys and what "different fingerprint" means (C3 / Judge N2)

`behavior_fingerprint()` = sha256 over the config subset excluding
`server`, `storage`, `ops_log`, `extraction_cache` (infra), and `_comment`.

It does **not** exclude *lazy keys* — config values that are inert under the
current effective configuration. Example (V0.5 Judge N2): `fusion.norm` has
no effect when `fusion.method=rrf`, yet both configs hash differently. So a
fingerprint difference proves at most "the config text differs", never
"different behavior"; conversely equal fingerprints do not prove equal
*outcomes* (data and code versions also feed outcomes). Any cross-round or
cross-arm comparison based on fingerprints must account for this. The same
applies to the atomic block: when `atomic.mode=off`, the other `atomic.*`
values (extractor/granularity/constants) are lazy — they change the
fingerprint but cannot change behavior.

`extraction_cache` (the cache *directory*) is excluded like db paths: its
*content* is what can change behavior, and that content is pinned per-run by
`EXTRACTION_CACHE_FP` instead.

## V2 family-discipline columns (WORKORDER_AML_V2_R1 §3.2)

| Field | Meaning |
|---|---|
| FAMILY_AGG | the family-aggregation discipline actually applied by the selection walk: `max` (V1 default, family rank = best member's rank) / `raw_only` / `support_gate` / `additive`; `off` when no family walk ran for the run's searches (atomic off or dedup off — note config validation hard-fails `family_agg≠max` in that situation, so `off` on a V2 row means the knob was `max` and inert); `unknown` only when the live service predates V2. Server-reported via /health. |
| FAMILIES_PROMOTED_BY_ATOMIC | measured per-run delta: among SELECTED families, those whose slot position was set by their atomic best outranking their raw best (or with no raw member in the fusion pool at all). Promotion count on the returned surface. |
| FAMILIES_GATE_BLOCKED | measured per-run delta (support_gate arms only): pool-wide count of families barred from selection entirely because the family had atomic members but NO raw member in the fusion candidate pool — blocked evidence-free promotions. |
| ORPHAN_PROMOTIONS | measured per-run delta: promoted selected families with no in-pool raw support. Structurally 0 under `support_gate` (G49 binding-correctness invariant); observable and honestly reported under `max`/`additive`. |

Historical rows (1–75) carry none of these fields; in the CSV projection they
keep empty cells in the new columns (G46 precedent), and the JSONL ledger is
byte-append-only.
