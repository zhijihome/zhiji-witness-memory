# ZVEM A1-DS — Technical Report (Cycle 2, Textual / Academic)

Candidate ID: **A1-DS** · Config: `10_config/competition.a1ds.json` ·
Manifest: `competition/MANIFEST.json`

## 1. What the system is

ZVEM (Zhiji Versioned Evidence Memory) is an evidence-preserving memory
service. For the AML contract it does exactly two things:

- `POST /add` — ingest conversation messages, extract atomic evidence units,
  index them.
- `POST /search` — return up to 100 ordered evidence items whose text the
  platform's answer model reads directly.

It deliberately does **not** generate answers, summaries, or any other text
at query or ingest time. Every generative-LLM code path is absent from the
submitted configuration (`adapter.tier = NO_LLM`), which makes the
"Search must not directly produce answers" rule structurally
unviolated rather than merely configured-off.

## 2. Pipeline

### 2.1 Add

```
messages → verbatim memory units (raw, one per message)
        → rule-based atomic extraction (fine granularity, max 8/message,
          keep_raw dedup policy, family aggregation = max)
        → per-unit embedding (DashScope text-embedding-v4, 1024 dims)
        → BM25 index + dense store + temporal anchors + speaker metadata
```

- **Atomic extraction** is deterministic rule-based fact-splitting
  (`11_atomic/extractors.py`, RULE_VERSION `atomic-extract-rule-v1-fine`).
  No model is invoked.
- **Provenance**: every derived atomic document inherits the timestamp
  (`ts_ms`) and speaker of its source message. Units lacking a source
  timestamp carry `ts_ms = None` — the API layer renders no date for them
  rather than borrowing the database insertion time.
- Writes are idempotent on `(user_id, request_id)`.

### 2.2 Search

```
query → BM25 channel + dense channel (v4 query embedding)
      → RRF fusion (k=60, minmax norm, equal weights)
      → cross-encoder rerank (local ms-marco MiniLM-L-6-v2)
      → session-neighbour expansion (window:1)
      → select top_k=100 by relevance → order relevance-desc
      → display render: [YYYY-MM-DD] <speaker>: <verbatim text>
```

- `top_k` is fixed at 100 (the contract surface). Screening experiments on a
  frozen stratified set showed reducing it (50/75) is strictly harmful.
- Ordering is relevance-descending; chronological ordering (either
  direction) measurably hurt answer quality in our controlled studies.

### 2.3 The display prefixes (what "DS" means)

`date_prefix` + `speaker_prefix` are two boolean switches in the frozen
config (`search.date_prefix`, `search.speaker_prefix`, default off). When
on, the API renders each item as `[YYYY-MM-DD] <speaker>: <content>` —
content itself verbatim.

Rationale: the only thing the platform's answer model is contractually
guaranteed to read is `content`. The `created_at` field is optional in the
contract and its consumption is unspecified; the speaker is not in the
contract at all. Baking both into the rendered text — from real stored
metadata only — is the single reliable channel to deliver them.

Both transforms are display-layer only; stored memory is byte-identical
with the switches off.

## 3. Why this configuration

The candidate was chosen by controlled ablation, not intuition. On a frozen
346-question local proxy set (public LOCOMO-style data, same answer model
class at temperature 0, identical scoring, **local harness `role` field
carrying real speaker names**):

| variant | overall F1 | notes |
|---|---|---|
| A1 baseline | 0.2968 | exact+BM25+dense+CE, no display metadata |
| + `date_prefix` | 0.3494 | temporal questions +24.7pt |
| + `speaker_prefix` (= A1-DS) | **0.4298** | refusal-category +26.7pt, no regression |
| Laya-style deep reranker (A2-p400) | 0.3420 | retired: loses to A1-DS, needs GPU chain |
| chronological ordering | ≤0.32 | rejected |
| top_k 50 / 75 | ≤0.31 | rejected |
| relative-date expansion | +0.001 | rejected (not material) |
| weekday `(Mon)` in prefix | −0.018 on confirm set | rejected (screening false positive) |

Protocol: variants were first screened on a frozen 81-question stratified
set, then re-validated on an independent 268-question confirmation set
before any full-set claim. Screening/confirmation disagreement ⇒ revert to
baseline (weekday is the worked example).

**These are local proxy figures, not official leaderboard results.** They
measure relative ordering of our own variants under a fixed harness; they
cannot be compared to other submissions' numbers.

### 3.1 Verification under the official payload shape

The table above was produced with the local harness putting real speaker
names in `role`. The official contract fixes `role` to `user`/`assistant`,
so `speaker_label()` returns `None` and the speaker prefix renders nothing —
we verified this end-to-end, it is not an assumption. Re-running the full
1986-question set with `role` forced to `user`/`assistant` (same code,
deterministic extractive surrogate answerer `fixed-extractive-v1`):

| variant | payload shape | overall | evidence recall@10 | recall@100 |
|---|---|---|---|---|
| A1 baseline | official | 0.0431 | 0.493 | 0.803 |
| + `date_prefix` | official | 0.0794 | 0.493 | 0.803 |
| A1-DS (date+speaker) | official | 0.0794 | 0.493 | 0.803 |
| A1-DS (date+speaker) | named `role` | 0.0839 | 0.723 | 0.918 |

Read-out, stated plainly:

- **`speaker_prefix` is exactly inert under official-shaped payloads** —
  A1-DS ≡ A1+date bit-for-bit, by design (the guard fabricates no speaker).
- `date_prefix` still contributes **+3.6pt** under official shape — the
  date channel survives because real `timestamp` metadata exists.
- Named-vs-generic roles cost only **−0.45pt overall** on this protocol
  (0.0839→0.0794), but evidence recall@100 drops **0.918→0.803**: names in
  the index text genuinely aid retrieval, the surrogate answerer just
  barely capitalizes on them.
- Absolute numbers on this protocol are *floored by the answerer*: a
  deterministic extractive stub cannot recover speaker identity from names
  inside `content` the way a generative answerer can. Treat them as
  variant-ordering evidence, not score predictions. First Smoke will read
  the actual `role` distribution from the official payloads; if names
  arrive inside `content` instead, the retrieval-side penalty partly
  returns.

The earlier 0.4298 figure should be read as **the upper bound when speaker
metadata is real** — not the expectation under the official contract.

## 4. Compliance posture

- Model used by the memory system in Add/Search: **none generative**
  (`NO_LLM`); dense embedding is the contractually-required
  `text-embedding-v4`.
- Return surface: `id / content / score / created_at`, relevance-desc,
  `top_k=100`, synchronous 200.
- Per-user isolation is exact (`user_isolation=exact_user_id`).
- Idempotent adds; no cross-sample state sharing; no hard-coded answers.
- Manifest generator performs credential-pattern scanning and refuses to
  emit unknown facts as anything but `UNKNOWN`/`NOT_BUILT` markers.

## 5. Limitations (honest)

- Multi-hop questions remain the weakest category (~0.18 local F1);
  grouping/aggregation variants tested so far did not help.
- Dense retrieval depends on the remote embedding API's availability and
  quota; a local `hashed_ngram` fallback exists but is not the submitted
  configuration.
- The display prefixes assume the answer model reads `content` only; if the
  platform also consumes `created_at`, dates appear twice — harmless.
