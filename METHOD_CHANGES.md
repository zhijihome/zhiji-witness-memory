# METHOD_CHANGES — provenance and change disclosure

Per the competition requirement to *"disclose the original method, authors,
technical report, and all method changes."*

## 1. Original method and authorship

- **Submitted system**: ZVEM candidate `A1-DS`.
- **Original method**: ZVEM derives from an internal long-running memory
  architecture ("玄神经"/Xuan Neural) developed by the same author — a
  versioned-evidence memory organism with lifecycle, continuity, and
  governance organs.
- **Author**: the submission author (see repository owner). The competition
  export is authored by the same person; agent-assist tooling was used
  during development.
- **No third-party method is incorporated.** No external memory framework
  (Mem0, Zep, Graphiti, Letta/MemGPT, HippoRAG, Generative Agents, etc.) is
  imported, vendored, or re-implemented. Related-work awareness informed
  design vocabulary only.

## 2. All changes vs. the source method

The source architecture is a living system (organ-based, day/night cycle,
initiative gating). The competition submission is a deliberately converged
projection. Every change relative to the source method is listed:

| # | Change | Reason |
|---|---|---|
| C1 | Lifecycle organs removed (night cycle, dream, initiative gate, continuity ledger, physiology pipeline) | The AML contract exercises only Add/Search; lifecycle features have no observable channel and add unverifiable state |
| C2 | Memory model converged to verbatim units + rule-based atomic derived docs | Determinism + reproducibility; no generative dependency |
| C3 | Retrieval = BM25 + dense(v4) + local cross-encoder, RRF k=60, relevance-desc | Measured best on controlled ablation; cheaper than deep rerankers (A2 retired after losing +8.8pt on matched set) |
| C4 | **Display-layer `[YYYY-MM-DD]` prefix** when source timestamp exists | +5.3pt local F1; delivers date through the only guaranteed channel (`content`) |
| C5 | **Display-layer `<speaker>:` prefix** when source speaker exists | +8.0pt local F1; cat-5 refusal +26.7pt |
| C6 | Provenance extension: derived docs inherit source `ts_ms` + speaker | Required for C4/C5 to be honest rather than fabricated |
| C7 | Missing-source-timestamp units render **no** date (never insertion time) | Prevents fabricated provenance (caught in review) |
| C8 | Dense backend `dashscope` (text-embedding-v4) added, with SQLite vector cache, retry/backoff, fail-loud | Academic-track requirement; local MiniLM retained for offline/dev |
| C9 | Governance/freeze tooling: competition config, manifest generator, spend gate, rule snapshot ID | Verifiability obligations of the competition |
| C10 | Rejected changes (tested, not adopted): chronological ordering, top_k reduction, relative-date expansion, evidence grouping, dedup, weekday labels | Each measured neutral-or-negative on independent confirmation sets; kept out of the frozen config |

## 3. What did NOT change

- The verbatim-storage position: stored memory is never rewritten by the
  system.
- Idempotent `(user_id, request_id)` writes.
- Exact per-user isolation.
- The select-then-order split of the return surface.

## 4. Third-party components (function-level)

See `ATTRIBUTION.md` for the full inventory: `text-embedding-v4`
(DashScope), `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`
(sentence-transformers, Apache-2.0), FastAPI/uvicorn/pydantic, and the
pinned stack in `requirements.lock`. LOCOMO-style public proxy data is
fetched for local benchmarking only (CC BY-NC 4.0) and is not part of the
running system.
