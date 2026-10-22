# ZVEM — Zhiji Versioned Evidence Memory

**Candidate `A1-DS`** — submission to the
[Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/), Cycle 2,
textual track, open-source methods (academic) group.

ZVEM is a verbatim-evidence memory service: it stores what it is given
without rewriting, and it answers `POST /search` with ordered, self-describing
evidence — never with a generated answer. This repository is the frozen
**competition export** of candidate `A1-DS`, reproducible end-to-end from a
clean checkout.

> **Lineage note.** ZVEM is the competition projection of a larger,
> long-running memory architecture (evidence organs, lifecycle/continuity
> machinery). Only the evidence + retrieval spine is submitted here; the
> broader architecture is not part of this submission and is not required
> to reproduce it.

## Design positions

- **No generative LLM anywhere in Add or Search.** Extraction is rule-based
  atomic fact-splitting; retrieval is BM25 + dense embeddings + cross-encoder
  rerank fused by RRF. There is no code path that calls a generative model —
  compliance is structural, not a flag.
- **Verbatim storage, enriched display.** Stored memory is never mutated.
  What changes is only the *returned text*: each item is rendered as
  `[YYYY-MM-DD] <speaker>: <verbatim content>` when — and only when — the
  source record actually carries that timestamp/speaker. Missing metadata is
  never invented (the database insertion time is never substituted as a
  source date). Under the official generic-role payload (`user`/`assistant`)
  no speaker is rendered — verified, not assumed; see TECHNICAL_REPORT §3.1.
- **Provenance through the pipeline.** Atomic derived documents inherit the
  timestamp and speaker of the source message they were extracted from, so
  every returned item can honestly say where and when it came from.
- **Reproducibility as a deliverable.** `competition/MANIFEST.json` pins the
  code SHA, config fingerprint, dependency-lock hash, model identity, and
  Docker digest; the manifest generator refuses to emit credential-shaped
  content.

## Provenance (frozen candidate `a1ds-r8`)

| artifact | value |
|---|---|
| source SHA (development tree) | `ff86223` |
| export SHA (this repo) | see tag `a1ds-r8` |
| image digest (`aml-a1ds:r8`) | `sha256:4bd8d5023cfffa0a3a25c24568c01a4ecf42ecdedef4ab592f344139e4a97cbb` |
| competition config hash | `sha256:bfb338dd2fb0a39563a9bba81c6da3d6a8f1e821cd2b86adb21e35e52515e7dc` |
| rule snapshot | `AML_CYCLE2_RULE_FREEZE_R2_20260921` |

## Quickstart (Docker, one command)

```bash
docker build -t zvem-a1ds -f 05_docker/Dockerfile .
docker run -d --name zvem -p 127.0.0.1:8093:8093 \
    -e AML_CONFIG_PATH=/app/10_config/competition.a1ds.json \
    -e AML_HOST=0.0.0.0 \
    -e AML_EMBEDDING_API_KEY=<your-dashscope-key> \
    zvem-a1ds
curl http://127.0.0.1:8093/health
```

The image bakes the pinned local reranker/dense models at build time; the
dense backend used by the frozen competition config is Alibaba Cloud
DashScope `text-embedding-v4` (1024 dims), required by the academic track —
supply your own key via `AML_EMBEDDING_API_KEY`. A pure-local
`hashed_ngram` backend exists for offline testing.

Bare metal: `make setup && AML_CONFIG_PATH=10_config/competition.a1ds.json make serve`
(Python 3.12+). Tests: `make test`. In-image smoke: `05_docker/smoke_in_image.sh`.

## API

Exactly the AML contract surface. `GET /health`, `POST /add`, `POST /search`
(schemas in `09_api/schemas.py`; auth via `Authorization`/`X-Api-Key` when a
token is configured). `POST /add` accepts `{request_id, session_id, user_id,
messages:[{role, content, timestamp?}]}`. `POST /search` returns
`{data:[{id, content, score, created_at}]}` ordered by relevance, capped at
`top_k=100`, synchronous HTTP 200.

Example returned item:

```json
{"id": "mem-000001",
 "content": "[2023-05-08] Caroline: I went to a LGBTQ support group yesterday…",
 "score": 0.0327,
 "created_at": "2023-05-08T00:00:00Z"}
```

## Layout

```
01_store 02_retrieval 03_adapter 09_api 10_config 11_atomic 12_governance
04_bench/      # measurement harness (LOCAL proxy evals; data fetched, not vendored)
05_docker/     # Dockerfile + compose + in-image smoke
08_tests/      # automated contract & behaviour gates
competition/   # MANIFEST.json + generator + RUNBOOK.md
00_contract/   # the external API contract documents we implement against
```

Frozen competition config: `10_config/competition.a1ds.json`.
Manifest: `competition/MANIFEST.json`. Reproduction guide:
`REPRODUCIBILITY.md`. Method report: `TECHNICAL_REPORT.md`. Change
disclosure vs. the source method: `METHOD_CHANGES.md`.

## License

MIT (see `LICENSE`). Third-party attributions in `ATTRIBUTION.md` /
`NOTICE`.
