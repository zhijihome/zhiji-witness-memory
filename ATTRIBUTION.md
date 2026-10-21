# ATTRIBUTION — third-party components and data

All components are used unmodified via their published packages or APIs.
Model weights are downloaded at image-build time at pinned revisions.

## Runtime dependencies (pinned in `requirements.lock`)

| component | role | license |
|---|---|---|
| FastAPI / Starlette / uvicorn | HTTP service | MIT / BSD / BSD |
| pydantic | request/response schemas | MIT |
| httpx | embedding API client | BSD-3-Clause |
| sentence-transformers | local dense backend (dev) + model hub | Apache-2.0 |
| torch (CPU build) | tensor runtime for local models | BSD-3-Clause |
| transformers / tokenizers / huggingface-hub | model loading | Apache-2.0 |
| numpy / scikit-learn / scipy / narwhals | numeric + BM25 utilities | BSD / BSD-3 / BSD / MIT |
| pytest + plugins | test suite | MIT |
| ruff | lint | MIT |

(51 pinned packages total; see `requirements.lock` for exact versions.)

## Models and remote APIs

| component | source | usage | license/terms |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243…` | Hugging Face | local dense fallback + smoke | Apache-2.0 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face | local reranker in the frozen pipeline | Apache-2.0 |
| `text-embedding-v4` (1024-dim) | Alibaba Cloud DashScope | competition-required embedding API | Alibaba Cloud service terms |

## Benchmark data (development only — never shipped or used at runtime)

| data | source | usage | license |
|---|---|---|---|
| LOCOMO-style conversation data | fetched by `04_bench/fetch_locomo.py` at dev time | local proxy evaluation only | **CC BY-NC 4.0** — non-commercial research use; not vendored in this repo |

## People / authorship

- Method and implementation: the repository owner (submission author).
- Agent-assist tooling (coding assistants) was used during development;
  all submitted code was reviewed and tested by the author.
- No third-party method framework code is included.
