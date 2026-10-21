# RUNBOOK — A1-DS competition instance

Everything needed to run the exact submitted configuration.

## 1. Environment

| variable | required | purpose |
|---|---|---|
| `AML_CONFIG_PATH` | yes | must point at `10_config/competition.a1ds.json` |
| `AML_EMBEDDING_API_KEY` | yes (dashscope backend) | DashScope API key for `text-embedding-v4` |
| `AML_HOST` | container only | `0.0.0.0` inside Docker |
| `AML_PORT` | optional | default 8093 |
| `AML_API_KEY` | recommended | token the service enforces; clients may send it as `Authorization: Token` (AML apply-form default), `Authorization: Bearer`, `X-Api-Key`, or `token` header |

## 2. Boot

```bash
docker build -t zvem-a1ds -f 05_docker/Dockerfile .
docker run -d --name zvem \
  -p 8443:8093 \
  -e AML_CONFIG_PATH=/app/10_config/competition.a1ds.json \
  -e AML_HOST=0.0.0.0 \
  -e AML_EMBEDDING_API_KEY=<key> \
  -e AML_API_KEY=<token> \
  zvem-a1ds
```

Without Docker: `make setup` then
`AML_CONFIG_PATH=10_config/competition.a1ds.json python -m aml_api.server`.

## 3. Verify

```bash
curl -s http://127.0.0.1:8093/health | python3 -m json.tool
# expect: "config_fingerprint" == manifest value, "tier":"NO_LLM"
python3 smoke.py --base-url http://127.0.0.1:8093
```

## 4. Capacity & limits (as submitted)

- Single uvicorn worker (in-memory indices are intentionally not shared;
  the store is designed for one worker — see `README`/store notes).
- `top_k=100`, synchronous HTTP 200 responses.
- Embedding calls go to DashScope (retry/backoff built in); a SQLite cache
  under `06_results/embedding_cache/` prevents double billing on repeats.
- Memory footprint ~1-2 GB typical; image ~5.4 GB (CPU-only torch + two
  baked models).
- Persistent runtime state lives under `06_results/` — mount a volume to
  keep it.

## 5. Failure behaviour (by design)

- Embedding API unreachable at boot → loud error, no silent fallback.
- Missing embedding key → same.
- Bad input → HTTP 422 with schema details; never a silent accept.
- Any generative-LLM call attempt → none exists in this configuration
  (structural NO_LLM).

## 6. Eval-data deletion (Cycle-2 rule: within 30 days post-run)

All eval data lives in two files under the app root — `06_results/aml.db`
(raw units + derived docs + journal) and `06_results/embedding_cache/
dashscope.sqlite` (content-keyed embedding cache) — plus their WAL/SHM
sidecars. The image mounts no volumes, so removing the container destroys
everything; for an in-place, evidenced wipe while the service stays up:

```bash
docker exec zvem python -m competition.purge_eval_data            # live wipe
docker exec zvem python -m competition.purge_eval_data --dry-run  # preview only
# or the nuclear option, if the instance is being retired anyway:
docker rm -f zvem   # writable layer gone => all eval data gone
```

A purge run should be logged (timestamp + output) as the deletion evidence.
