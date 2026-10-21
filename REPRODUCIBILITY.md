# REPRODUCIBILITY — how to verify this submission

Everything below is checkable from this repository alone plus one DashScope
API key (the academic-track-required embedding model).

## 1. Frozen artifacts

| artifact | location | pinned value |
|---|---|---|
| Competition config | `10_config/competition.a1ds.json` | `config_fingerprint = fbabe24e…` (full value in manifest) |
| Manifest | `competition/MANIFEST.json` | code SHA `95cef0f2…`, `git_dirty=false`, lock hash, model identity, docker digest |
| Dependencies | `requirements.lock` | fully pinned, single lock |
| Local models | baked in image | `all-MiniLM-L6-v2@1110a243…`, `ms-marco-MiniLM-L-6-v2` |
| Docker image | `05_docker/Dockerfile` | digest recorded in manifest (`docker_digest_if_available`) |

Regenerate and compare at any time:

```bash
python3 competition/generate_manifest.py \
    --config 10_config/competition.a1ds.json --out /tmp/m.json
diff /tmp/m.json competition/MANIFEST.json   # expect identical modulo generated_at/digest
```

## 2. Cold-start verification

```bash
docker build -t zvem-a1ds -f 05_docker/Dockerfile .
docker run -d --name zvem -p 127.0.0.1:8093:8093 \
    -e AML_CONFIG_PATH=/app/10_config/competition.a1ds.json \
    -e AML_HOST=0.0.0.0 -e AML_EMBEDDING_API_KEY=$KEY zvem-a1ds
python3 smoke.py --base-url http://127.0.0.1:8093
```

The smoke checks: `/health` 2xx, add→search round-trip, exact per-user
isolation, idempotent re-add, response shape. `GET /health` echoes
`config_fingerprint`, which must equal the manifest value.

CPU-only note: the Dockerfile installs the `+cpu` build of the pinned torch
(local-version label satisfies `==2.14.0`) so the image carries no CUDA
dependencies. `HF_HUB_DISABLE_XET=1` is set so mirrors work; remove it if
building against huggingface.co directly.

## 3. Behaviour contract you should observe

- `POST /search` returns `[YYYY-MM-DD] <speaker>: <verbatim>` items only
  when the stored source record actually has that metadata.
- A record without a source timestamp renders **no** date — it never
  borrows the write-time clock.
- Re-POSTing the same `(user_id, request_id)` add is a no-op.
- Two different `user_id`s never see each other's items.
- `GET /health` reports `tier: NO_LLM`, `llm_model_id: none` — no
  generative call exists in the process.

## 4. Known non-reproducible-by-design inputs

- DashScope embedding responses (remote API; the SQLite cache in
  `06_results/embedding_cache/` makes repeat runs deterministic on the same
  text, but the cache is runtime state, not shipped).
- `MANIFEST.json` `generated_at` timestamp.
