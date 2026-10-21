# zhiji-aml V0.5 service image — reproducible build from the lockfile.
FROM python:3.14-slim

WORKDIR /app

# dependencies first (layer cache friendly, fully pinned). Single lock since
# V0.5: the default dense backend (sentence_transformers) ships in it, so the
# image always matches the default config it is meant to run.
#
# Pre-install the CPU-only torch wheel from the PyTorch index: the lock pins
# torch==2.14.0 and PyPI's linux wheel drags in several GB of nvidia-* CUDA
# wheels that a CPU deployment never loads. `==2.14.0` accepts the `+cpu`
# local-version build, so the subsequent locked install sees torch satisfied.
# --no-deps: torch's pure-python deps are all pinned in requirements.lock and
# come from PIP_INDEX_URL (the pytorch index is slow/erratic from CN hosts).
RUN pip install --no-cache-dir torch==2.14.0 --no-deps \
    --index-url https://download.pytorch.org/whl/cpu
COPY requirements.lock ./requirements.lock
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.lock

# Bake the default dense model into the image at the revision pinned in
# 10_config/defaults.json, so first boot (and the in-image smoke) never needs
# network. Offline hosts without this cache must pre-download the weights or
# run the hashed_ngram fallback backend.
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT}
# hf-mirror.com (and other HF mirrors) cannot serve the xet CAS backend used by
# newer huggingface_hub — force the classic resolve path so mirrors work.
ENV HF_HUB_DISABLE_XET=1
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"

# Bake the ce_rrf reranker model used by the A1-DS competition candidate so the
# frozen image never needs network on first search either.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# application code
COPY pyproject.toml README.md ./
COPY 01_store ./01_store
COPY 02_retrieval ./02_retrieval
COPY 03_adapter ./03_adapter
COPY 09_api ./09_api
COPY 10_config ./10_config
COPY 11_atomic ./11_atomic
COPY 12_governance ./12_governance
COPY 04_bench ./04_bench
COPY 05_docker ./05_docker
COPY 08_tests ./08_tests
COPY conftest.py _bootstrap.py smoke.py ./
RUN pip install --no-cache-dir --no-deps .

# runtime state under /app/06_results (mount a volume to persist)
ENV AML_PORT=8093
EXPOSE 8093
HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8093/health', timeout=3).status < 300 else 1)"

CMD ["python", "-m", "aml_api.server"]
