"""Dense retrieval with local embeddings.

Default backend `sentence_transformers` (V0.5): true semantic embeddings from a
local sentence-transformers model (config `dense.sentence_transformers_model`,
revision pinned via `dense.sentence_transformers_revision`). Device is
configurable (`dense.device`: auto|cpu|mps|cuda) and the resolved device is
reported on /health and into the experiment ledger — latency and low-order
floats are only comparable within the same device class.

Fallback backend `hashed_ngram`: a fastText-style hashed bag of features —
word unigrams/bigrams plus character 3–5-grams — projected through a stable
MD5-based signed hash into `dim` buckets, sublinear-TF weighted and L2
normalized. It is fully local, dependency-free and bit-for-bit deterministic:
the supported no-torch path (G33) and the only backend for the first ~20
ledger rows of V0 history.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import threading
import time
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from aml_store.model import speaker_label

from .tokenize import STOPWORDS, content_tokens, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from .engine import SearchIndex


class EmbeddingBackend(Protocol):
    def embed(self, text: str) -> dict[int, float]:
        """Sparse vector: bucket -> signed value (already L2-normalized)."""
        ...

    def embed_many(self, texts: list[str]) -> list[dict[int, float]]:
        """Optional batched path — backends without it fall back to per-text
        `embed` at the call site (identical result, fewer round trips)."""
        ...


class HashedNgramEmbedder:
    """Stable hashed n-gram embeddings (no Python `hash()` — md5 only)."""

    device = "none"  # pure-Python deterministic math; reported in the ledger

    def __init__(
        self,
        dim: int = 65536,
        char_ngrams: tuple[int, ...] = (3, 4, 5),
        char_weight: float = 0.4,
        word_weight: float = 1.0,
        use_bigrams: bool = True,
    ) -> None:
        self.dim = dim
        self.char_ngrams = tuple(char_ngrams)
        self.char_weight = char_weight
        self.word_weight = word_weight
        self.use_bigrams = use_bigrams

    def _bucket_and_sign(self, feature: str) -> tuple[int, int]:
        digest = hashlib.md5(feature.encode("utf-8"), usedforsecurity=False).digest()
        value = int.from_bytes(digest[:8], "big")
        bucket = value % self.dim
        sign = -1 if (value >> 63) & 1 else 1
        return bucket, sign

    def embed(self, text: str) -> dict[int, float]:
        vec: dict[int, float] = {}

        def _acc(feature: str, weight: float, tf: int) -> None:
            bucket, sign = self._bucket_and_sign(feature)
            vec[bucket] = vec.get(bucket, 0.0) + sign * weight * (1.0 + math.log(tf))

        words = [t for t in tokenize(text) if t not in STOPWORDS]
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        for w, tf in counts.items():
            _acc(f"w:{w}", self.word_weight, tf)
        if self.use_bigrams:
            bigrams: dict[str, int] = {}
            for a, b in zip(words, words[1:], strict=False):
                g = f"g:{a}|{b}"
                bigrams[g] = bigrams.get(g, 0) + 1
            for g, tf in bigrams.items():
                _acc(g, self.word_weight, tf)
        char_source = " ".join(words)
        for n in self.char_ngrams:
            grams: dict[str, int] = {}
            for i in range(len(char_source) - n + 1):
                g = char_source[i : i + n]
                grams[g] = grams.get(g, 0) + 1
            for g, tf in grams.items():
                _acc(f"c{n}:{g}", self.char_weight, tf)

        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        return vec


def resolve_embedding_device(device: str) -> str:
    """Resolve `dense.device` (auto|cpu|mps|cuda) without loading any model.
    Latency and low-order floats are only comparable between runs on the same
    resolved device — the resolved value is echoed on /health and into the
    ledger."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing/broken → cpu
        pass
    return "cpu"


class SentenceTransformersEmbedder:
    """True local semantic embeddings (pinned dependency as of V0.5).

    Inference is strictly local; the model weights (~90MB) download once on
    first use. Offline hosts must either pre-download the weights (the Docker
    image bakes them at build time) or switch `dense.backend` back to
    `hashed_ngram` — a missing model is a loud error, never a silent fallback.
    """

    def __init__(self, model_name: str, device: str = "auto", revision: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on installed deps
            raise RuntimeError(
                "dense.backend='sentence_transformers' needs the pinned dependencies "
                "(requirements.lock installs sentence-transformers + torch). The supported "
                "no-torch path is dense.backend='hashed_ngram'."
            ) from exc
        self.requested_device = device
        self.device = resolve_embedding_device(device)
        kwargs: dict[str, Any] = {"device": self.device}
        if revision:
            kwargs["revision"] = revision
        try:
            self._model = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load sentence-transformers model {model_name!r} "
                f"(device={self.device}, revision={revision or 'latest'}): {exc}. "
                "First use downloads the model weights (~90MB); offline hosts must "
                "pre-download them or set dense.backend='hashed_ngram'."
            ) from exc
        self.dim = self._model.get_sentence_embedding_dimension() or 384

    def embed(self, text: str) -> dict[int, float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return {i: float(v) for i, v in enumerate(vector) if v != 0.0}


class DashScopeEmbedder:
    """EXP-EMB-V4 (2026-10-15): OpenAI-compatible embeddings API backend —
    LOCAL-coordinate experiment only. Swaps the dense space to a remote model
    (e.g. `text-embedding-v4`) without touching the frozen backends.

    - Key is read from env (`dense.api_key_env`, default `DASHSCOPE_API_KEY`)
      at build time; it is never written to disk or the ledger. Missing key is
      a loud error — no silent fallback.
    - Persistent SQLite cache (`dense.api_cache_path`) keyed by
      sha256(model|dimensions|text): replays and crash-resumes never re-bill.
    - Vectors are L2-normalized here so `DenseRetriever.score`'s
      "doc norms are 1" invariant holds.
    - `embed_many` batches misses into one HTTP call per `batch_size` texts;
      `engine` warms the cache via this path, keeping per-doc `embed` order
      bit-identical.
    """

    device = "remote-api"  # echoed on /health and into the ledger

    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        dimensions: int,
        cache_path: str,
        batch_size: int = 10,
        timeout: float = 60.0,
        max_retries: int = 4,
    ) -> None:
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"dense.backend='dashscope' needs ${api_key_env} in the "
                "environment (embedding API key). Fail-closed: no fallback."
            )
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("dense.backend='dashscope' needs httpx installed") from exc
        self.model = model
        self.dimensions = dimensions
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self.api_calls = 0
        self.api_tokens = 0
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI sync endpoints run on an AnyIO worker pool — the embedder is
        # lazily built on whichever thread served the first request, so the
        # cache connection must be usable from every worker thread.
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db_lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (h BLOB PRIMARY KEY, v BLOB)")
        self._db.commit()

    def _key(self, text: str) -> bytes:
        return hashlib.sha256(
            f"{self.model}|{self.dimensions}|{text}".encode("utf-8")
        ).digest()

    @staticmethod
    def _pack(vec: list[float]) -> bytes:
        return array("f", vec).tobytes()

    @staticmethod
    def _unpack(blob: bytes) -> list[float]:
        a = array("f")
        a.frombytes(blob)
        return list(a)

    def _fetch_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post("/embeddings", json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"transient HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"embedding API HTTP {resp.status_code} (non-retryable): {resp.text[:300]}"
                    )
                body = resp.json()
                rows = body.get("data") or []
                if len(rows) != len(texts):
                    raise RuntimeError(
                        f"embedding API returned {len(rows)} vectors for {len(texts)} inputs"
                    )
                self.api_calls += 1
                self.api_tokens += int((body.get("usage") or {}).get("total_tokens", 0))
                by_index = {int(r["index"]): list(r["embedding"]) for r in rows}
                return [by_index[i] for i in range(len(texts))]
            except Exception as exc:  # network/transient → bounded backoff
                last_exc = exc
                if "non-retryable" in str(exc):
                    raise
                time.sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError(f"embedding API failed after {self.max_retries} tries: {last_exc}")

    def embed_many(self, texts: list[str]) -> list[dict[int, float]]:
        keys = [self._key(t) for t in texts]
        vecs: list[list[float] | None] = [None] * len(texts)
        pending: dict[bytes, list[int]] = {}
        with self._db_lock:
            cur = self._db.cursor()
            for i, k in enumerate(keys):
                row = cur.execute("SELECT v FROM emb WHERE h=?", (k,)).fetchone()
                if row is not None:
                    vecs[i] = self._unpack(row[0])
                else:
                    pending.setdefault(k, []).append(i)
        if pending:
            miss_keys = list(pending)
            for off in range(0, len(miss_keys), self.batch_size):
                chunk = miss_keys[off : off + self.batch_size]
                chunk_vecs = self._fetch_batch([texts[pending[k][0]] for k in chunk])
                with self._db_lock:
                    self._db.executemany(
                        "INSERT OR REPLACE INTO emb (h, v) VALUES (?, ?)",
                        [(k, self._pack(v)) for k, v in zip(chunk, chunk_vecs)],
                    )
                    self._db.commit()
                for k, v in zip(chunk, chunk_vecs):
                    for i in pending[k]:
                        vecs[i] = v
        return [self._to_sparse(v) for v in vecs]  # type: ignore[arg-type]

    def _to_sparse(self, vec: list[float]) -> dict[int, float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0:
            return {}
        return {i: v / norm for i, v in enumerate(vec) if v != 0.0}

    def embed(self, text: str) -> dict[int, float]:
        return self.embed_many([text])[0]


def build_embedder(dense_cfg: dict[str, Any]) -> EmbeddingBackend:
    backend = dense_cfg.get("backend", "hashed_ngram")
    if backend == "hashed_ngram":
        return HashedNgramEmbedder(
            dim=int(dense_cfg.get("dim", 65536)),
            char_ngrams=tuple(dense_cfg.get("char_ngrams", (3, 4, 5))),
            char_weight=float(dense_cfg.get("char_weight", 0.4)),
            word_weight=float(dense_cfg.get("word_weight", 1.0)),
            use_bigrams=bool(dense_cfg.get("use_bigrams", True)),
        )
    if backend == "sentence_transformers":
        return SentenceTransformersEmbedder(
            str(dense_cfg.get("sentence_transformers_model", "")),
            device=str(dense_cfg.get("device", "auto")),
            revision=str(dense_cfg.get("sentence_transformers_revision", "")),
        )
    if backend == "dashscope":
        return DashScopeEmbedder(
            base_url=str(dense_cfg.get("api_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            api_key_env=str(dense_cfg.get("api_key_env", "DASHSCOPE_API_KEY")),
            model=str(dense_cfg.get("api_model", "text-embedding-v4")),
            dimensions=int(dense_cfg.get("api_dimensions", 1024)),
            cache_path=str(dense_cfg.get("api_cache_path", "06_results/embedding_cache/dashscope.sqlite")),
            batch_size=int(dense_cfg.get("api_batch_size", 10)),
        )
    raise ValueError(f"unknown dense backend: {backend!r}")


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: EmbeddingBackend, min_score: float = 0.05) -> None:
        self.embedder = embedder
        self.min_score = min_score  # cosine floor: drops hash-collision-level noise

    def embed_query(self, query: str) -> dict[int, float]:
        return self.embedder.embed(query)

    def score(self, index: SearchIndex, query_vec: dict[int, float]) -> dict[int, float]:
        """Cosine similarity — packed matrix matmul for dense backends, sparse
        inverted index for hash backends (doc vectors pre-normalized either way)."""
        matrix = getattr(index, "dense_matrix", None)
        if matrix is not None and query_vec:
            import numpy as np

            q = np.zeros(matrix.shape[1], dtype=np.float32)
            for b, v in query_vec.items():
                if 0 <= b < matrix.shape[1]:
                    q[b] = v
            q_norm = float(np.linalg.norm(q))
            if q_norm <= 0:
                return {}
            sims = matrix @ (q / q_norm)  # doc norms are 1 by construction
            hits = np.nonzero(sims >= self.min_score)[0]
            rows = index.dense_rows
            return {rows[int(r)]: float(sims[r]) for r in hits}
        if not query_vec or not index.dense_postings:
            return {}
        q_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if q_norm <= 0:
            return {}
        dots: dict[int, float] = {}
        for bucket, q_val in query_vec.items():
            postings = index.dense_postings.get(bucket)
            if not postings:
                continue
            for idx, d_val in postings:
                dots[idx] = dots.get(idx, 0.0) + q_val * d_val
        return {
            idx: dot / q_norm
            for idx, dot in dots.items()
            if dot / q_norm >= self.min_score  # doc norms are 1 by construction
        }


def embed_content_for_index(embedder: EmbeddingBackend, role: str | None, content: str) -> dict[int, float]:
    """Embedding used when indexing a unit (speaker label included when the
    role carries one: identity is strong retrieval signal in two-participant
    conversations; generic chat roles contribute nothing)."""
    label = speaker_label(role)
    return embedder.embed(f"{label}: {content}" if label else content)


def query_feature_count(text: str) -> int:  # small helper used by tests
    return len(content_tokens(text))
