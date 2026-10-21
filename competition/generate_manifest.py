#!/usr/bin/env python3
"""AML_COMPETITION_CANDIDATE_MANIFEST generator (Cycle2 packaging R1).

Emits a single reproducible JSON manifest describing the competition
candidate: code provenance, the frozen competition config (hash-pinned), the
three model identity layers, the G45 cache namespace, dependency/env pins and
the governing official rule snapshot.

Honesty contract (workorder AML-CYCLE2-PACKAGING-R1 §11): a field that cannot
be determined at packaging time is emitted as UNKNOWN / NOT_BUILT — never
fabricated. This round performs ZERO official eval / smoke / full calls, so
upstream_reported_model is UNKNOWN_NOT_OBSERVED and docker_digest_if_available
is NOT_BUILT unless a digest is supplied explicitly (e.g. AML_DOCKER_DIGEST).

Reproducibility contract (G-C8): all content is derived from the tree + the
frozen config + the dependency lock. `generated_at` defaults to now (UTC) and
can be pinned via --generated-at; with identical inputs (and pinned
time/environment) two runs in the same tree emit byte-identical manifests.

Secret hygiene contract (§12): the generator refuses to emit any manifest
whose serialized content matches credential patterns — the last line of
defense after the config/manifest staying secret-free by construction.

Usage:
    python3 competition/generate_manifest.py \
        [--config 10_config/competition.a1kr.json] [--out competition/MANIFEST.json] \
        [--generated-at "2026-09-07T00:00:00+00:00"] [--docker-digest sha256:...] \
        [--environment-json '{"python": "3.14.7"}']
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import _bootstrap  # noqa: F401  (must precede aml_* imports)

from aml_config import load_config  # noqa: E402
from aml_config.config import canonical_endpoint_identity  # noqa: E402
from aml_atomic.extractors import RULE_VERSION, prompt_version  # noqa: E402

# Frozen provenance (control tower, workorder AML-CYCLE2-PACKAGING-R1 §1):
CODE_CANONICAL_SHA = "66f76cca45574d2184c91c0ad68796447b41b6b0"
DEFAULT_RULE_SNAPSHOT_ID = "AML_CYCLE2_RULE_FREEZE_R1_20260907"

# §12 secret hygiene — pattern-based scan over the serialized manifest.
# Deliberately narrow to credential SHAPES: the manifest legitimately carries
# many sha256 hex digests and the public env-var *name* (official_api_key_env),
# none of which is a secret.
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),  # OpenAI-style key literal
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{8,}"),  # Authorization header value
    re.compile(r"(?i)\bauthorization\b\s*[:=]"),
    re.compile(r"(?i)\b(password|passwd|cookie)\b\s*[:=]"),
    re.compile(r"(?i)\bapi[_-]?key\b[\"']?\s*[:=]\s*[\"'][^\"']*[^_A-Za-z0-9\"']"),  # a key VALUE, not the env-var name
    re.compile(r"(?i)\btoken\b[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_\-]{8,}"),
    re.compile(r"://[^/\s@]+:[^/\s@]+@"),  # userinfo (credentials) inside a URL
)


def _git(args: list[str]) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extractor_version(atomic: dict) -> str:
    """Extraction-version pin: rule extractor uses RULE_VERSION (no prompt,
    no LLM), llm extractor uses the prompt version. Mirrors runner.py."""
    granularity = str(atomic.get("granularity", "coarse"))
    if str(atomic.get("extractor", "rule")) == "llm":
        return prompt_version(granularity)
    return f"{RULE_VERSION}-{granularity}"


def cache_namespace(model_id: str, granularity: str, endpoint_identity: str) -> str:
    """G45 namespace digest: the per-candidate fixed part of the extraction
    cache key — sha256 over (prompt_version, model_id, granularity,
    endpoint_identity). The per-unit content hash varies and is intentionally
    NOT part of the namespace."""
    material = "\x1f".join((prompt_version(granularity), model_id, granularity, endpoint_identity))
    return "g45:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_manifest(
    config_path: Path,
    *,
    generated_at: str | None = None,
    docker_digest: str | None = None,
    environment_summary: dict | None = None,
) -> dict:
    cfg = load_config(path=config_path)
    comp = cfg.competition_cfg
    atomic = cfg.atomic_cfg
    adapter = cfg.adapter_cfg
    granularity = str(atomic.get("granularity", "coarse"))
    model_id = str(adapter.get("official_model_id", ""))
    endpoint_identity = canonical_endpoint_identity(str(adapter.get("official_base_url", "")))

    lock_path = REPO / "requirements.lock"
    git_sha = _git(["rev-parse", "HEAD"])
    git_dirty = bool(
        _git(["status", "--porcelain", "--", ".", ":!competition/MANIFEST.json"]).strip()
    )
    descendant = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CODE_CANONICAL_SHA, git_sha],
        cwd=REPO,
        capture_output=True,
        check=False,
    ).returncode == 0

    manifest = {
        "candidate_id": str(comp.get("candidate_id", "UNKNOWN")),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "code_canonical_sha": CODE_CANONICAL_SHA,
        "code_canonical_is_ancestor": descendant,
        "competition_config_path": str(config_path.relative_to(REPO)),
        "competition_config_hash": "sha256:" + sha256_file(config_path),
        "competition_mode": cfg.competition_enabled,
        "config_fingerprint": cfg.behavior_fingerprint(),
        "track": str(comp.get("track", "UNKNOWN")),
        "submission_class": str(comp.get("submission_class", "UNKNOWN")),
        "top_k": comp.get("top_k", "UNKNOWN"),
        # model identity: CONFIGURED / REQUESTED are the same frozen literal
        # by construction and by the adapter's pre-call gate; UPSTREAM is
        # unknown until a real official call happens (zero calls this round).
        "configured_model": model_id,
        "requested_model": model_id,
        "upstream_reported_model": "UNKNOWN_NOT_OBSERVED",
        "endpoint_identity": endpoint_identity,
        "prompt_version": extractor_version(atomic),
        "cache_namespace": cache_namespace(model_id, granularity, endpoint_identity),
        "dependency_lock_hash": "sha256:" + sha256_file(lock_path),
        "docker_digest_if_available": docker_digest or "NOT_BUILT",
        "dataset_contract_version": "UNKNOWN",  # official snapshot does not version the dataset bundle
        "seed_if_any": 0,  # adapter body: temperature 0 + seed 0 on every official call
        "environment_summary": environment_summary
        or {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dense_backend": str(cfg.dense_cfg.get("backend", "")),
            "dense_backend_revision": str(cfg.dense_cfg.get("sentence_transformers_revision", "")),
            "dense_device_config": str(cfg.dense_cfg.get("device", "auto")),
        },
        "generated_at": generated_at or datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "official_rule_snapshot_id": str(
            comp.get("official_rule_snapshot_id", DEFAULT_RULE_SNAPSHOT_ID)
        ),
    }
    return manifest


def assert_secret_free(text: str) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise SystemExit(
                f"secret-hygiene violation: manifest text matches {pattern.pattern!r} — refusing to emit"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="10_config/competition.a1kr.json")
    parser.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--docker-digest", default=None)
    parser.add_argument("--environment-json", default=None)
    args = parser.parse_args()

    config_path = (REPO / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    environment = json.loads(args.environment_json) if args.environment_json else None
    manifest = build_manifest(
        config_path,
        generated_at=args.generated_at,
        docker_digest=args.docker_digest,
        environment_summary=environment,
    )
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert_secret_free(text)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"manifest written: {out_path}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
