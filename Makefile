PY := .venv/bin/python
PIP := .venv/bin/pip
PORT ?= 8093
BASE := http://127.0.0.1:$(PORT)

.DEFAULT_GOAL := help

.PHONY: help setup lint test serve smoke bench docker-build docker-serve docker-smoke docker-stop fetch-data

help:
	@echo "targets: setup lint test serve smoke bench docker-build docker-serve docker-smoke fetch-data"

setup:
	@test -d .venv || python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.lock
	$(PIP) install -q -e . --no-deps
	@echo "setup done"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest 08_tests

serve:
	$(PY) serve.py --port $(PORT)

smoke:
	@tmpdb=$$(mktemp -d)/smoke.db; \
	trap 'kill $$srv 2>/dev/null' EXIT; \
	$(PY) serve.py --port $(PORT) --set storage.db_path=$$tmpdb & srv=$$!; \
	$(PY) smoke.py --base-url $(BASE) --wait 30

fetch-data:
	$(PY) 04_bench/fetch_locomo.py

# One-command LOCAL benchmark: boots a throwaway service, runs the full
# LOCOMO proxy set, appends to the Experiment Ledger, tears down.
bench: setup
	@test -f 04_bench/data/locomo10.json || $(PY) 04_bench/fetch_locomo.py
	@tmpdb=$$(mktemp -d)/bench.db; \
	trap 'kill $$srv 2>/dev/null' EXIT; \
	$(PY) serve.py --port $(PORT) --set storage.db_path=$$tmpdb & srv=$$!; \
	$(PY) smoke.py --base-url $(BASE) --wait 30 && \
	$(PY) 04_bench/run_bench.py --base-url $(BASE) --label local-baseline

docker-build:
	@find . -name '._*' -not -path './.venv/*' -not -path './.git/*' -delete 2>/dev/null || true
	docker build -t zhiji-aml:v0 -f 05_docker/Dockerfile .

docker-serve:
	docker compose -f 05_docker/compose.yml up -d --build

docker-stop:
	docker compose -f 05_docker/compose.yml down

# G10: clean-environment build + run + smoke, entirely inside the image.
docker-smoke: docker-build
	docker run --rm --entrypoint bash zhiji-aml:v0 05_docker/smoke_in_image.sh
