#!/usr/bin/env bash
# In-image smoke (G10): boots the service inside the container, runs the
# zero-dependency smoke client plus the contract test subset, exits nonzero on
# any failure. Run with:
#   docker run --rm --entrypoint bash zhiji-aml:v0 05_docker/smoke_in_image.sh
set -euo pipefail

cd /app
python -m aml_api.server --host 127.0.0.1 --port 8093 --set storage.db_path=/tmp/smoke.db &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

python smoke.py --base-url http://127.0.0.1:8093 --wait 60

# contract + isolation + knobs subset of the full suite, inside the image
python -m pytest 08_tests -k "api_contract or store or knobs" -q

echo "IN-IMAGE SMOKE PASS"
