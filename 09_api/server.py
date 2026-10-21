"""Uvicorn entrypoint: `python -m aml_api.server` (or uvicorn aml_api.app:app)."""

from __future__ import annotations

import os

import uvicorn
from aml_config import load_config, merge_set_overrides

from .app import create_app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="zhiji-aml V0 service")
    parser.add_argument("--config", default=os.environ.get("AML_CONFIG_PATH"), help="JSON config file")
    parser.add_argument("--set", dest="sets", action="append", default=[], help="dotted.path=value override")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    overrides = merge_set_overrides(args.sets)
    if args.host:
        overrides.setdefault("server", {})["host"] = args.host
    if args.port is not None:
        overrides.setdefault("server", {})["port"] = args.port
    if os.environ.get("AML_HOST"):
        overrides.setdefault("server", {})["host"] = os.environ["AML_HOST"]
    if os.environ.get("AML_PORT"):
        overrides.setdefault("server", {})["port"] = int(os.environ["AML_PORT"])

    cfg = load_config(path=args.config, overrides=overrides or None)
    app = create_app(cfg)
    server = cfg.server_cfg
    uvicorn.run(app, host=server.get("host", "127.0.0.1"), port=int(server.get("port", 8093)), workers=1)


if __name__ == "__main__":
    main()
