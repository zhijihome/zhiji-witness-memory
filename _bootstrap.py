"""Import bootstrap for running from a source checkout without `pip install -e .`.

Maps the numbered layer directories to their package names, mirroring the
[tool.setuptools] package-dir mapping in pyproject.toml (keep in sync).

Usage (tests, bench scripts, dev shells):

    import _bootstrap  # noqa: F401  — must precede any aml_* import
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

_LAYER_PACKAGES = {
    # install order matters: dependencies first
    "aml_config": "10_config",
    "aml_store": "01_store",
    "aml_adapter": "03_adapter",
    "aml_atomic": "11_atomic",
    "aml_governance": "12_governance",
    "aml_retrieval": "02_retrieval",
    "aml_api": "09_api",
}


def _install() -> None:
    for pkg, folder in _LAYER_PACKAGES.items():
        if pkg in sys.modules:
            continue
        init = _ROOT / folder / "__init__.py"
        if not init.exists():
            # layer not written yet — skip so partial checkouts still work
            continue
        spec = importlib.util.spec_from_file_location(
            pkg, init, submodule_search_locations=[str(_ROOT / folder)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = module
        spec.loader.exec_module(module)


_install()
