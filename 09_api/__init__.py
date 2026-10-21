"""aml_api — HTTP surface (architecture layer 08)."""

from .app import VERSION, create_app
from .service import MemoryService

__all__ = ["VERSION", "create_app", "MemoryService"]
