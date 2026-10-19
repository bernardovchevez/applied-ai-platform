"""BFF path constants and sys.path setup for monorepo imports."""

from __future__ import annotations

import sys
from pathlib import Path

BFF_DIR = Path(__file__).resolve().parent
LAB_ROOT = BFF_DIR.parent.parent
OPENAPI_PATH = BFF_DIR / "openapi.yaml"
PROMPTS_DIR = LAB_ROOT / "guardrails" / "prompts"
API_VERSION = "0.1.0"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

_PATHS_CONFIGURED = False


def configure_import_paths() -> None:
    global _PATHS_CONFIGURED
    if _PATHS_CONFIGURED:
        return
    for entry in (LAB_ROOT, LAB_ROOT / "rag", BFF_DIR.parent, BFF_DIR):
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_CONFIGURED = True
