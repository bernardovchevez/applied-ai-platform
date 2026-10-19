"""Media Ops Applied AI Platform BFF.

Run from this directory:
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

_bff_dir = Path(__file__).resolve().parent
if str(_bff_dir) not in sys.path:
    sys.path.insert(0, str(_bff_dir))

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bootstrap import API_VERSION, OPENAPI_PATH, configure_import_paths, DEFAULT_CORS_ORIGINS

configure_import_paths()

from shared.config import get_env, is_production  # noqa: E402

from routes import agent, chat, evals, feedback_http, meta, retrieve  # noqa: E402


def _docs_enabled() -> bool:
    explicit = get_env("BFF_ENABLE_DOCS")
    if explicit is not None:
        return explicit.strip().lower() in ("1", "true", "yes")
    return not is_production()


def _cors_origins() -> list[str]:
    raw = get_env("BFF_CORS_ORIGINS")
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Media Ops Applied AI Platform BFF",
    version=API_VERSION,
    description=(
        "Presentation layer for web, mobile, and CTV clients. "
        "Platform logic lives in applied-ai-platform (pgvector, agents, MCP)."
    ),
    docs_url="/docs" if _docs_enabled() else None,
    redoc_url="/redoc" if _docs_enabled() else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-Client-Platform",
        "X-Session-Id",
        "Authorization",
        "X-API-Key",
    ],
    expose_headers=["X-Prompt-Version", "X-Embed-Model", "X-Environment"],
)


def _custom_openapi() -> dict:
    if OPENAPI_PATH.exists():
        return yaml.safe_load(OPENAPI_PATH.read_text(encoding="utf-8"))
    return app.openapi_schema or {}


app.openapi = _custom_openapi  # type: ignore[method-assign]

app.include_router(meta.router)
app.include_router(retrieve.router)
app.include_router(chat.router)
app.include_router(feedback_http.router)
app.include_router(evals.router)
app.include_router(agent.router)
