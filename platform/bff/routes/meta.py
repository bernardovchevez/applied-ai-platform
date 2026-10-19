"""Health, version, OpenAPI file, root."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from bootstrap import API_VERSION, OPENAPI_PATH
from health_checks import build_health_report
from models import HealthResponse
from rag_helpers import prompt_version
from shared.config import app_env, state_store_backend
from shared.embeddings import active_embedding_model_id

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    responses={503: {"model": HealthResponse}},
)
def health() -> HealthResponse | JSONResponse:
    report = build_health_report(version=API_VERSION)
    response = HealthResponse.model_validate(report)
    if response.status == "degraded":
        return JSONResponse(status_code=503, content=response.model_dump(exclude_none=True))
    return response


@router.get("/v1/version")
def version() -> JSONResponse:
    pv = prompt_version()
    embed_model = active_embedding_model_id()
    env_name = app_env()

    body = {
        "api_version": API_VERSION,
        "environment": env_name,
        "state_store": state_store_backend(),
        "prompt_version": pv,
        "embed_model": embed_model,
        "index_document_count": None,
    }
    return JSONResponse(
        content=body,
        headers={
            "X-Prompt-Version": pv,
            "X-Embed-Model": embed_model,
            "X-Environment": env_name,
        },
    )


@router.get("/openapi.yaml", include_in_schema=False)
def serve_openapi_file() -> FileResponse:
    return FileResponse(OPENAPI_PATH, media_type="application/yaml")


@router.get("/", include_in_schema=False)
def root() -> PlainTextResponse:
    return PlainTextResponse(
        f"Media Ops AI Platform BFF v{API_VERSION} ({app_env()})\n"
        "Docs: /docs\n"
        "OpenAPI: /openapi.yaml\n"
        "Health: /health\n"
    )
