"""FastAPI dependencies."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from shared.config import get_env


def verify_bff_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Verify BFF_API_KEY when configured (stage/production). No-op when unset."""
    expected = get_env("BFF_API_KEY")
    if not expected:
        return

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Valid BFF API key required."},
        )
