"""Environment and path configuration (localhost, stage, production)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

AppEnv = Literal["localhost", "stage", "production"]

LAB_ROOT = Path(__file__).resolve().parent.parent
LAKE_RAW_DIR = Path(
    os.getenv("LAKE_RAW_DIR", str(LAB_ROOT / "data" / "lake" / "raw"))
).resolve()
SYNTHETIC_DIR = LAB_ROOT / "data" / "synthetic"
# Published corpus (default): lake bronze written by applied-ai-docs publish pipeline.
DATA_DIR = Path(os.getenv("DATA_DIR", str(LAKE_RAW_DIR))).resolve()
LAKE_META = LAKE_RAW_DIR / "sharepoint-meta.json"
SYNTHETIC_META = SYNTHETIC_DIR / "sharepoint-meta.json"

_ENV_ALIASES: dict[str, AppEnv] = {
    "local": "localhost",
    "localhost": "localhost",
    "dev": "localhost",
    "development": "localhost",
    "stage": "stage",
    "staging": "stage",
    "prod": "production",
    "production": "production",
}


def normalize_app_env(raw: str | None) -> AppEnv:
    if not raw:
        return "localhost"
    return _ENV_ALIASES.get(raw.strip().lower(), "localhost")


def _load_env_files() -> AppEnv:
    """Load base .env and optional .env.<APP_ENV> override."""
    base_env = LAB_ROOT / ".env"
    if base_env.exists():
        load_dotenv(base_env, override=False)

    current_env = normalize_app_env(os.getenv("APP_ENV"))
    env_specific = LAB_ROOT / f".env.{current_env}"
    if env_specific.exists():
        load_dotenv(env_specific, override=True)

    # Re-evaluate in case .env set APP_ENV
    return normalize_app_env(os.getenv("APP_ENV"))


_ACTIVE_ENV: AppEnv = _load_env_files()


def app_env() -> AppEnv:
    """Return canonical runtime environment: localhost | stage | production."""
    return normalize_app_env(os.getenv("APP_ENV", _ACTIVE_ENV))


def is_local() -> bool:
    return app_env() == "localhost"


def is_stage() -> bool:
    return app_env() == "stage"


def is_production() -> bool:
    return app_env() == "production"


def state_store_backend() -> Literal["sqlite", "postgres"]:
    """Return persistence backend for BFF sessions and feedback.

    Defaults to sqlite on localhost and postgres on stage/production.
    Override explicitly with STATE_STORE_BACKEND=sqlite|postgres.
    """
    explicit = (os.getenv("STATE_STORE_BACKEND") or "").strip().lower()
    if explicit in ("postgres", "postgresql", "pg"):
        return "postgres"
    if explicit == "sqlite":
        return "sqlite"
    return "sqlite" if is_local() else "postgres"


def get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return val


def require_env(key: str) -> str:
    value = get_env(key)
    if not value:
        raise RuntimeError(
            f"Missing required env var: {key} (APP_ENV={app_env()}). "
            "Configure it in your environment or .env file."
        )
    return value


def available_providers() -> list[str]:
    from shared.ollama_client import is_available as ollama_available

    providers = []
    if get_env("OPENAI_API_KEY"):
        providers.append("openai")
    if get_env("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if get_env("GOOGLE_API_KEY"):
        providers.append("gemini")
    if is_local() and ollama_available():
        providers.append("ollama")
    return providers

