"""Environment and path configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_ROOT / "data" / "synthetic"
SYNTHETIC_META = DATA_DIR / "sharepoint-meta.json"

# override=True: .env wins over empty exported vars in the shell
load_dotenv(LAB_ROOT / ".env", override=True)


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}. Copy .env.example to .env")
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
    if ollama_available():
        providers.append("ollama")
    return providers
