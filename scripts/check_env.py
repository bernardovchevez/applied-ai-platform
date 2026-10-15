#!/usr/bin/env python3
"""Check .env and runtime environment configuration without printing secrets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import LAB_ROOT, app_env, available_providers, get_env, is_local, state_store_backend
from shared.embeddings import active_embedding_model_id, embedding_provider
from shared.llm import llm_provider

ENV_PATH = LAB_ROOT / ".env"


def _file_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


def check(key: str) -> None:
    file_val = _file_value(ENV_PATH, key)
    shell_val = os.environ.get(key)
    loaded = get_env(key)

    print(f"\n{key}")
    print(f"  .env file exists:     {ENV_PATH.exists()}")
    print(f"  value in .env file:   {'yes' if file_val else 'EMPTY or missing'}")
    if file_val:
        print(f"  .env value length:    {len(file_val)}")
    print(f"  shell env set:        {'yes' if shell_val is not None else 'no'}")
    print(f"  loaded by get_env():  {'yes' if loaded else 'NO'}")


def main() -> None:
    active_env = app_env()
    env_override_path = LAB_ROOT / f".env.{active_env}"

    print("Applied AI Platform - Environment Check")
    print(f"Python:                 {sys.executable}")
    print(f"APP_ENV:                {active_env}")
    print(f"Base .env present:      {ENV_PATH.exists()}")
    print(f"Override .env.{active_env}:  {env_override_path.exists()}")
    print(f"State store backend:    {state_store_backend()}")
    print(f"BFF API key configured: {'yes' if get_env('BFF_API_KEY') else 'no (open in localhost)'}")

    if is_local():
        from shared.ollama_client import base_url, is_available, list_models

        print("\nOllama (localhost)")
        print(f"  base URL:             {base_url()}")
        print(f"  reachable:            {is_available()}")
        if is_available():
            try:
                models = list_models()
                print(
                    f"  models pulled:        {', '.join(models) if models else '(none - run ollama pull llama3.2)'}"
                )
            except Exception as exc:
                print(f"  list models error:    {exc}")

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        check(name)

    print("\nActive Runtime Selection")
    print(f"  LLM provider:         {llm_provider()}")
    print(f"  Embedding provider:   {embedding_provider()}")
    print(f"  Embedding model ID:   {active_embedding_model_id()}")
    print(f"  Compare providers:    {', '.join(available_providers()) or '(none)'}")


if __name__ == "__main__":
    main()
