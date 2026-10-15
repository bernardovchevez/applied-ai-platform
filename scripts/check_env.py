#!/usr/bin/env python3
"""Check .env loading without printing secrets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import LAB_ROOT, get_env

ENV_PATH = LAB_ROOT / ".env"


def _file_value(key: str) -> str | None:
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


def check(key: str) -> None:
    file_val = _file_value(key)
    shell_val = os.environ.get(key)
    loaded = get_env(key)

    print(f"\n{key}")
    print(f"  .env file exists:     {ENV_PATH.exists()}")
    print(f"  .env path:            {ENV_PATH}")
    print(f"  value in .env file:   {'yes' if file_val else 'EMPTY or missing'}")
    if file_val:
        print(f"  .env value length:    {len(file_val)}")
        print(f"  .env format ok:       {file_val.startswith('sk-') if 'KEY' in key else bool(file_val)}")
    print(f"  shell env set:        {'yes' if shell_val is not None else 'no'}")
    if shell_val is not None:
        print(f"  shell value length:   {len(shell_val)}")
    print(f"  loaded by get_env():  {'yes' if loaded else 'NO'}")


def main() -> None:
    print("Applied AI Platform — env check")
    print(f"Python: {sys.executable}")

    from shared.ollama_client import base_url, is_available, list_models

    print(f"\nOllama")
    print(f"  base URL:             {base_url()}")
    print(f"  reachable:            {is_available()}")
    if is_available():
        try:
            models = list_models()
            print(f"  models pulled:        {', '.join(models) if models else '(none — run ollama pull llama3.2)'}")
        except Exception as exc:
            print(f"  list models error:    {exc}")

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        check(name)

    from shared.config import available_providers
    from shared.embeddings import embedding_provider
    from shared.llm import llm_provider

    print(f"\nActive (auto-detect)")
    print(f"  LLM provider:         {llm_provider()}")
    print(f"  Embedding provider:   {embedding_provider()}")
    print(f"  Compare providers:    {', '.join(available_providers()) or '(none)'}")

    if not available_providers():
        print("\nFix:")
        print("  Cloud: add keys to .env")
        print("  Local: brew install ollama && ollama serve && ollama pull llama3.2")
    elif not get_env("OPENAI_API_KEY") and is_available():
        print("\nTip: Using Ollama locally. Set LLM_PROVIDER=ollama in .env to skip OpenAI.")


if __name__ == "__main__":
    main()
