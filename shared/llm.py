"""Unified LLM chat for OpenAI or local Ollama."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from shared.config import get_env, is_local, require_env
from shared.ollama_client import chat as ollama_chat
from shared.ollama_client import chat_stream as ollama_chat_stream
from shared.ollama_client import is_available as ollama_available


def llm_provider() -> str:
    """Active provider: ollama | openai (env LLM_PROVIDER overrides auto-detect).

    In stage and production, defaults to openai unless OLLAMA_IN_CLOUD=true.
    """
    explicit = (get_env("LLM_PROVIDER") or "").lower()
    if not is_local():
        if explicit and explicit != "ollama":
            return explicit
        if explicit == "ollama" and get_env("OLLAMA_IN_CLOUD") == "true":
            return "ollama"
        return "openai"
    if explicit:
        return explicit
    if get_env("OPENAI_API_KEY"):
        return "openai"
    if ollama_available():
        return "ollama"
    return "openai"


def chat_completion(
    user: str,
    system: str | None = None,
    max_tokens: int = 600,
) -> str:
    provider = llm_provider()
    if provider == "ollama":
        text, _ = ollama_chat(user, system=system)
        return text

    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    response = client.chat.completions.create(
        model=get_env("CHAT_MODEL", "gpt-4o-mini"),
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def stream_chat_completion(
    user: str,
    system: str | None = None,
    max_tokens: int = 600,
) -> Iterator[str]:
    """Yield LLM tokens for streaming BFF / UI clients."""
    provider = llm_provider()
    if provider == "ollama":
        yield from ollama_chat_stream(user, system=system, num_predict=max_tokens)
        return

    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    stream = client.chat.completions.create(
        model=get_env("CHAT_MODEL", "gpt-4o-mini"),
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
