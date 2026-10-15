"""Local Ollama API client (chat + embeddings)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from shared.config import get_env

DEFAULT_BASE_URL = "http://localhost:11434"


def base_url() -> str:
    return (get_env("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def chat_model() -> str:
    return get_env("OLLAMA_CHAT_MODEL", "llama3.2")


def embed_model() -> str:
    return get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def chat_temperature() -> float:
    return float(get_env("OLLAMA_TEMPERATURE", "0"))


def _chat_options(num_predict: int) -> dict[str, Any]:
    return {"num_predict": num_predict, "temperature": chat_temperature()}


def is_available(timeout: float = 2.0) -> bool:
    """True if Ollama is running and reachable."""
    if get_env("OLLAMA_ENABLED", "true").lower() == "false":
        return False
    try:
        r = httpx.get(f"{base_url()}/api/tags", timeout=timeout)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def list_models(timeout: float = 5.0) -> list[str]:
    r = httpx.get(f"{base_url()}/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return [m["name"] for m in data.get("models", [])]


def chat(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    num_predict: int = 512,
) -> tuple[str, float]:
    """Send a chat prompt; returns (text, latency_seconds)."""
    model = model or chat_model()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    r = httpx.post(
        f"{base_url()}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": _chat_options(num_predict),
        },
        timeout=120.0,
    )
    r.raise_for_status()
    elapsed = time.perf_counter() - start
    data = r.json()
    text = data.get("message", {}).get("content", "")
    return text, elapsed


def chat_stream(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    num_predict: int = 512,
) -> Iterator[str]:
    """Stream chat tokens from Ollama (/api/chat, stream=true)."""
    model = model or chat_model()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    with httpx.stream(
        "POST",
        f"{base_url()}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": _chat_options(num_predict),
        },
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content")
            if token:
                yield token
            if data.get("done"):
                break


def _normalize_texts(texts: str | list[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _request_embeddings(base: str, model: str, texts: list[str]) -> list[list[float]]:
    """Call Ollama embed API (new /api/embed, fallback /api/embeddings)."""
    if not texts:
        return []

    # Prefer newer batch-friendly endpoint
    payload: dict[str, Any] = {"model": model, "input": texts if len(texts) > 1 else texts[0]}
    r = httpx.post(f"{base}/api/embed", json=payload, timeout=120.0)
    if r.status_code == 200:
        embeddings = r.json().get("embeddings")
        if embeddings:
            return embeddings

    # Legacy per-text endpoint
    vectors: list[list[float]] = []
    for text in texts:
        r = httpx.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60.0,
        )
        r.raise_for_status()
        vectors.append(r.json()["embedding"])
    return vectors


def embed(texts: str | list[str], model: str | None = None) -> list[list[float]]:
    """Embed one or more texts via Ollama."""
    model = model or embed_model()
    return _request_embeddings(base_url(), model, _normalize_texts(texts))


class OllamaEmbeddingFunction:
    """ChromaDB-compatible embedding function using Ollama."""

    def __init__(self, model: str | None = None, base: str | None = None) -> None:
        self.model = model or embed_model()
        self._base = (base or base_url()).rstrip("/")

    def __call__(self, input: list[str]) -> list[list[float]]:
        return _request_embeddings(self._base, self.model, _normalize_texts(input))

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        """ChromaDB passes a list of query strings; must return list of vectors."""
        return self.__call__(_normalize_texts(input))

    @staticmethod
    def name() -> str:
        return "ollama"

    def get_config(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self._base}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> OllamaEmbeddingFunction:
        return OllamaEmbeddingFunction(
            model=config.get("model"),
            base=config.get("base_url"),
        )
