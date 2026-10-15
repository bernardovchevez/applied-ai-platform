"""Embedding backend for ChromaDB and pgvector (OpenAI or Ollama)."""

from __future__ import annotations

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from shared.config import get_env, is_local, require_env
from shared.ollama_client import OllamaEmbeddingFunction, embed_model as ollama_embed_model
from shared.ollama_client import is_available as ollama_available


def embedding_provider() -> str:
    """Active embedding provider: ollama | openai.

    In stage and production, defaults to openai unless OLLAMA_IN_CLOUD=true.
    """
    explicit = (get_env("EMBEDDING_PROVIDER") or "").lower()
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


def get_embedding_function():
    provider = embedding_provider()
    if provider == "ollama":
        if not ollama_available():
            raise RuntimeError(
                "Ollama not reachable. Run: ollama serve && ollama pull nomic-embed-text"
            )
        return OllamaEmbeddingFunction()
    require_env("OPENAI_API_KEY")
    return OpenAIEmbeddingFunction(
        api_key=get_env("OPENAI_API_KEY"),
        model_name=get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
    )


# Known dimensions for pgvector schema
_EMBEDDING_DIMS: dict[str, int] = {
    "ollama": 768,  # nomic-embed-text
    "openai": 1536,  # text-embedding-3-small
}


def embedding_dimensions() -> int:
    """Return vector width for the active embedding provider."""
    provider = embedding_provider()
    if provider == "ollama":
        return _EMBEDDING_DIMS["ollama"]
    return _EMBEDDING_DIMS["openai"]


def active_embedding_model_id() -> str:
    """Return a versioned identifier for the active embedding model and vector width."""
    provider = embedding_provider()
    dim = embedding_dimensions()
    if provider == "ollama":
        return f"ollama-{ollama_embed_model()}-{dim}"
    model = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    return f"openai-{model}-{dim}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via the configured provider (pgvector/Pinecone stores)."""
    if not texts:
        return []
    fn = get_embedding_function()
    return fn(texts)
