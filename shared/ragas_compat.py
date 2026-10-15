"""RAGAS import compatibility (ragas 0.4.x + langchain-community 0.4.x)."""

from __future__ import annotations

import sys
from types import ModuleType

from langchain_core.language_models.chat_models import BaseChatModel

from shared.config import get_env


def patch_ragas_imports() -> None:
    """Stub missing Vertex AI chat model so ``import ragas`` succeeds."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    stub = ModuleType(module_name)

    class ChatVertexAI(BaseChatModel):  # type: ignore[misc]
        @property
        def _llm_type(self) -> str:
            return "vertexai-stub"

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = stub


def resolve_judge_provider(explicit: str | None = None) -> str:
    """Pick RAGAS judge: openai preferred when key set, else ollama."""
    if explicit:
        return explicit.lower()
    env = get_env("RAGAS_JUDGE_PROVIDER") or get_env("LLM_PROVIDER")
    if env:
        return env.lower()
    if get_env("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def get_ragas_llm(provider: str | None = None):
    """RAGAS judge LLM — OpenAI (recommended) or Ollama fallback."""
    provider = resolve_judge_provider(provider)

    if provider == "openai":
        from openai import OpenAI
        from ragas.llms import llm_factory

        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("RAGAS judge provider openai requires OPENAI_API_KEY")
        model = get_env("RAGAS_JUDGE_MODEL", get_env("CHAT_MODEL", "gpt-4o-mini"))
        client = OpenAI(api_key=api_key)
        return llm_factory(model, client=client), f"openai/{model}"

    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    model = get_env("OLLAMA_CHAT_MODEL", "llama3.2")
    return LangchainLLMWrapper(ChatOllama(model=model, temperature=0)), f"ollama/{model}"


def get_ragas_embeddings(provider: str | None = None):
    """RAGAS embeddings — match judge provider when possible."""
    provider = resolve_judge_provider(provider)

    if provider == "openai" and get_env("OPENAI_API_KEY"):
        from openai import OpenAI
        from ragas.embeddings import OpenAIEmbeddings

        client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
        model = get_env("RAGAS_EMBED_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(client=client, model=model), f"openai/{model}"

    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    model = get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return LangchainEmbeddingsWrapper(OllamaEmbeddings(model=model)), f"ollama/{model}"
