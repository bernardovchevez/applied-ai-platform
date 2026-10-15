"""Optional Langfuse tracing for BFF routes.

No-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are unset — safe for local dev
without a Langfuse project.
"""

from __future__ import annotations

import logging
from typing import Any

from shared.config import get_env

logger = logging.getLogger(__name__)

_client: Any | None = None
_init_attempted = False


def langfuse_enabled() -> bool:
    return bool(get_env("LANGFUSE_PUBLIC_KEY") and get_env("LANGFUSE_SECRET_KEY"))


def active_chat_model() -> str:
    from shared.llm import llm_provider
    from shared.ollama_client import chat_model

    if llm_provider() == "ollama":
        return chat_model()
    return get_env("CHAT_MODEL", "gpt-4o-mini")


def get_langfuse_client() -> Any | None:
    global _client, _init_attempted
    if not langfuse_enabled():
        return None
    if _init_attempted:
        return _client
    _init_attempted = True
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=get_env("LANGFUSE_PUBLIC_KEY"),
            secret_key=get_env("LANGFUSE_SECRET_KEY"),
            host=get_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as exc:
        logger.warning("Langfuse init failed: %s", exc)
        _client = None
    return _client


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if not client:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Langfuse flush failed: %s", exc)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class ChatStreamTrace:
    """Records POST /v1/chat/stream: retriever span + generation + root trace."""

    def __init__(
        self,
        *,
        session_id: str,
        question: str,
        prompt_version: str,
        client_platform: str | None,
        top_k: int,
        format: str,
    ) -> None:
        self._client = get_langfuse_client()
        self._root: Any | None = None
        self._retrieve_obs: Any | None = None
        self._generation_obs: Any | None = None
        if not self._client:
            return
        try:
            from langfuse.types import TraceContext

            trace_id = self._client.create_trace_id()
            ctx = TraceContext(trace_id=trace_id, session_id=session_id)
            self._root = self._client.start_observation(
                name="v1/chat/stream",
                trace_context=ctx,
                input={
                    "question": question,
                    "top_k": top_k,
                    "format": format,
                },
                metadata={
                    "prompt_version": prompt_version,
                    "client_platform": client_platform,
                },
            )
            self._retrieve_obs = self._root.start_observation(
                name="pgvector.retrieve",
                as_type="retriever",
                input={"question": question, "top_k": top_k},
            )
        except Exception as exc:
            logger.warning("Langfuse chat trace start failed: %s", exc)
            self._root = None
            self._retrieve_obs = None

    def end_retrieve(self, sources: list[dict[str, Any]], hit_count: int) -> None:
        if not self._retrieve_obs:
            return
        try:
            self._retrieve_obs.update(
                output={
                    "hit_count": hit_count,
                    "source_count": len(sources),
                    "sources": sources,
                }
            )
            self._retrieve_obs.end()
        except Exception as exc:
            logger.warning("Langfuse retrieve span failed: %s", exc)
        finally:
            self._retrieve_obs = None

    def start_generation(
        self,
        *,
        model: str,
        system: str,
        user_msg: str,
        max_tokens: int,
    ) -> None:
        if not self._root:
            return
        try:
            self._generation_obs = self._root.start_observation(
                name="llm.stream",
                as_type="generation",
                model=model,
                model_parameters={"max_tokens": max_tokens},
                input=[
                    {"role": "system", "content": _truncate(system, 500)},
                    {"role": "user", "content": _truncate(user_msg, 2000)},
                ],
            )
        except Exception as exc:
            logger.warning("Langfuse generation start failed: %s", exc)

    def end_success(
        self,
        *,
        final_answer: str,
        latency_ms: int,
        provider: str,
        prompt_version: str,
    ) -> None:
        if self._generation_obs:
            try:
                self._generation_obs.update(output={"text": final_answer})
                self._generation_obs.end()
            except Exception as exc:
                logger.warning("Langfuse generation end failed: %s", exc)
            finally:
                self._generation_obs = None
        if self._root:
            try:
                self._root.update(
                    output={
                        "final_answer": final_answer,
                        "latency_ms": latency_ms,
                        "provider": provider,
                        "prompt_version": prompt_version,
                    }
                )
                self._root.end()
            except Exception as exc:
                logger.warning("Langfuse root end failed: %s", exc)
            finally:
                self._root = None
        flush_langfuse()

    def end_error(self, *, error_type: str, message: str, stage: str) -> None:
        status = f"{stage}: {error_type} — {message}"
        for obs in (self._generation_obs, self._retrieve_obs, self._root):
            if not obs:
                continue
            try:
                obs.update(level="ERROR", status_message=status)
                obs.end()
            except Exception:
                pass
        self._generation_obs = None
        self._retrieve_obs = None
        self._root = None
        flush_langfuse()


class AgentRunTrace:
    """Records POST /v1/agent/run: agent span + tool observations."""

    def __init__(
        self,
        *,
        session_id: str,
        prompt: str | None,
        client_platform: str | None,
        auto_approve: bool = False,
        resume: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self._client = get_langfuse_client()
        self._root: Any | None = None
        self._agent_obs: Any | None = None
        if not self._client:
            return
        try:
            from langfuse.types import TraceContext

            trace_id = self._client.create_trace_id()
            ctx = TraceContext(trace_id=trace_id, session_id=session_id)
            self._root = self._client.start_observation(
                name="v1/agent/run",
                trace_context=ctx,
                input={
                    "prompt": prompt,
                    "auto_approve": auto_approve,
                    "resume": resume,
                    "thread_id": thread_id,
                },
                metadata={
                    "client_platform": client_platform,
                    "thread_id": thread_id,
                },
            )
            if resume:
                return
            self._agent_obs = self._root.start_observation(
                name="agents.workflow",
                as_type="agent",
                input={"prompt": prompt},
                metadata={"auto_approve": auto_approve},
            )
        except Exception as exc:
            logger.warning("Langfuse agent trace start failed: %s", exc)
            self._root = None
            self._agent_obs = None

    def record_tool_trace(self, tool_trace: list[dict[str, Any]]) -> None:
        parent = self._agent_obs or self._root
        if not parent:
            return
        for entry in tool_trace:
            tool_name = str(entry.get("tool") or "unknown")
            try:
                obs = parent.start_observation(
                    name=tool_name,
                    as_type="tool",
                    input=entry,
                    output=entry,
                )
                obs.end()
            except Exception as exc:
                logger.warning("Langfuse tool span %s failed: %s", tool_name, exc)

    def end_completed(
        self,
        *,
        answer: str,
        sources: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        latency_ms: int,
        status: str,
    ) -> None:
        self.record_tool_trace(tool_trace)
        if self._agent_obs:
            try:
                self._agent_obs.update(
                    output={
                        "answer": _truncate(answer, 4000),
                        "source_count": len(sources),
                        "status": status,
                    }
                )
                self._agent_obs.end()
            except Exception as exc:
                logger.warning("Langfuse agent span end failed: %s", exc)
            finally:
                self._agent_obs = None
        if self._root:
            try:
                self._root.update(
                    output={
                        "answer": _truncate(answer, 4000),
                        "latency_ms": latency_ms,
                        "status": status,
                        "tool_count": len(tool_trace),
                    }
                )
                self._root.end()
            except Exception as exc:
                logger.warning("Langfuse agent root end failed: %s", exc)
            finally:
                self._root = None
        flush_langfuse()

    def end_paused(
        self,
        *,
        thread_id: str,
        proposed_output: str,
        tool_trace: list[dict[str, Any]],
        latency_ms: int,
    ) -> None:
        self.record_tool_trace(tool_trace)
        if self._agent_obs:
            try:
                self._agent_obs.update(
                    output={
                        "proposed_output": _truncate(proposed_output, 4000),
                        "status": "awaiting_approval",
                    }
                )
                self._agent_obs.end()
            except Exception as exc:
                logger.warning("Langfuse agent span pause failed: %s", exc)
            finally:
                self._agent_obs = None
        if self._root:
            try:
                self._root.update(
                    metadata={"thread_id": thread_id},
                    output={
                        "thread_id": thread_id,
                        "status": "paused",
                        "latency_ms": latency_ms,
                        "tool_count": len(tool_trace),
                    },
                )
                self._root.end()
            except Exception as exc:
                logger.warning("Langfuse agent root pause failed: %s", exc)
            finally:
                self._root = None
        flush_langfuse()

    def end_resume(
        self,
        *,
        decision: str,
        answer: str,
        tool_trace: list[dict[str, Any]],
    ) -> None:
        self.record_tool_trace(tool_trace)
        if self._root:
            try:
                self._root.update(
                    output={
                        "decision": decision,
                        "answer": _truncate(answer, 4000) if answer else "",
                        "status": decision,
                        "tool_count": len(tool_trace),
                    }
                )
                self._root.end()
            except Exception as exc:
                logger.warning("Langfuse agent resume end failed: %s", exc)
            finally:
                self._root = None
        flush_langfuse()

    def end_error(self, *, error_type: str, message: str, stage: str) -> None:
        status = f"{stage}: {error_type} — {message}"
        for obs in (self._agent_obs, self._root):
            if not obs:
                continue
            try:
                obs.update(level="ERROR", status_message=status)
                obs.end()
            except Exception:
                pass
        self._agent_obs = None
        self._root = None
        flush_langfuse()
