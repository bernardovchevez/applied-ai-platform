#!/usr/bin/env python3
"""promptfoo Python provider calling RAG backends."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
RAG_DIR = LAB_ROOT / "rag"
sys.path.insert(0, str(LAB_ROOT))


def _load_query_module(backend: str):
    files = {
        "chroma": "query.py",
        "langchain": "query_langchain.py",
        "llamaindex": "query_llamaindex.py",
    }
    if backend not in files:
        raise ValueError(f"Unknown backend: {backend}. Use chroma, langchain, or llamaindex.")

    path = RAG_DIR / files[backend]
    spec = importlib.util.spec_from_file_location(f"rag_{backend}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RUNNERS: dict[str, object] = {}


def _get_runner(backend: str):
    if backend in _RUNNERS:
        return _RUNNERS[backend]

    mod = _load_query_module(backend)
    if backend == "chroma":
        runner = mod.answer
    elif backend == "langchain":
        vectorstore = mod.get_vectorstore()
        chain = mod.build_chain(vectorstore)
        runner = lambda q: mod.answer(q, chain)
    else:
        index = mod.load_index()
        query_engine = mod.build_query_engine(index)
        runner = lambda q: mod.answer(q, query_engine)

    _RUNNERS[backend] = runner
    return runner


def call_api(prompt, options, context):
    """promptfoo entrypoint; prompt is {{question}} from config."""
    config = (options or {}).get("config") or {}
    backend = config.get("backend", "chroma")
    vars_ = (context or {}).get("vars") or {}
    question = vars_.get("question") or prompt

    if not question or not str(question).strip():
        return {"error": "Missing question in test vars or prompt"}

    try:
        result = _get_runner(backend)(str(question).strip())
        payload = {
            "answer": result.get("answer") or "",
            "sources": result.get("sources") or [],
            "backend": backend,
        }
        return {
            "output": json.dumps(payload, ensure_ascii=False),
            "metadata": {"backend": backend, "sources": payload["sources"]},
        }
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    # Smoke test: python promptfoo_provider.py
    demo = call_api(
        "",
        {"config": {"backend": "chroma"}},
        {"vars": {"question": "Is the theme song cleared for EMEA streaming?"}},
    )
    print(json.dumps(demo, indent=2))
