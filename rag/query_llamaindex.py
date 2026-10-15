#!/usr/bin/env python3
"""LlamaIndex RAG over synthetic media ops docs (Ollama)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from shared.config import DATA_DIR, get_env

STORAGE_DIR = Path(__file__).parent / ".llamaindex_storage"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "guardrails" / "prompts"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def build_splitter() -> LangchainNodeParser:
    """Match LangChain path: 512 characters, 50 overlap (not token-based)."""
    return LangchainNodeParser(
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )


def configure_settings() -> None:
    chat_model = get_env("OLLAMA_CHAT_MODEL", "llama3.2")
    embed_model = get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    Settings.llm = Ollama(model=chat_model, request_timeout=120.0, temperature=0)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model, request_timeout=120.0)


def load_system_prompt() -> str:
    system_path = PROMPTS_DIR / "system_rag.txt"
    return system_path.read_text() if system_path.exists() else "Answer from context only."


def build_qa_prompt() -> PromptTemplate:
    system = load_system_prompt()
    template = (
        f"{system}\n\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    return PromptTemplate(template)


def load_documents(data_dir: Path):
    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=True,
        required_exts=[".md", ".txt"],
    )
    docs = reader.load_data()
    if not docs:
        raise RuntimeError(f"No .md/.txt documents found in {data_dir}")
    return docs


def build_index(data_dir: Path | None = None, reset: bool = True) -> VectorStoreIndex:
    data_dir = data_dir or DATA_DIR
    configure_settings()

    if reset and STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)

    documents = load_documents(data_dir)
    nodes = build_splitter().get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    print(
        f"LlamaIndex index: {len(nodes)} nodes from {len(documents)} docs → {STORAGE_DIR}"
    )
    return index


def load_index() -> VectorStoreIndex:
    configure_settings()
    if not STORAGE_DIR.exists() or not any(STORAGE_DIR.iterdir()):
        raise RuntimeError("Index not found. Run with --reindex first.")
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_context)


def build_query_engine(index: VectorStoreIndex):
    return index.as_query_engine(
        similarity_top_k=4,
        text_qa_template=build_qa_prompt(),
    )


def unique_sources(source_nodes) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for node in source_nodes or []:
        meta = node.metadata or {}
        source = meta.get("file_name") or meta.get("source") or meta.get("filename")
        if source and source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


def answer(question: str, query_engine) -> dict:
    response = query_engine.query(question)
    return {
        "question": question,
        "answer": str(response),
        "sources": unique_sources(response.source_nodes),
        "source_nodes": response.source_nodes,
    }


def show_context(index: VectorStoreIndex, question: str) -> None:
    retriever = index.as_retriever(similarity_top_k=4)
    nodes = retriever.retrieve(question)
    print("=== Retrieved context (LlamaIndex) ===")
    for i, node in enumerate(nodes, 1):
        meta = node.metadata or {}
        src = meta.get("file_name") or meta.get("source") or "unknown"
        text = node.get_content()
        print(f"\n--- Chunk {i} ({src}, score={node.score:.4f}) ---")
        print(text[:400] + ("..." if len(text) > 400 else ""))
    print("\n=== Answer ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex RAG query")
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument("--reindex", action="store_true", help="Rebuild index before query")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    args = parser.parse_args()

    if not args.question:
        parser.error(
            'Provide a question, e.g. query_llamaindex.py "Is the theme song cleared for EMEA?"'
        )

    question = " ".join(args.question)

    if args.reindex or not STORAGE_DIR.exists() or not any(STORAGE_DIR.iterdir()):
        index = build_index(reset=True)
    else:
        index = load_index()

    if args.show_context:
        show_context(index, question)

    query_engine = build_query_engine(index)
    result = answer(question, query_engine)

    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
