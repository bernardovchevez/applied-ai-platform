#!/usr/bin/env python3
"""LangChain RAG over synthetic media ops docs (Ollama + Chroma)."""

from __future__ import annotations
 
import argparse
import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.config import DATA_DIR, get_env

CHROMA_DIR = Path(__file__).parent / ".chroma_langchain"
COLLECTION_NAME = "media_docs_langchain"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "guardrails" / "prompts"


def load_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={
                    "source": path.name,
                    "path": str(path.relative_to(data_dir)),
                    "doc_type": path.parent.name if path.parent != data_dir else "general",
                },
            )
        )
    return docs


def build_vectorstore(data_dir: Path | None = None, reset: bool = True) -> Chroma:
    data_dir = data_dir or DATA_DIR
    embed_model = get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embed_model)

    if reset and CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR)

    raw_docs = load_documents(data_dir)
    if not raw_docs:
        raise RuntimeError(f"No .md/.txt documents found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"LangChain index: {len(chunks)} chunks from {len(raw_docs)} docs → {CHROMA_DIR}")
    return vectorstore


def get_vectorstore() -> Chroma:
    embed_model = get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embed_model)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def build_chain(vectorstore: Chroma):
    chat_model = get_env("OLLAMA_CHAT_MODEL", "llama3.2")
    llm = ChatOllama(model=chat_model, temperature=0)

    system_path = PROMPTS_DIR / "system_rag.txt"
    system = system_path.read_text() if system_path.exists() else "Answer from context only."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system + "\n\nContext:\n{context}"),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 4}), combine_docs_chain)


def unique_sources(documents: list) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source") if hasattr(doc, "metadata") else None
        if source and source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


def answer(question: str, chain) -> dict:
    result = chain.invoke({"input": question})
    docs = result.get("context", [])
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "sources": unique_sources(docs),
        "chunks": docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain RAG query")
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument("--reindex", action="store_true", help="Rebuild Chroma index before query")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    args = parser.parse_args()

    if not args.question:
        parser.error('Provide a question, e.g. query_langchain.py "Is the theme song cleared for EMEA?"')

    if args.reindex or not CHROMA_DIR.exists():
        build_vectorstore(reset=True)
    elif not any(CHROMA_DIR.iterdir()):
        build_vectorstore(reset=True)

    vectorstore = get_vectorstore()
    chain = build_chain(vectorstore)
    question = " ".join(args.question)

    if args.show_context:
        docs = vectorstore.similarity_search(question, k=4)
        print("=== Retrieved context (LangChain) ===")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            print(f"\n--- Chunk {i} ({src}) ---")
            print(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
        print("\n=== Answer ===")

    result = answer(question, chain)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
