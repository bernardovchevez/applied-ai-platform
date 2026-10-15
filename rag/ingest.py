#!/usr/bin/env python3
"""Ingest synthetic media ops documents into ChromaDB."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from shared.config import DATA_DIR
from shared.embeddings import embedding_provider, get_embedding_function

CHROMA_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "media_docs"
CHUNK_SIZE = 512  # approximate characters per chunk
CHUNK_OVERLAP = 50


def load_documents(data_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        docs.append(
            {
                "id": path.stem,
                "text": text,
                "metadata": {
                    "source": path.name,
                    "path": str(path.relative_to(data_dir)),
                    "doc_type": path.parent.name if path.parent != data_dir else "general",
                },
            }
        )
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def ingest(data_dir: Path | None = None, reset: bool = True) -> int:
    data_dir = data_dir or DATA_DIR
    embedding_fn = get_embedding_function()
    provider = embedding_provider()
    print(f"Using embedding provider: {provider}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "Synthetic media operations corpus"},
    )

    documents = load_documents(data_dir)
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_id = hashlib.md5(f"{doc['id']}-{i}".encode()).hexdigest()
            ids.append(chunk_id)
            texts.append(chunk)
            meta = {**doc["metadata"], "chunk_index": i}
            metadatas.append(meta)

    if not texts:
        raise RuntimeError(f"No .md/.txt documents found in {data_dir}")

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(texts)} chunks from {len(documents)} documents → {CHROMA_DIR}")
    return len(texts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()
    ingest(data_dir=args.data_dir, reset=not args.no_reset)
