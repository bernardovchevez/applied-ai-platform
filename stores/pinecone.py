"""Pinecone store — managed serverless vectors with metadata filters."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from pinecone import NotFoundError, Pinecone, ServerlessSpec

from shared.config import DATA_DIR, get_env, require_env
from shared.embeddings import embed_texts, embedding_dimensions, embedding_provider
from stores.base import ChunkHit, IngestStats, VectorStore
from stores.corpus import chunk_text, load_documents

DEFAULT_INDEX_NAME = "media-docs-index"
DEFAULT_NAMESPACE = "media_docs"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
UPSERT_BATCH = 100


def _pinecone_metadata(meta: dict[str, Any], text: str) -> dict[str, Any]:
    """Pinecone metadata: no nulls; tags as list[str]; chunk text for retrieval."""
    cleaned: dict[str, Any] = {"text": text}
    for key, value in meta.items():
        if value is None:
            continue
        if key == "tags":
            cleaned[key] = [str(tag) for tag in value]
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _build_filter(
    *,
    doc_type: str | None = None,
    confidentiality: str | None = None,
    rights_tier: str | None = None,
    department: str | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if doc_type:
        clauses.append({"doc_type": {"$eq": doc_type}})
    if confidentiality:
        clauses.append({"confidentiality": {"$eq": confidentiality}})
    if rights_tier:
        clauses.append({"rights_tier": {"$eq": rights_tier}})
    if department:
        clauses.append({"department": {"$eq": department}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class PineconeStore(VectorStore):
    """Pinecone serverless backend using the shared VectorStore interface."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        cloud: str | None = None,
        region: str | None = None,
    ) -> None:
        self.api_key = api_key or require_env("PINECONE_API_KEY")
        self.index_name = index_name or get_env("PINECONE_INDEX", DEFAULT_INDEX_NAME)
        self.namespace = namespace
        self.cloud = cloud or get_env("PINECONE_CLOUD", DEFAULT_CLOUD)
        self.region = region or get_env("PINECONE_REGION", DEFAULT_REGION)
        self._client: Pinecone | None = None

    def _pc(self) -> Pinecone:
        if self._client is None:
            self._client = Pinecone(api_key=self.api_key)
        return self._client

    def _index(self):
        self._ensure_index()
        return self._pc().Index(self.index_name)

    def _ensure_index(self) -> None:
        pc = self._pc()
        dim = embedding_dimensions()
        names = {idx.name for idx in pc.list_indexes()}

        if self.index_name not in names:
            print(
                f"Creating Pinecone index {self.index_name!r} "
                f"(dim={dim}, metric=cosine, {self.cloud}/{self.region})..."
            )
            pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
            self._wait_until_ready()
            return

        desc = pc.describe_index(self.index_name)
        if desc.dimension != dim:
            raise RuntimeError(
                f"Pinecone index {self.index_name!r} has dimension {desc.dimension}, "
                f"but embedding provider expects {dim}. "
                "Delete the index in Pinecone console or use a different PINECONE_INDEX."
            )

    def _wait_until_ready(self, timeout_sec: int = 120) -> None:
        pc = self._pc()
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            desc = pc.describe_index(self.index_name)
            if desc.status.ready:
                return
            time.sleep(2)
        raise RuntimeError(f"Pinecone index {self.index_name!r} not ready after {timeout_sec}s")

    def reset(self) -> None:
        index = self._index()
        try:
            index.delete(delete_all=True, namespace=self.namespace)
        except NotFoundError:
            # Fresh index — namespace does not exist until first upsert
            pass

    def ingest(self, data_dir: Path | None = None, reset: bool = True) -> IngestStats:
        data_dir = data_dir or DATA_DIR
        if reset:
            self.reset()

        documents = load_documents(data_dir)
        if not documents:
            raise RuntimeError(f"No .md/.txt documents found in {data_dir}")

        print(f"Using embedding provider: {embedding_provider()}")

        records: list[tuple[str, dict[str, Any], str]] = []
        for doc in documents:
            meta = {**doc["metadata"], "doc_key": doc["id"]}
            for i, chunk in enumerate(chunk_text(doc["text"])):
                chunk_id = hashlib.md5(f"{doc['id']}-{i}".encode()).hexdigest()
                chunk_meta = {**meta, "chunk_index": i}
                records.append((chunk_id, chunk_meta, chunk))

        index = self._index()
        for start in range(0, len(records), UPSERT_BATCH):
            batch = records[start : start + UPSERT_BATCH]
            vectors = embed_texts([text for _, _, text in batch])
            upserts = [
                (
                    chunk_id,
                    vectors[i],
                    _pinecone_metadata(meta, text),
                )
                for i, (chunk_id, meta, text) in enumerate(batch)
            ]
            index.upsert(vectors=upserts, namespace=self.namespace)

        return IngestStats(documents=len(documents), chunks=len(records))

    def search(
        self,
        query: str,
        top_k: int = 4,
        *,
        doc_type: str | None = None,
        confidentiality: str | None = None,
        rights_tier: str | None = None,
        department: str | None = None,
    ) -> list[ChunkHit]:
        query_vec = embed_texts([query])[0]
        metadata_filter = _build_filter(
            doc_type=doc_type,
            confidentiality=confidentiality,
            rights_tier=rights_tier,
            department=department,
        )

        kwargs: dict[str, Any] = {
            "vector": query_vec,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        response = self._index().query(**kwargs)

        hits: list[ChunkHit] = []
        for match in response.matches:
            meta = dict(match.metadata or {})
            text = meta.pop("text", "")
            score = float(match.score or 0.0)
            hits.append(
                ChunkHit(
                    text=text,
                    distance=1.0 - score,  # align with pgvector cosine distance
                    metadata=meta,
                )
            )
        return hits

    def sql_join_demo(self, query: str, top_k: int = 3) -> list[dict]:
        """Warehouse-style demo: cleared rights_clearance memos only."""
        hits = self.search(
            query,
            top_k=top_k,
            doc_type="rights_clearance",
            rights_tier="cleared",
        )
        return [
            {
                "file": h.metadata.get("source"),
                "doc_type": h.metadata.get("doc_type"),
                "rights_tier": h.metadata.get("rights_tier"),
                "sharepoint_path": h.metadata.get("sharepoint_path"),
                "distance": round(h.distance, 4),
                "snippet": h.text[:120] + ("..." if len(h.text) > 120 else ""),
            }
            for h in hits
        ]
