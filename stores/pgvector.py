"""pgvector store — vectors in Postgres alongside relational document metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json, execute_values

from shared.config import DATA_DIR, get_env
from shared.embeddings import embed_texts, embedding_dimensions, embedding_provider
from stores.base import ChunkHit, IngestStats, VectorStore
from stores.clearance_rows import build_clearance_rows
from stores.corpus import chunk_text, load_document_file, load_documents

DEFAULT_DATABASE_URL = "postgresql://postgres:dev@localhost:5432/applied_ai"
_EXTENSION_READY_URLS: set[str] = set()


class PgVectorStore(VectorStore):
    """Postgres + pgvector backend with a relational documents table."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_env("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.connect_timeout = int(get_env("PG_CONNECT_TIMEOUT", "5") or "5")

    def _connect_raw(self):
        return psycopg2.connect(self.database_url, connect_timeout=self.connect_timeout)

    def _ensure_extension(self) -> None:
        """Create pgvector extension once per database URL before register_vector() runs."""
        if self.database_url in _EXTENSION_READY_URLS:
            return
        with self._connect_raw() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        _EXTENSION_READY_URLS.add(self.database_url)

    def _connect(self):
        self._ensure_extension()
        conn = psycopg2.connect(self.database_url, connect_timeout=self.connect_timeout)
        register_vector(conn)
        return conn

    def reset(self) -> None:
        self._ensure_extension()
        with self._connect_raw() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
                cur.execute("DROP TABLE IF EXISTS clearance_territories CASCADE")
                cur.execute("DROP TABLE IF EXISTS documents CASCADE")
            conn.commit()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        dim = embedding_dimensions()
        self._ensure_extension()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        doc_key TEXT UNIQUE NOT NULL,
                        file_name TEXT NOT NULL,
                        sharepoint_path TEXT,
                        department TEXT,
                        confidentiality TEXT,
                        series TEXT,
                        doc_type TEXT,
                        rights_tier TEXT,
                        territory_status JSONB DEFAULT '{}'::jsonb,
                        tags JSONB DEFAULT '[]'::jsonb,
                        rel_path TEXT,
                        content_hash TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS content_hash TEXT
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id TEXT PRIMARY KEY,
                        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector({dim}),
                        source TEXT,
                        path TEXT,
                        rights_tier TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS rights_tier TEXT
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks USING hnsw (embedding vector_cosine_ops)
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS documents_doc_type_idx ON documents (doc_type)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS documents_rights_tier_idx ON documents (rights_tier)"
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clearance_territories (
                        id SERIAL PRIMARY KEY,
                        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        series TEXT NOT NULL,
                        territory TEXT NOT NULL,
                        platform TEXT,
                        status TEXT NOT NULL,
                        through_date TEXT,
                        source TEXT,
                        sharepoint_path TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS clearance_territories_series_territory_idx
                    ON clearance_territories (lower(series), territory)
                    """
                )
            conn.commit()

    def ingest(
        self,
        data_dir: Path | None = None,
        reset: bool = True,
        *,
        lora_tag: bool = False,
        lora_adapter_dir: Path | None = None,
    ) -> IngestStats:
        data_dir = data_dir or DATA_DIR
        if reset:
            self.reset()
        else:
            self._ensure_schema()

        documents = load_documents(data_dir)
        if not documents:
            raise RuntimeError(f"No .md/.txt documents found in {data_dir}")

        print(f"Using embedding provider: {embedding_provider()}")

        tagger = None
        if lora_tag:
            from shared.rights_tier_tagger import get_tagger

            tagger = get_tagger(lora_adapter_dir)
            print(f"LoRA tier tagging enabled → {tagger.adapter_dir}")

        chunk_rows: list[tuple] = []
        doc_id_by_key: dict[str, int] = {}
        lora_tagged_chunks = 0

        with self._connect() as conn:
            with conn.cursor() as cur:
                for doc in documents:
                    meta = doc["metadata"]
                    chunks = chunk_text(doc["text"])
                    chunk_labels: list[str] = []
                    if tagger and chunks:
                        chunk_labels = tagger.tag_snippets(chunks)
                        lora_tagged_chunks += len(chunk_labels)

                    doc_tier = meta.get("rights_tier")
                    if tagger:
                        doc_tier = tagger.document_tier(
                            chunk_labels,
                            warehouse_tier=meta.get("rights_tier"),
                            doc_type=meta.get("doc_type"),
                        )

                    cur.execute(
                        """
                        INSERT INTO documents (
                            doc_key, file_name, sharepoint_path, department,
                            confidentiality, series, doc_type, rights_tier,
                            territory_status, tags, rel_path
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (doc_key) DO UPDATE SET
                            file_name = EXCLUDED.file_name,
                            sharepoint_path = EXCLUDED.sharepoint_path,
                            department = EXCLUDED.department,
                            confidentiality = EXCLUDED.confidentiality,
                            series = EXCLUDED.series,
                            doc_type = EXCLUDED.doc_type,
                            rights_tier = EXCLUDED.rights_tier,
                            territory_status = EXCLUDED.territory_status,
                            tags = EXCLUDED.tags,
                            rel_path = EXCLUDED.rel_path
                        RETURNING id
                        """,
                        (
                            doc["id"],
                            meta["source"],
                            meta.get("sharepoint_path"),
                            meta.get("department"),
                            meta.get("confidentiality"),
                            meta.get("series"),
                            meta.get("doc_type"),
                            doc_tier,
                            Json(meta.get("territory_status") or {}),
                            Json(meta.get("tags") or []),
                            meta.get("path"),
                        ),
                    )
                    doc_id = cur.fetchone()[0]
                    doc_id_by_key[doc["id"]] = doc_id

                    if reset:
                        cur.execute(
                            "DELETE FROM document_chunks WHERE document_id = %s",
                            (doc_id,),
                        )

                    for i, chunk in enumerate(chunks):
                        chunk_id = hashlib.md5(f"{doc['id']}-{i}".encode()).hexdigest()
                        chunk_tier = chunk_labels[i] if chunk_labels else None
                        chunk_rows.append(
                            (
                                chunk_id,
                                doc_id,
                                i,
                                chunk,
                                meta.get("source"),
                                meta.get("path"),
                                chunk_tier,
                            )
                        )

                conn.commit()

                if reset:
                    cur.execute("DELETE FROM clearance_territories")
                    territory_rows = build_clearance_rows(documents)
                    ct_values = []
                    for row in territory_rows:
                        source = row.get("source") or ""
                        doc_id = None
                        for doc in documents:
                            if doc["metadata"].get("source") == source:
                                doc_id = doc_id_by_key.get(doc["id"])
                                break
                        if doc_id is None:
                            continue
                        ct_values.append(
                            (
                                doc_id,
                                row.get("series") or "Harbor Line",
                                row.get("territory"),
                                row.get("platform"),
                                row.get("status"),
                                row.get("through_date"),
                                source,
                                row.get("sharepoint_path"),
                            )
                        )
                    if ct_values:
                        execute_values(
                            cur,
                            """
                            INSERT INTO clearance_territories (
                                document_id, series, territory, platform,
                                status, through_date, source, sharepoint_path
                            ) VALUES %s
                            """,
                            ct_values,
                        )
                    conn.commit()

        if chunk_rows:
            self._embed_chunk_rows(chunk_rows)

        return IngestStats(
            documents=len(documents),
            chunks=len(chunk_rows),
            lora_tagged_chunks=lora_tagged_chunks,
        )

    def _embed_chunk_rows(self, chunk_rows: list[tuple]) -> None:
        batch_size = 16
        with self._connect() as conn:
            with conn.cursor() as cur:
                for start in range(0, len(chunk_rows), batch_size):
                    batch = chunk_rows[start : start + batch_size]
                    vectors = embed_texts([row[3] for row in batch])
                    values = [
                        (
                            row[0],
                            row[1],
                            row[2],
                            row[3],
                            vectors[i],
                            row[4],
                            row[5],
                            row[6],
                        )
                        for i, row in enumerate(batch)
                    ]
                    execute_values(
                        cur,
                        """
                        INSERT INTO document_chunks (
                            id, document_id, chunk_index, content, embedding,
                            source, path, rights_tier
                        ) VALUES %s
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            rights_tier = EXCLUDED.rights_tier
                        """,
                        values,
                        template="(%s, %s, %s, %s, %s::vector, %s, %s, %s)",
                    )
                conn.commit()

    def ingest_document(
        self,
        doc: dict,
        *,
        content_hash: str | None = None,
        lora_tag: bool = False,
        lora_adapter_dir: Path | None = None,
        skip_unchanged: bool = True,
    ) -> IngestStats:
        """Incremental upsert for one document (event-driven embed worker)."""
        self._ensure_schema()
        meta = doc["metadata"]
        digest = content_hash or hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, content_hash FROM documents WHERE doc_key = %s",
                    (doc["id"],),
                )
                existing = cur.fetchone()
                if skip_unchanged and existing and existing[1] == digest:
                    return IngestStats(documents=0, chunks=0, lora_tagged_chunks=0)

        tagger = None
        if lora_tag:
            from shared.rights_tier_tagger import get_tagger

            tagger = get_tagger(lora_adapter_dir)

        chunks = chunk_text(doc["text"])
        chunk_labels: list[str] = []
        if tagger and chunks:
            chunk_labels = tagger.tag_snippets(chunks)

        doc_tier = meta.get("rights_tier")
        if tagger:
            doc_tier = tagger.document_tier(
                chunk_labels,
                warehouse_tier=meta.get("rights_tier"),
                doc_type=meta.get("doc_type"),
            )

        chunk_rows: list[tuple] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (
                        doc_key, file_name, sharepoint_path, department,
                        confidentiality, series, doc_type, rights_tier,
                        territory_status, tags, rel_path, content_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_key) DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        sharepoint_path = EXCLUDED.sharepoint_path,
                        department = EXCLUDED.department,
                        confidentiality = EXCLUDED.confidentiality,
                        series = EXCLUDED.series,
                        doc_type = EXCLUDED.doc_type,
                        rights_tier = EXCLUDED.rights_tier,
                        territory_status = EXCLUDED.territory_status,
                        tags = EXCLUDED.tags,
                        rel_path = EXCLUDED.rel_path,
                        content_hash = EXCLUDED.content_hash
                    RETURNING id
                    """,
                    (
                        doc["id"],
                        meta["source"],
                        meta.get("sharepoint_path"),
                        meta.get("department"),
                        meta.get("confidentiality"),
                        meta.get("series"),
                        meta.get("doc_type"),
                        doc_tier,
                        Json(meta.get("territory_status") or {}),
                        Json(meta.get("tags") or []),
                        meta.get("path"),
                        digest,
                    ),
                )
                doc_id = cur.fetchone()[0]

                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
                cur.execute(
                    "DELETE FROM clearance_territories WHERE document_id = %s",
                    (doc_id,),
                )

                for i, chunk in enumerate(chunks):
                    chunk_id = hashlib.md5(f"{doc['id']}-{i}".encode()).hexdigest()
                    chunk_rows.append(
                        (
                            chunk_id,
                            doc_id,
                            i,
                            chunk,
                            meta.get("source"),
                            meta.get("path"),
                            chunk_labels[i] if chunk_labels else None,
                        )
                    )

                territory_rows = build_clearance_rows([doc])
                ct_values = [
                    (
                        doc_id,
                        row.get("series") or "Harbor Line",
                        row.get("territory"),
                        row.get("platform"),
                        row.get("status"),
                        row.get("through_date"),
                        row.get("source") or meta.get("source"),
                        row.get("sharepoint_path"),
                    )
                    for row in territory_rows
                ]
                if ct_values:
                    execute_values(
                        cur,
                        """
                        INSERT INTO clearance_territories (
                            document_id, series, territory, platform,
                            status, through_date, source, sharepoint_path
                        ) VALUES %s
                        """,
                        ct_values,
                    )
                conn.commit()

        if chunk_rows:
            self._embed_chunk_rows(chunk_rows)

        return IngestStats(
            documents=1,
            chunks=len(chunk_rows),
            lora_tagged_chunks=len(chunk_labels),
        )

    def delete_document(self, doc_key: str) -> bool:
        """Remove document and cascading chunks / clearance rows."""
        self._ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE doc_key = %s RETURNING id", (doc_key,))
                row = cur.fetchone()
                conn.commit()
        return row is not None

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
        filters: list[str] = []
        filter_params: list[Any] = []

        if doc_type:
            filters.append("d.doc_type = %s")
            filter_params.append(doc_type)
        if confidentiality:
            filters.append("d.confidentiality = %s")
            filter_params.append(confidentiality)
        if rights_tier:
            filters.append("COALESCE(c.rights_tier, d.rights_tier) = %s")
            filter_params.append(rights_tier)
        if department:
            filters.append("d.department = %s")
            filter_params.append(department)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        params = [query_vec, *filter_params, query_vec, top_k]

        sql = f"""
            SELECT
                c.content,
                c.source,
                c.path,
                c.chunk_index,
                COALESCE(c.rights_tier, d.rights_tier) AS rights_tier,
                d.doc_type,
                d.department,
                d.confidentiality,
                d.territory_status,
                d.sharepoint_path,
                d.series,
                d.tags,
                c.embedding <=> %s::vector AS distance
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        hits: list[ChunkHit] = []
        for row in rows:
            (
                content,
                source,
                path,
                chunk_index,
                rt,
                dt,
                dept,
                conf,
                territory_status,
                sp_path,
                series,
                tags,
                distance,
            ) = row
            hits.append(
                ChunkHit(
                    text=content,
                    distance=float(distance),
                    metadata={
                        "source": source,
                        "path": path,
                        "chunk_index": chunk_index,
                        "doc_type": dt,
                        "department": dept,
                        "confidentiality": conf,
                        "rights_tier": rt,
                        "territory_status": (
                            territory_status
                            if isinstance(territory_status, dict)
                            else json.loads(territory_status or "{}")
                        ),
                        "sharepoint_path": sp_path,
                        "series": series,
                        "tags": tags if isinstance(tags, list) else json.loads(tags or "[]"),
                    },
                )
            )
        return hits

    def lookup_clearance_territories(
        self,
        series: str,
        territories: list[str],
    ) -> list[dict]:
        """SQL join: territory clearance rows for a series (Option C warehouse)."""
        if not territories:
            return []
        series_key = series.lower()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        ct.territory,
                        ct.platform,
                        ct.status,
                        ct.through_date,
                        ct.source,
                        ct.sharepoint_path,
                        d.rights_tier,
                        d.series
                    FROM clearance_territories ct
                    JOIN documents d ON d.id = ct.document_id
                    WHERE lower(ct.series) = %s
                      AND ct.territory = ANY(%s)
                    ORDER BY ct.territory, ct.source
                    """,
                    (series_key, territories),
                )
                rows = cur.fetchall()
        results: list[dict] = []
        for row in rows:
            territory, platform, status, through_date, source, sp_path, tier, row_series = row
            results.append(
                {
                    "territory": territory,
                    "platform": platform,
                    "status": status,
                    "through_date": through_date,
                    "source": source,
                    "sharepoint_path": sp_path,
                    "rights_tier": tier,
                    "series": row_series,
                }
            )
        return results

    def sql_join_demo(self, query: str, top_k: int = 3) -> list[dict]:
        """Warehouse-style query: vector hits joined to documents metadata."""
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
