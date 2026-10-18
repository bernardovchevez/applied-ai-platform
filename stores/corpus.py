"""Shared document loading and chunking for all vector stores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from shared.config import DATA_DIR, LAKE_META, SEED_META

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_sharepoint_meta(meta_path: Path | None = None) -> dict[str, dict]:
    """Index sharepoint-meta.json by file_name (lake catalog first, synthetic seed fallback)."""
    if meta_path is not None:
        path = meta_path
    elif LAKE_META.exists():
        path = LAKE_META
    elif SEED_META.exists():
        path = SEED_META
    else:
        path = LAKE_META
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["file_name"]: row for row in data.get("documents", [])}


def derive_rights_tier(meta: dict) -> str | None:
    """Simulate warehouse rights_tier from SharePoint metadata."""
    doc_type = meta.get("doc_type")
    if doc_type != "rights_clearance":
        return None
    tags = [t.lower() for t in meta.get("tags") or []]
    territory_status = meta.get("territory_status") or {}
    if isinstance(territory_status, dict):
        values = {str(v).lower() for v in territory_status.values()}
        if "excluded" in values:
            return "excluded"
    if "excluded" in tags:
        return "excluded"
    if "pending" in tags:
        return "pending"
    return "cleared"


def load_documents(data_dir: Path | None = None) -> list[dict]:
    """Load .md/.txt files and merge SharePoint warehouse metadata."""
    data_dir = data_dir or DATA_DIR
    meta_by_file = load_sharepoint_meta()

    docs: list[dict] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.name == "sharepoint-meta.json":
            continue

        rel_path = str(path.relative_to(data_dir))
        sp = meta_by_file.get(path.name, {})
        doc_type = sp.get("doc_type") or (
            path.parent.name if path.parent != data_dir else "general"
        )

        docs.append(
            {
                "id": path.stem,
                "text": path.read_text(encoding="utf-8"),
                "metadata": {
                    "source": path.name,
                    "path": rel_path,
                    "doc_type": doc_type,
                    "department": sp.get("department"),
                    "confidentiality": sp.get("confidentiality"),
                    "series": sp.get("series"),
                    "sharepoint_path": sp.get("sharepoint_path"),
                    "rights_tier": derive_rights_tier(sp),
                    "tags": sp.get("tags") or [],
                    "territory_status": sp.get("territory_status") or {},
                    "clearance_territories": sp.get("clearance_territories") or [],
                },
            }
        )
    return docs


def load_document_file(path: Path, data_dir: Path) -> dict:
    """Load a single corpus file with SharePoint metadata (event worker)."""
    meta_path = data_dir / "sharepoint-meta.json"
    meta_by_file = load_sharepoint_meta(meta_path if meta_path.exists() else None)
    rel_path = str(path.relative_to(data_dir))
    sp = meta_by_file.get(path.name, {})
    doc_type = sp.get("doc_type") or (
        path.parent.name if path.parent != data_dir else "general"
    )
    return {
        "id": path.stem,
        "text": path.read_text(encoding="utf-8"),
        "metadata": {
            "source": path.name,
            "path": rel_path,
            "doc_type": doc_type,
            "department": sp.get("department"),
            "confidentiality": sp.get("confidentiality"),
            "series": sp.get("series"),
            "sharepoint_path": sp.get("sharepoint_path"),
            "rights_tier": derive_rights_tier(sp),
            "tags": sp.get("tags") or [],
            "territory_status": sp.get("territory_status") or {},
            "clearance_territories": sp.get("clearance_territories") or [],
        },
    }


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def iter_corpus_chunks(
    data_dir: Path | None = None,
) -> Iterator[tuple[str, str, dict]]:
    """Yield (chunk_id, text, metadata) aligned with Chroma / pgvector ingest IDs."""
    for doc in load_documents(data_dir):
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_id = hashlib.md5(f"{doc['id']}-{i}".encode()).hexdigest()
            meta = {**doc["metadata"], "chunk_index": i}
            yield chunk_id, chunk, meta
