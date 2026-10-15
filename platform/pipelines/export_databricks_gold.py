#!/usr/bin/env python3
"""Export lake corpus -> gold_document_embeddings parquet for Databricks Vector Search.

Vectors use the same embedding model as pgvector for parity across backends.

Usage:
  python platform/pipelines/export_databricks_gold.py
  python platform/pipelines/export_databricks_gold.py --write-query-vectors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINES_DIR = Path(__file__).resolve().parent
LAB_ROOT = PIPELINES_DIR.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from shared.embeddings import embed_texts, embedding_dimensions, embedding_provider  # noqa: E402
from stores.corpus import iter_corpus_chunks  # noqa: E402

DEFAULT_RAW = LAB_ROOT / "data" / "lake" / "raw"
DEFAULT_OUT = LAB_ROOT / "data" / "lake" / "databricks"
DEFAULT_PARQUET = DEFAULT_OUT / "gold_document_embeddings.parquet"
DEFAULT_QUERIES = DEFAULT_OUT / "query_vectors.json"

DEMO_QUESTIONS = [
    "Is the theme song cleared for EMEA streaming?",
    "EMEA music clearance",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_gold_rows(data_dir: Path) -> list[dict]:
    pending: list[tuple[str, str, dict]] = list(iter_corpus_chunks(data_dir))
    if not pending:
        raise SystemExit(f"No documents under {data_dir}")

    texts = [text for _, text, _ in pending]
    print(f"Embedding {len(texts)} chunks via {embedding_provider()} ({embedding_dimensions()}d)...")
    vectors = embed_texts(texts)

    rows: list[dict] = []
    for (chunk_id, text, meta), embedding in zip(pending, vectors):
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows.append(
            {
                "chunk_id": chunk_id,
                "doc_key": Path(meta.get("path", meta.get("source", ""))).stem or chunk_id[:16],
                "chunk_index": int(meta.get("chunk_index", 0)),
                "chunk_text": text,
                "embedding": embedding,
                "file_name": meta.get("source"),
                "series": meta.get("series"),
                "doc_type": meta.get("doc_type"),
                "department": meta.get("department"),
                "confidentiality": meta.get("confidentiality"),
                "rights_tier": meta.get("rights_tier"),
                "content_hash": content_hash,
                "embed_model_version": f"{embedding_provider()}-{embedding_dimensions()}",
                "updated_at": _now_iso(),
            }
        )
    return rows


def write_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow required.\n  pip install -r requirements-pipelines.txt"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="snappy")
    print(f"Wrote {len(rows)} rows → {path}")


def write_query_vectors(questions: list[str], path: Path) -> None:
    print(f"Embedding {len(questions)} demo queries...")
    vectors = embed_texts(questions)
    payload = {
        "embed_model_version": f"{embedding_provider()}-{embedding_dimensions()}",
        "generated_at": _now_iso(),
        "queries": {
            q: vec for q, vec in zip(questions, vectors)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote query vectors → {path}")


def run_export(
    *,
    data_dir: Path,
    parquet_path: Path,
    query_vectors_path: Path,
    write_query_vectors_flag: bool,
) -> dict:
    rows = build_gold_rows(data_dir)
    write_parquet(rows, parquet_path)

    doc_types: dict[str, int] = {}
    for row in rows:
        key = row.get("doc_type") or "unknown"
        doc_types[key] = doc_types.get(key, 0) + 1

    summary = {
        "chunk_count": len(rows),
        "embedding_dim": embedding_dimensions(),
        "embed_model": embedding_provider(),
        "doc_types": doc_types,
        "parquet_path": str(parquet_path),
    }

    if write_query_vectors_flag:
        write_query_vectors(DEMO_QUESTIONS, query_vectors_path)
        summary["query_vectors_path"] = str(query_vectors_path)

    print(f"Doc types: {doc_types}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gold embeddings parquet for Databricks")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--write-query-vectors",
        action="store_true",
        help="Also write query_vectors.json (pre-embed demo questions for DB notebook)",
    )
    args = parser.parse_args()

    run_export(
        data_dir=args.data_dir,
        parquet_path=args.out_dir / "gold_document_embeddings.parquet",
        query_vectors_path=args.out_dir / "query_vectors.json",
        write_query_vectors_flag=args.write_query_vectors,
    )


if __name__ == "__main__":
    main()
