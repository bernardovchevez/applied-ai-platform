#!/usr/bin/env python3
"""Embed worker: consume document_ingested events → pgvector upsert.

Usage:
  python platform/pipelines/embed_worker.py
  python platform/pipelines/embed_worker.py --event data/lake/events/document_ingested/music-clearance-emea.json
  python platform/pipelines/embed_worker.py --lora-tag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINES_DIR = Path(__file__).resolve().parent
LAB_ROOT = PIPELINES_DIR.parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(PIPELINES_DIR))

from document_events import (  # noqa: E402
    is_event_stale,
    list_pending_events,
    load_event,
    mark_processed,
)
from stores.corpus import load_document_file  # noqa: E402
from stores.pgvector import PgVectorStore  # noqa: E402

DEFAULT_RAW = LAB_ROOT / "data" / "lake" / "raw"
DEFAULT_EVENTS = LAB_ROOT / "data" / "lake" / "events" / "document_ingested"
DEFAULT_PROCESSED = LAB_ROOT / "data" / "lake" / "events" / "processed"


def process_event(
    event_path: Path,
    *,
    raw_dir: Path,
    processed_dir: Path,
    lora_tag: bool,
    lora_adapter: Path | None,
    force: bool,
) -> dict:
    event = load_event(event_path)
    doc_meta = event["document"]
    doc_key = doc_meta["doc_key"]

    if not force and is_event_stale(event, processed_dir):
        return {"doc_key": doc_key, "status": "skipped", "reason": "unchanged content_hash"}

    store = PgVectorStore()
    operation = event.get("operation", "upsert")

    if operation == "delete":
        deleted = store.delete_document(doc_key)
        mark_processed(event, processed_dir, status="deleted" if deleted else "missing")
        return {"doc_key": doc_key, "status": "deleted" if deleted else "missing"}

    if operation == "metadata_only":
        mark_processed(event, processed_dir, status="metadata_only_stub")
        return {"doc_key": doc_key, "status": "metadata_only_stub"}

    rel_path = doc_meta.get("rel_path") or doc_meta.get("file_name")
    file_path = raw_dir / rel_path
    if not file_path.exists():
        mark_processed(event, processed_dir, status="error_missing_file")
        return {"doc_key": doc_key, "status": "error", "reason": f"file not found: {file_path}"}

    doc = load_document_file(file_path, raw_dir)
    stats = store.ingest_document(
        doc,
        content_hash=doc_meta.get("content_hash"),
        lora_tag=lora_tag,
        lora_adapter_dir=lora_adapter,
        skip_unchanged=not force,
    )

    if stats.documents == 0 and stats.chunks == 0:
        mark_processed(event, processed_dir, status="skipped_unchanged")
        return {"doc_key": doc_key, "status": "skipped", "reason": "unchanged in store"}

    mark_processed(event, processed_dir, status="ok")
    return {
        "doc_key": doc_key,
        "status": "ok",
        "chunks": stats.chunks,
        "lora_tagged_chunks": stats.lora_tagged_chunks,
    }


def run_worker(
    *,
    raw_dir: Path,
    events_dir: Path,
    processed_dir: Path,
    event_path: Path | None,
    lora_tag: bool,
    lora_adapter: Path | None,
    force: bool,
) -> list[dict]:
    paths = [event_path] if event_path else list_pending_events(events_dir, processed_dir)
    if not paths:
        print(f"No pending events in {events_dir}")
        return []

    results: list[dict] = []
    for path in paths:
        result = process_event(
            path,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            lora_tag=lora_tag,
            lora_adapter=lora_adapter,
            force=force,
        )
        results.append(result)
        print(f"{result['doc_key']}: {result['status']}", end="")
        if result.get("chunks"):
            print(f" ({result['chunks']} chunks)", end="")
        if result.get("reason"):
            print(f" — {result['reason']}", end="")
        print()

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"Processed {len(results)} events — ok={ok} skipped={skipped}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume document_ingested events → pgvector")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--event", type=Path, default=None, help="Process one event file")
    parser.add_argument("--lora-tag", action="store_true")
    parser.add_argument("--lora-adapter", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Re-embed even if content_hash unchanged")
    args = parser.parse_args()

    run_worker(
        raw_dir=args.raw_dir,
        events_dir=args.events_dir,
        processed_dir=args.processed_dir,
        event_path=args.event,
        lora_tag=args.lora_tag,
        lora_adapter=args.lora_adapter,
        force=args.force,
    )


if __name__ == "__main__":
    main()
