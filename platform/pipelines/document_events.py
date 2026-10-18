"""Local document_ingested event stub (Kafka simulation)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_TYPE = "document_ingested"
EVENT_VERSION = "1.0"


def doc_key_from_rel_path(rel_path: str) -> str:
    """Stable key aligned with pgvector documents.doc_key (path stem)."""
    return Path(rel_path).stem


def build_event(
    row: dict[str, Any],
    *,
    raw_dir: Path,
    operation: str = "upsert",
    correlation_id: str | None = None,
) -> dict[str, Any]:
    rel_path = row["rel_path"]
    meta = {
        "tags": json.loads(row.get("tags") or "[]"),
        "territory_status": json.loads(row.get("territory_status") or "{}"),
        "clearance_territories": row.get("clearance_territories") or [],
    }
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": EVENT_TYPE,
        "event_version": EVENT_VERSION,
        "occurred_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "operation": operation,
        "document": {
            "doc_key": doc_key_from_rel_path(rel_path),
            "file_name": row["file_name"],
            "rel_path": rel_path,
            "content_hash": row["content_hash"],
            "content_uri": str((raw_dir / rel_path).resolve()),
            "byte_size": row.get("byte_size"),
            "series": row.get("series"),
            "doc_type": row.get("doc_type"),
            "department": row.get("department"),
            "confidentiality": row.get("confidentiality"),
            "sharepoint_path": row.get("sharepoint_path"),
            "rights_tier": row.get("rights_tier"),
            "source_system": row.get("source_system", "lake_raw"),
        },
        "metadata": meta,
        "trace": {
            "correlation_id": correlation_id or f"batch-{datetime.now(timezone.utc):%Y%m%d}",
            "producer": "ingest_batch.py",
        },
    }


def _write_event_file(event: dict, events_dir: Path) -> Path:
    events_dir.mkdir(parents=True, exist_ok=True)
    doc_key = event["document"]["doc_key"]
    path = events_dir / f"{doc_key}.json"
    path.write_text(json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def emit_events(
    rows: list[dict],
    events_dir: Path,
    *,
    raw_dir: Path,
    correlation_id: str | None = None,
    operation: str = "upsert",
) -> list[Path]:
    paths: list[Path] = []
    for row in rows:
        event = build_event(
            row,
            raw_dir=raw_dir,
            operation=operation,
            correlation_id=correlation_id,
        )
        paths.append(_write_event_file(event, events_dir))
    return paths


def emit_delta_events(
    diff: dict,
    rows: list[dict],
    previous_manifest: dict | None,
    events_dir: Path,
    *,
    raw_dir: Path,
    correlation_id: str | None = None,
) -> list[Path]:
    """Emit upsert events for added/changed docs and delete events for removed."""
    rows_by_rel = {r["rel_path"]: r for r in rows}
    old_by_rel: dict[str, dict] = {}
    if previous_manifest:
        for doc in previous_manifest.get("documents") or []:
            rel = doc.get("rel_path")
            if rel:
                old_by_rel[rel] = doc

    paths: list[Path] = []
    for rel in diff.get("added", []) + diff.get("changed_hash", []):
        row = rows_by_rel.get(rel)
        if not row:
            continue
        event = build_event(row, raw_dir=raw_dir, operation="upsert", correlation_id=correlation_id)
        paths.append(_write_event_file(event, events_dir))

    for rel in diff.get("removed", []):
        old = old_by_rel.get(rel, {})
        row = {
            "rel_path": rel,
            "file_name": old.get("file_name") or Path(rel).name,
            "content_hash": old.get("content_hash") or "",
            "byte_size": 0,
            "tags": "[]",
            "territory_status": "{}",
            "clearance_territories": [],
        }
        event = build_event(row, raw_dir=raw_dir, operation="delete", correlation_id=correlation_id)
        paths.append(_write_event_file(event, events_dir))

    return paths


def load_event(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def processed_marker_path(processed_dir: Path, doc_key: str) -> Path:
    return processed_dir / f"{doc_key}.json"


def is_event_stale(event: dict, processed_dir: Path) -> bool:
    """Skip if same content_hash already processed for this doc_key."""
    doc_key = event["document"]["doc_key"]
    marker = processed_marker_path(processed_dir, doc_key)
    if not marker.exists():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return state.get("content_hash") == event["document"].get("content_hash")


def mark_processed(event: dict, processed_dir: Path, *, status: str = "ok") -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    doc = event["document"]
    payload = {
        "doc_key": doc["doc_key"],
        "event_id": event.get("event_id"),
        "content_hash": doc.get("content_hash"),
        "processed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
    }
    path = processed_marker_path(processed_dir, doc["doc_key"])
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def list_pending_events(events_dir: Path, processed_dir: Path) -> list[Path]:
    if not events_dir.exists():
        return []
    pending: list[Path] = []
    for path in sorted(events_dir.glob("*.json")):
        try:
            event = load_event(path)
        except json.JSONDecodeError:
            continue
        if not is_event_stale(event, processed_dir):
            pending.append(path)
    return pending
