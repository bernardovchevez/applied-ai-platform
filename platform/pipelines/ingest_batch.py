#!/usr/bin/env python3
"""Batch ingest: lake raw → parquet manifest (+ optional synthetic seed copy).

Usage:
  python platform/pipelines/ingest_batch.py
  python platform/pipelines/ingest_batch.py --emit-events
  python platform/pipelines/ingest_batch.py --seed-from-synthetic --emit-events
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_events import emit_delta_events, emit_events
from shared.config import DOCS_SEED_DIR
from stores.corpus import derive_rights_tier, load_sharepoint_meta
LAKE_RAW = LAB_ROOT / "data" / "lake" / "raw"
LAKE_PARQUET = LAB_ROOT / "data" / "lake" / "parquet"
LAKE_EVENTS = LAB_ROOT / "data" / "lake" / "events" / "document_ingested"
MANIFEST_PATH = LAKE_PARQUET / "ingest_manifest.json"
DOCUMENTS_PARQUET = LAKE_PARQUET / "documents.parquet"

TEXT_SUFFIXES = {".md", ".txt"}


def sync_seed_to_lake(raw_dir: Path, seed_dir: Path | None = None) -> int:
    """Copy applied-ai-docs seed pack into local lake raw folder (dev fallback only)."""
    seed_root = seed_dir or DOCS_SEED_DIR
    if not seed_root.is_dir():
        raise SystemExit(
            f"Seed corpus not found: {seed_root}\n"
            "Use applied-ai-docs (Import seed) or clone seed/harbor-line into applied-ai-docs."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in seed_root.rglob("*"):
        if src.is_file() and src.suffix.lower() in {".md", ".txt", ".json"}:
            rel = src.relative_to(seed_root)
            dest = raw_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            count += 1
    print(f"Synced {count} files → {raw_dir}")
    return count


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_key(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]


def build_document_rows(raw_dir: Path) -> list[dict]:
    """Scan lake raw files and build bronze document records."""
    meta_path = raw_dir / "sharepoint-meta.json"
    meta_by_file = load_sharepoint_meta(meta_path if meta_path.exists() else None)

    rows: list[dict] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == "sharepoint-meta.json":
            continue

        rel_path = str(path.relative_to(raw_dir))
        text = path.read_text(encoding="utf-8")
        sp = meta_by_file.get(path.name, {})
        doc_type = sp.get("doc_type") or (
            path.parent.name if path.parent != raw_dir else "general"
        )

        rows.append(
            {
                "doc_key": _doc_key(rel_path),
                "file_name": path.name,
                "rel_path": rel_path,
                "content": text,
                "content_hash": _content_hash(text),
                "byte_size": path.stat().st_size,
                "department": sp.get("department"),
                "confidentiality": sp.get("confidentiality"),
                "series": sp.get("series"),
                "doc_type": doc_type,
                "sharepoint_path": sp.get("sharepoint_path"),
                "rights_tier": derive_rights_tier(sp),
                "tags": json.dumps(sp.get("tags") or [], ensure_ascii=False),
                "territory_status": json.dumps(
                    sp.get("territory_status") or {}, ensure_ascii=False
                ),
                "source_system": "lake_raw",
            }
        )
    return rows


def write_parquet(rows: list[dict], parquet_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow required for parquet export.\n"
            "  pip install -r requirements-pipelines.txt"
        ) from exc

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path, compression="snappy")


def load_previous_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def diff_manifest(previous: dict | None, rows: list[dict]) -> dict:
    """Compare new lake scan to prior ingest_manifest.json (keyed by rel_path)."""
    new_by_rel = {r["rel_path"]: r for r in rows}
    if not previous:
        return {
            "has_previous": False,
            "added": sorted(new_by_rel.keys()),
            "removed": [],
            "changed_hash": [],
            "unchanged_count": 0,
        }

    old_by_rel = {
        d["rel_path"]: d
        for d in previous.get("documents") or []
        if d.get("rel_path")
    }
    added = sorted(set(new_by_rel) - set(old_by_rel))
    removed = sorted(set(old_by_rel) - set(new_by_rel))
    changed_hash: list[str] = []
    unchanged_count = 0
    for rel in sorted(set(new_by_rel) & set(old_by_rel)):
        if new_by_rel[rel]["content_hash"] != old_by_rel[rel].get("content_hash"):
            changed_hash.append(rel)
        else:
            unchanged_count += 1

    return {
        "has_previous": True,
        "previous_run_id": previous.get("run_id"),
        "previous_ingested_at": previous.get("ingested_at"),
        "added": added,
        "removed": removed,
        "changed_hash": changed_hash,
        "unchanged_count": unchanged_count,
    }


def print_manifest_diff(diff: dict) -> None:
    if not diff.get("has_previous"):
        print(f"Manifest diff: first run ({len(diff.get('added', []))} documents)")
        return
    print(
        "Manifest diff:",
        f"added={len(diff['added'])}",
        f"removed={len(diff['removed'])}",
        f"changed_hash={len(diff['changed_hash'])}",
        f"unchanged={diff['unchanged_count']}",
    )
    if diff["added"]:
        print("  added:", ", ".join(diff["added"][:8]), ("..." if len(diff["added"]) > 8 else ""))
    if diff["removed"]:
        print("  removed:", ", ".join(diff["removed"][:8]), ("..." if len(diff["removed"]) > 8 else ""))
    if diff["changed_hash"]:
        print("  changed:", ", ".join(diff["changed_hash"][:8]), ("..." if len(diff["changed_hash"]) > 8 else ""))


def write_manifest(
    *,
    rows: list[dict],
    raw_dir: Path,
    parquet_path: Path,
    manifest_path: Path,
    synced_files: int | None,
    diff_from_previous: dict | None = None,
) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "run_id": hashlib.sha256(f"{now}|{len(rows)}".encode()).hexdigest()[:16],
        "ingested_at": now,
        "raw_dir": str(raw_dir),
        "parquet_path": str(parquet_path),
        "document_count": len(rows),
        "synced_source_files": synced_files,
        "doc_types": {},
        "documents": [
            {
                "doc_key": row["doc_key"],
                "file_name": row["file_name"],
                "rel_path": row["rel_path"],
                "content_hash": row["content_hash"],
                "doc_type": row["doc_type"],
                "rights_tier": row["rights_tier"],
            }
            for row in rows
        ],
    }
    if diff_from_previous is not None:
        manifest["diff_from_previous"] = diff_from_previous
    for row in rows:
        doc_type = row["doc_type"] or "unknown"
        manifest["doc_types"][doc_type] = manifest["doc_types"].get(doc_type, 0) + 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def run_batch_ingest(
    *,
    raw_dir: Path,
    parquet_path: Path,
    manifest_path: Path,
    skip_sync: bool = True,
    emit_events_flag: bool = False,
    emit_all_events: bool = False,
    events_dir: Path | None = None,
) -> dict:
    synced = None if skip_sync else sync_seed_to_lake(raw_dir)
    if skip_sync and not raw_dir.exists():
        raise SystemExit(
            f"Raw lake not found: {raw_dir}\n"
            "Publish from applied-ai-docs or run with --seed-from-synthetic (dev seed only)."
        )

    rows = build_document_rows(raw_dir)
    if not rows:
        raise SystemExit(f"No .md/.txt documents under {raw_dir}")

    previous = load_previous_manifest(manifest_path)
    diff = diff_manifest(previous, rows)
    print_manifest_diff(diff)

    write_parquet(rows, parquet_path)
    manifest = write_manifest(
        rows=rows,
        raw_dir=raw_dir,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        synced_files=synced,
        diff_from_previous=diff,
    )

    print(f"Wrote {len(rows)} documents → {parquet_path}")
    print(f"Manifest → {manifest_path}")
    print(f"Doc types: {manifest['doc_types']}")

    if emit_events_flag:
        target = events_dir or LAKE_EVENTS
        if emit_all_events or not diff.get("has_previous"):
            event_paths = emit_events(
                rows,
                target,
                raw_dir=raw_dir,
                correlation_id=manifest["run_id"],
            )
        else:
            event_paths = emit_delta_events(
                diff,
                rows,
                previous,
                target,
                raw_dir=raw_dir,
                correlation_id=manifest["run_id"],
            )
        print(f"Emitted {len(event_paths)} document_ingested events → {target}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ingest published lake raw corpus to parquet manifest"
    )
    parser.add_argument("--raw-dir", type=Path, default=LAKE_RAW)
    parser.add_argument("--parquet-dir", type=Path, default=LAKE_PARQUET)
    parser.add_argument(
        "--seed-from-synthetic",
        action="store_true",
        help="Dev only: copy applied-ai-docs seed/harbor-line into lake raw before manifest",
    )
    parser.add_argument(
        "--emit-events",
        action="store_true",
        help="Write document_ingested JSON events for embed_worker.py (delta vs prior manifest when present)",
    )
    parser.add_argument(
        "--emit-events-all",
        action="store_true",
        help="With --emit-events, emit upsert for every document (ignore manifest diff)",
    )
    parser.add_argument("--events-dir", type=Path, default=LAKE_EVENTS)
    args = parser.parse_args()

    parquet_path = args.parquet_dir / "documents.parquet"
    manifest_path = args.parquet_dir / "ingest_manifest.json"

    run_batch_ingest(
        raw_dir=args.raw_dir,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        skip_sync=not args.seed_from_synthetic,
        emit_events_flag=args.emit_events,
        emit_all_events=args.emit_events_all,
        events_dir=args.events_dir,
    )


if __name__ == "__main__":
    main()
