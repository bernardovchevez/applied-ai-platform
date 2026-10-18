"""Embed worker event handling (upsert/delete) without Postgres."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINES_DIR = REPO_ROOT / "platform" / "pipelines"
for p in (str(REPO_ROOT), str(PIPELINES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from document_events import build_event, mark_processed  # noqa: E402
from embed_worker import process_event  # noqa: E402
from stores.base import IngestStats  # noqa: E402


def _write_event(
    events_dir: Path,
    raw_dir: Path,
    *,
    rel_path: str = "rights/demo-clearance.md",
    file_name: str = "demo-clearance.md",
    content_hash: str = "abc123",
    operation: str = "upsert",
) -> Path:
    row = {
        "rel_path": rel_path,
        "file_name": file_name,
        "content_hash": content_hash,
        "byte_size": 42,
        "doc_type": "memo",
        "department": "rights",
        "tags": "[]",
        "territory_status": "{}",
        "clearance_territories": [],
    }
    event = build_event(row, raw_dir=raw_dir, operation=operation)
    events_dir.mkdir(parents=True, exist_ok=True)
    path = events_dir / f"{event['document']['doc_key']}.json"
    path.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
    return path


class FakePgVectorStore:
    """Records calls; no database."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.ingested: list[tuple[dict, dict]] = []
        self.delete_results: dict[str, bool] = {}
        self.ingest_stats = IngestStats(documents=1, chunks=2, lora_tagged_chunks=0)

    def delete_document(self, doc_key: str) -> bool:
        self.deleted.append(doc_key)
        return self.delete_results.get(doc_key, True)

    def ingest_document(self, doc: dict, **kwargs) -> IngestStats:
        self.ingested.append((doc, kwargs))
        return self.ingest_stats


@pytest.fixture
def lake_dirs(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "lake" / "raw"
    events = tmp_path / "events"
    processed = tmp_path / "processed"
    raw.mkdir(parents=True)
    return {"raw": raw, "events": events, "processed": processed}


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakePgVectorStore:
    store = FakePgVectorStore()
    monkeypatch.setattr("embed_worker.PgVectorStore", lambda: store)
    return store


def test_delete_event_removes_document(lake_dirs, fake_store) -> None:
    event_path = _write_event(
        lake_dirs["events"],
        lake_dirs["raw"],
        operation="delete",
    )
    result = process_event(
        event_path,
        raw_dir=lake_dirs["raw"],
        processed_dir=lake_dirs["processed"],
        lora_tag=False,
        lora_adapter=None,
        force=False,
    )
    assert result == {"doc_key": "demo-clearance", "status": "deleted"}
    assert fake_store.deleted == ["demo-clearance"]
    marker = lake_dirs["processed"] / "demo-clearance.json"
    assert marker.exists()
    assert json.loads(marker.read_text())["status"] == "deleted"


def test_delete_event_missing_in_store(lake_dirs, fake_store) -> None:
    fake_store.delete_results["demo-clearance"] = False
    event_path = _write_event(
        lake_dirs["events"],
        lake_dirs["raw"],
        operation="delete",
    )
    result = process_event(
        event_path,
        raw_dir=lake_dirs["raw"],
        processed_dir=lake_dirs["processed"],
        lora_tag=False,
        lora_adapter=None,
        force=False,
    )
    assert result["status"] == "missing"


def test_upsert_event_ingests_lake_file(lake_dirs, fake_store) -> None:
    rel = "rights/demo-clearance.md"
    body = "# Demo\n\nEMEA clearance pending."
    path = lake_dirs["raw"] / rel
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")

    event_path = _write_event(lake_dirs["events"], lake_dirs["raw"], content_hash="hash-v1")
    result = process_event(
        event_path,
        raw_dir=lake_dirs["raw"],
        processed_dir=lake_dirs["processed"],
        lora_tag=False,
        lora_adapter=None,
        force=False,
    )
    assert result["status"] == "ok"
    assert result["chunks"] == 2
    assert len(fake_store.ingested) == 1
    doc, kwargs = fake_store.ingested[0]
    assert "EMEA" in doc["text"]
    assert kwargs["content_hash"] == "hash-v1"


def test_upsert_skips_when_store_unchanged(lake_dirs, fake_store) -> None:
    fake_store.ingest_stats = IngestStats(documents=0, chunks=0, lora_tagged_chunks=0)
    rel = "rights/demo-clearance.md"
    (lake_dirs["raw"] / rel).parent.mkdir(parents=True)
    (lake_dirs["raw"] / rel).write_text("# Demo", encoding="utf-8")
    event_path = _write_event(lake_dirs["events"], lake_dirs["raw"])

    result = process_event(
        event_path,
        raw_dir=lake_dirs["raw"],
        processed_dir=lake_dirs["processed"],
        lora_tag=False,
        lora_adapter=None,
        force=False,
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "unchanged in store"


def test_upsert_errors_when_lake_file_missing(lake_dirs, fake_store) -> None:
    event_path = _write_event(lake_dirs["events"], lake_dirs["raw"])
    result = process_event(
        event_path,
        raw_dir=lake_dirs["raw"],
        processed_dir=lake_dirs["processed"],
        lora_tag=False,
        lora_adapter=None,
        force=False,
    )
    assert result["status"] == "error"
    assert "file not found" in result["reason"]
    assert fake_store.ingested == []


def test_skips_stale_event_with_same_content_hash(lake_dirs, fake_store) -> None:
    rel = "rights/demo-clearance.md"
    (lake_dirs["raw"] / rel).parent.mkdir(parents=True)
    (lake_dirs["raw"] / rel).write_text("# Demo", encoding="utf-8")
    event_path = _write_event(lake_dirs["events"], lake_dirs["raw"], content_hash="same-hash")
    event = json.loads(event_path.read_text())
    mark_processed(event, lake_dirs["processed"], status="ok")

    result = process_event(
        event_path,
        raw_dir=lake_dirs["raw"],
        processed_dir=lake_dirs["processed"],
        lora_tag=False,
        lora_adapter=None,
        force=False,
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "unchanged content_hash"
    assert fake_store.ingested == []
