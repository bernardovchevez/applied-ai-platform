"""Manifest diff and delta event emission for ingest_batch."""

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

from document_events import emit_delta_events  # noqa: E402
from ingest_batch import diff_manifest  # noqa: E402


def _row(rel_path: str, content_hash: str, file_name: str | None = None) -> dict:
    name = file_name or Path(rel_path).name
    return {
        "doc_key": "k",
        "rel_path": rel_path,
        "file_name": name,
        "content_hash": content_hash,
    }


def test_diff_first_run_all_added() -> None:
    rows = [_row("a/x.md", "h1"), _row("b/y.md", "h2")]
    diff = diff_manifest(None, rows)
    assert diff["has_previous"] is False
    assert diff["added"] == ["a/x.md", "b/y.md"]
    assert diff["removed"] == []
    assert diff["changed_hash"] == []


def test_diff_detects_added_removed_and_changed() -> None:
    previous = {
        "run_id": "prev-run",
        "documents": [
            {"rel_path": "keep.md", "file_name": "keep.md", "content_hash": "same"},
            {"rel_path": "gone.md", "file_name": "gone.md", "content_hash": "old"},
            {"rel_path": "edit.md", "file_name": "edit.md", "content_hash": "v1"},
        ],
    }
    rows = [
        _row("keep.md", "same"),
        _row("edit.md", "v2"),
        _row("new.md", "h-new"),
    ]
    diff = diff_manifest(previous, rows)
    assert diff["has_previous"] is True
    assert diff["previous_run_id"] == "prev-run"
    assert diff["added"] == ["new.md"]
    assert diff["removed"] == ["gone.md"]
    assert diff["changed_hash"] == ["edit.md"]
    assert diff["unchanged_count"] == 1


def test_emit_delta_events_writes_upsert_and_delete(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    events = tmp_path / "events"
    previous = {
        "documents": [
            {"rel_path": "gone.md", "file_name": "gone.md", "content_hash": "dead"},
        ],
    }
    diff = {
        "added": ["rights/new.md"],
        "removed": ["gone.md"],
        "changed_hash": ["rights/changed.md"],
    }
    rows = [
        _row("rights/new.md", "h-new"),
        _row("rights/changed.md", "h-changed"),
    ]
    (raw / "rights").mkdir(parents=True)
    (raw / "rights/new.md").write_text("# new", encoding="utf-8")
    (raw / "rights/changed.md").write_text("# changed", encoding="utf-8")

    paths = emit_delta_events(
        diff,
        rows,
        previous,
        events,
        raw_dir=raw,
        correlation_id="test-run",
    )
    assert len(paths) == 3
    by_name = {p.stem: json.loads(p.read_text()) for p in paths}
    assert by_name["new"]["operation"] == "upsert"
    assert by_name["changed"]["operation"] == "upsert"
    assert by_name["gone"]["operation"] == "delete"
    assert by_name["gone"]["document"]["doc_key"] == "gone"
