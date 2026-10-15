"""SQLite persistence for human feedback."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "feedback.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    rating TEXT NOT NULL,
    sources_json TEXT,
    comment TEXT,
    client_platform TEXT,
    prompt_version TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_session_idx ON feedback (session_id);
CREATE INDEX IF NOT EXISTS feedback_created_idx ON feedback (created_at);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> Path:
    path = db_path or DEFAULT_DB_PATH
    with _connect(path) as conn:
        conn.executescript(_CREATE_SQL)
        conn.commit()
    return path


def insert_feedback(
    *,
    session_id: str,
    prompt: str,
    response: str,
    rating: str,
    sources: list[dict[str, Any]] | None = None,
    comment: str | None = None,
    client_platform: str | None = None,
    prompt_version: str | None = None,
    db_path: Path | None = None,
) -> dict[str, str]:
    path = init_db(db_path)
    row_id = str(uuid.uuid4())
    created_at = _now_iso()
    sources_json = json.dumps(sources or [], ensure_ascii=False)

    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                id, session_id, prompt, response, rating,
                sources_json, comment, client_platform, prompt_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                session_id,
                prompt,
                response,
                rating,
                sources_json,
                comment,
                client_platform,
                prompt_version,
                created_at,
            ),
        )
        conn.commit()

    return {"id": row_id, "created_at": created_at}


def list_feedback(
    *,
    limit: int = 50,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    path = init_db(db_path)
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, prompt, response, rating, sources_json,
                   comment, client_platform, prompt_version, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "prompt": row["prompt"],
                "response": row["response"],
                "rating": row["rating"],
                "sources": json.loads(row["sources_json"] or "[]"),
                "comment": row["comment"],
                "client_platform": row["client_platform"],
                "prompt_version": row["prompt_version"],
                "created_at": row["created_at"],
            }
        )
    return results
