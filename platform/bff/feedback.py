"""Persistence for human feedback (SQLite in localhost, PostgreSQL in stage/production)."""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

BFF_DIR = Path(__file__).resolve().parent
LAB_ROOT = BFF_DIR.parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from shared.config import get_env, state_store_backend  # noqa: E402
from stores.pgvector import DEFAULT_DATABASE_URL  # noqa: E402

DEFAULT_DB_PATH = BFF_DIR / "data" / "feedback.db"

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

_PG_TABLE_READY = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _use_postgres(db_path: Path | None) -> bool:
    if db_path is not None:
        return False
    return state_store_backend() == "postgres"


def _connect_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_pg():
    db_url = get_env("DATABASE_URL", DEFAULT_DATABASE_URL)
    timeout = int(get_env("PG_CONNECT_TIMEOUT", "5") or "5")
    return psycopg2.connect(db_url, connect_timeout=timeout)


def _ensure_pg_table() -> None:
    global _PG_TABLE_READY
    if _PG_TABLE_READY:
        return
    with _connect_pg() as conn:
        with conn.cursor() as cur:
            for stmt in [s.strip() for s in _CREATE_SQL.split(";") if s.strip()]:
                cur.execute(stmt)
        conn.commit()
    _PG_TABLE_READY = True


def init_db(db_path: Path | None = None) -> Path:
    if _use_postgres(db_path):
        _ensure_pg_table()
        return DEFAULT_DB_PATH
    path = db_path or DEFAULT_DB_PATH
    with _connect_sqlite(path) as conn:
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
    row_id = str(uuid.uuid4())
    created_at = _now_iso()
    sources_json = json.dumps(sources or [], ensure_ascii=False)

    if _use_postgres(db_path):
        _ensure_pg_table()
        with _connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback (
                        id, session_id, prompt, response, rating,
                        sources_json, comment, client_platform, prompt_version, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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

    path = init_db(db_path)
    with _connect_sqlite(path) as conn:
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
    if _use_postgres(db_path):
        _ensure_pg_table()
        with _connect_pg() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, session_id, prompt, response, rating, sources_json,
                           comment, client_platform, prompt_version, created_at
                    FROM feedback
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
    else:
        path = init_db(db_path)
        with _connect_sqlite(path) as conn:
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
