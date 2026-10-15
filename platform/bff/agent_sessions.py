"""Persist agent runs awaiting human approval (BFF /v1/agent/run).

Supports local SQLite (default in APP_ENV=localhost) and shared PostgreSQL via
DATABASE_URL (default in APP_ENV=stage and APP_ENV=production, or when
STATE_STORE_BACKEND=postgres).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psycopg2
from psycopg2.extras import RealDictCursor

BFF_DIR = Path(__file__).resolve().parent
LAB_ROOT = BFF_DIR.parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from shared.config import get_env, state_store_backend  # noqa: E402
from stores.pgvector import DEFAULT_DATABASE_URL  # noqa: E402

DEFAULT_DB_PATH = BFF_DIR / "data" / "agent_sessions.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    thread_id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    proposed_output TEXT NOT NULL,
    result_json TEXT NOT NULL,
    status TEXT NOT NULL,
    client_platform TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_sessions_status_idx ON agent_sessions (status);
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


def init_agent_db(db_path: Path | None = None) -> Path:
    if _use_postgres(db_path):
        _ensure_pg_table()
        return DEFAULT_DB_PATH
    path = db_path or DEFAULT_DB_PATH
    with _connect_sqlite(path) as conn:
        conn.executescript(_CREATE_SQL)
        conn.commit()
    return path


def save_pending_session(
    *,
    prompt: str,
    proposed_output: str,
    result_payload: dict[str, Any],
    thread_id: str | None = None,
    client_platform: str | None = None,
    db_path: Path | None = None,
) -> str:
    tid = thread_id or str(uuid.uuid4())
    now = _now_iso()
    payload_str = json.dumps(result_payload, ensure_ascii=False)

    if _use_postgres(db_path):
        _ensure_pg_table()
        with _connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_sessions (
                        thread_id, prompt, proposed_output, result_json,
                        status, client_platform, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, 'pending', %s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE SET
                        prompt = EXCLUDED.prompt,
                        proposed_output = EXCLUDED.proposed_output,
                        result_json = EXCLUDED.result_json,
                        status = 'pending',
                        client_platform = EXCLUDED.client_platform,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (tid, prompt, proposed_output, payload_str, client_platform, now, now),
                )
            conn.commit()
        return tid

    path = init_agent_db(db_path)
    with _connect_sqlite(path) as conn:
        conn.execute(
            """
            INSERT INTO agent_sessions (
                thread_id, prompt, proposed_output, result_json,
                status, client_platform, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                prompt = excluded.prompt,
                proposed_output = excluded.proposed_output,
                result_json = excluded.result_json,
                status = 'pending',
                client_platform = excluded.client_platform,
                updated_at = excluded.updated_at
            """,
            (tid, prompt, proposed_output, payload_str, client_platform, now, now),
        )
        conn.commit()
    return tid


def get_session(thread_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    if _use_postgres(db_path):
        _ensure_pg_table()
        with _connect_pg() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM agent_sessions WHERE thread_id = %s",
                    (thread_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "thread_id": row["thread_id"],
            "prompt": row["prompt"],
            "proposed_output": row["proposed_output"],
            "result": json.loads(row["result_json"] or "{}"),
            "status": row["status"],
            "client_platform": row["client_platform"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    path = init_agent_db(db_path)
    with _connect_sqlite(path) as conn:
        row = conn.execute(
            "SELECT * FROM agent_sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "thread_id": row["thread_id"],
        "prompt": row["prompt"],
        "proposed_output": row["proposed_output"],
        "result": json.loads(row["result_json"] or "{}"),
        "status": row["status"],
        "client_platform": row["client_platform"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def resolve_session(
    thread_id: str,
    decision: Literal["approve", "reject"],
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    session = get_session(thread_id, db_path=db_path)
    if not session or session["status"] != "pending":
        return None
    status = "approved" if decision == "approve" else "rejected"
    now = _now_iso()

    if _use_postgres(db_path):
        _ensure_pg_table()
        with _connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_sessions
                    SET status = %s, updated_at = %s
                    WHERE thread_id = %s
                    """,
                    (status, now, thread_id),
                )
            conn.commit()
    else:
        path = init_agent_db(db_path)
        with _connect_sqlite(path) as conn:
            conn.execute(
                """
                UPDATE agent_sessions
                SET status = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (status, now, thread_id),
            )
            conn.commit()

    session["status"] = status
    session["updated_at"] = now
    return session
