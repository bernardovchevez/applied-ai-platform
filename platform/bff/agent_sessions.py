"""Persist agent runs awaiting human approval (BFF /v1/agent/run)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "agent_sessions.db"

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_agent_db(db_path: Path | None = None) -> Path:
    path = db_path or DEFAULT_DB_PATH
    with _connect(path) as conn:
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
    path = init_agent_db(db_path)
    tid = thread_id or str(uuid.uuid4())
    now = _now_iso()
    with _connect(path) as conn:
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
            (
                tid,
                prompt,
                proposed_output,
                json.dumps(result_payload, ensure_ascii=False),
                client_platform,
                now,
                now,
            ),
        )
        conn.commit()
    return tid


def get_session(thread_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    path = init_agent_db(db_path)
    with _connect(path) as conn:
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
    path = init_agent_db(db_path)
    session = get_session(thread_id, db_path=path)
    if not session or session["status"] != "pending":
        return None
    status = "approved" if decision == "approve" else "rejected"
    now = _now_iso()
    with _connect(path) as conn:
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
