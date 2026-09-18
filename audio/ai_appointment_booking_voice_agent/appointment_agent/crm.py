"""
Lightweight CRM store backed by SQLite.

Two tables: appointments (what got booked) and call_summaries (the structured
post-call record plus the Inference-generated follow-up). A single file on disk
is the whole database, which is all a demo front desk needs.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from appointment_agent.config import SETTINGS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    SETTINGS.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SETTINGS.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_ref  TEXT UNIQUE NOT NULL,
                caller_name  TEXT NOT NULL,
                caller_phone TEXT NOT NULL,
                service      TEXT NOT NULL,
                start_iso    TEXT NOT NULL,
                end_iso      TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'booked',
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS call_summaries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                caller_phone    TEXT,
                summary         TEXT,
                structured_json TEXT,
                follow_up       TEXT,
                created_at      TEXT NOT NULL
            );
            """
        )


def is_slot_taken(start_iso: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM appointments WHERE start_iso = ? AND status = 'booked' LIMIT 1",
            (start_iso,),
        ).fetchone()
        return row is not None


def add_appointment(
    booking_ref: str,
    caller_name: str,
    caller_phone: str,
    service: str,
    start_iso: str,
    end_iso: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO appointments
                (booking_ref, caller_name, caller_phone, service, start_iso, end_iso, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'booked', ?)
            """,
            (booking_ref, caller_name, caller_phone, service, start_iso, end_iso, _now_iso()),
        )


def list_appointments() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments ORDER BY start_iso DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def add_summary(
    conversation_id: str | None,
    caller_phone: str | None,
    summary: str | None,
    structured: dict | None,
    follow_up: str | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO call_summaries
                (conversation_id, caller_phone, summary, structured_json, follow_up, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                caller_phone,
                summary,
                json.dumps(structured) if structured is not None else None,
                follow_up,
                _now_iso(),
            ),
        )


def list_summaries() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM call_summaries ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
