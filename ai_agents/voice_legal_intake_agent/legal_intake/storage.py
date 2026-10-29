"""Local SQLite storage for reviewed intake summaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from legal_intake.models import CaseRecord, IntakeFacts


DATABASE_PATH = Path("data") / "intakes.sqlite3"


def _connection() -> sqlite3.Connection:
    """Create the local database only when a reviewed draft is saved."""

    DATABASE_PATH.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS intakes (
        case_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        summary_markdown TEXT NOT NULL,
        facts_json TEXT NOT NULL
        )"""
    )
    return connection


def save_reviewed_intake(summary_markdown: str, facts: IntakeFacts) -> CaseRecord:
    """Save only a human-reviewed summary to the local SQLite database."""

    record = CaseRecord(
        case_id=str(uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        summary_markdown=summary_markdown,
        facts=facts,
    )
    with _connection() as connection:
        connection.execute(
            "INSERT INTO intakes VALUES (?, ?, ?, ?)",
            (record.case_id, record.created_at, record.summary_markdown, record.facts.model_dump_json()),
        )
    return record


def load_recent_intakes(limit: int = 5) -> list[CaseRecord]:
    """Load the most recent local records for the dashboard only."""

    if not DATABASE_PATH.exists():
        return []
    with _connection() as connection:
        rows = connection.execute(
            "SELECT case_id, created_at, summary_markdown, facts_json FROM intakes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        CaseRecord(
            case_id=row[0],
            created_at=row[1],
            summary_markdown=row[2],
            facts=IntakeFacts.model_validate(json.loads(row[3])),
        )
        for row in rows
    ]
