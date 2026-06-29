"""
SQLite expense ledger via pandas.
Stores extracted receipt data and provides query helpers for the dashboard.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("./expenses.db")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS expenses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor      TEXT,
    date        TEXT,
    line_items  TEXT,
    subtotal    REAL,
    tax         REAL,
    total       REAL,
    category    TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE)
    conn.commit()
    return conn


def save_expense(record: dict) -> int:
    """Insert one extracted receipt record. Returns the new row id."""
    conn = _connect()
    cur = conn.execute(
        """
        INSERT INTO expenses (vendor, date, line_items, subtotal, tax, total, category)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("vendor", ""),
            record.get("date", ""),
            json.dumps(record.get("line_items", [])),
            float(record.get("subtotal") or 0),
            float(record.get("tax") or 0),
            float(record.get("total") or 0),
            record.get("category", "Other"),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def load_expenses(
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    """Return expenses as a DataFrame, optionally filtered."""
    conn = _connect()
    query = "SELECT * FROM expenses WHERE 1=1"
    params: list = []

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY date DESC, id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def delete_expense(expense_id: int) -> None:
    conn = _connect()
    conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    conn.commit()
    conn.close()


def get_category_totals(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Total spending grouped by category for the dashboard chart."""
    df = load_expenses(start_date=start_date, end_date=end_date)
    if df.empty:
        return pd.DataFrame(columns=["category", "total"])
    return (
        df.groupby("category")["total"]
        .sum()
        .reset_index()
        .sort_values("total", ascending=False)
    )


CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transport",
    "Shopping",
    "Utilities",
    "Healthcare",
    "Entertainment",
    "Other",
]
