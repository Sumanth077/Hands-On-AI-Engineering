"""
run_sql(query) - lets the agent query the loaded datasets directly.

Errors are returned as data, not raised, so the harness can hand them back
to the model as an observation ("this column doesn't exist") instead of
crashing the investigation. See agent/harness.py for how a failed query
becomes the trigger for a follow-up inspect_dataset() call.
"""

from __future__ import annotations

import duckdb

MAX_ROWS_RETURNED = 50


def run_sql(con: duckdb.DuckDBPyConnection, query: str) -> dict:
    try:
        df = con.execute(query).fetchdf()
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this becomes model-facing data
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "columns": list(df.columns),
        "row_count": len(df),
        "rows": df.head(MAX_ROWS_RETURNED).to_dict(orient="records"),
        "truncated": len(df) > MAX_ROWS_RETURNED,
    }
