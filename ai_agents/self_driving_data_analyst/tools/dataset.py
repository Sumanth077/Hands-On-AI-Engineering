"""
inspect_dataset() - the agent's first move in almost every investigation.

Deliberately kept primitive: it returns schema, types, row counts, and a
few sample rows, and nothing more. It does not summarize, does not guess
at what's interesting, and does not suggest what to check next. That
judgment is the model's job, not the tool's.
"""

from __future__ import annotations

import duckdb


def inspect_dataset(con: duckdb.DuckDBPyConnection) -> dict:
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    schema_info: dict[str, dict] = {}
    for table_name in tables:
        columns_df = con.execute(f"DESCRIBE {table_name}").fetchdf()
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        sample_rows = (
            con.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf().to_dict(orient="records")
        )
        schema_info[table_name] = {
            "columns": [
                {"name": r["column_name"], "type": r["column_type"]}
                for r in columns_df.to_dict(orient="records")
            ],
            "row_count": row_count,
            "sample_rows": sample_rows,
        }

    return {"tables": schema_info}
