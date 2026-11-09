"""SQLite access functions bound to one database path per agent run."""

import sqlite3
from pathlib import Path


def create_database_functions(db_path):
    """Create schema/query functions that always use the supplied database."""
    database_path = Path(db_path)

    def get_schema() -> str:
        """Return the full database schema — table names, columns, and types."""
        conn = sqlite3.connect(database_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            schema_parts = []
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                cols = cursor.fetchall()
                col_defs = "\n".join([f"    {col[1]:30s} {col[2]}" for col in cols])
                schema_parts.append(f"TABLE: {table}\n{col_defs}")
            return "\n\n".join(schema_parts)
        finally:
            conn.close()

    def execute_sql(query: str) -> str:
        """Execute a SELECT query against this agent run's SQLite database."""
        query = query.strip()
        if not query.upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."

        conn = sqlite3.connect(database_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            if not rows:
                return "Query executed successfully but returned no results."

            col_widths = [
                max(len(col), max((len(str(row[i])) for row in rows), default=0))
                for i, col in enumerate(columns)
            ]
            header = " | ".join(col.ljust(width) for col, width in zip(columns, col_widths))
            separator = "-+-".join("-" * width for width in col_widths)
            data_rows = [
                " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(columns)))
                for row in rows[:100]
            ]
            result = "\n".join([header, separator] + data_rows)
            if len(rows) > 100:
                result += f"\n... ({len(rows)} total rows, showing first 100)"
            return result
        except sqlite3.Error as exc:
            return f"SQL Error: {exc}"
        finally:
            conn.close()

    return get_schema, execute_sql
