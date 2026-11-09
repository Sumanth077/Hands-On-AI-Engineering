"""Validation helpers for read-only SQL queries."""

import re

FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|"
    r"GRANT|REVOKE|ATTACH|DETACH|MERGE|CALL|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)
READ_ONLY_SQL = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE | re.DOTALL)


def _validate_read_only(query: str) -> str:
    """Raise ValueError unless the query is a read-only SELECT or WITH statement."""
    cleaned = query.strip().rstrip(";")
    if FORBIDDEN_SQL.search(cleaned):
        raise ValueError("Only read-only SELECT queries are allowed.")
    if not READ_ONLY_SQL.match(cleaned):
        raise ValueError("Only read-only SELECT queries are allowed.")
    return cleaned
