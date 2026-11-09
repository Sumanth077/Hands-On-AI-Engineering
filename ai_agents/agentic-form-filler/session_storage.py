"""Safe filesystem path handling for persisted form sessions."""

import re
from pathlib import Path


_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")


def session_file_path(session_dir, session_id):
    """Return a session JSON path only for a single safe path component."""
    if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
        return None
    return Path(session_dir) / f"{session_id}.json"
