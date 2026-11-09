"""Validation and canonicalization for YouTube video URLs."""

import re
from urllib.parse import parse_qs, urlsplit


VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def extract_video_id(url: str) -> str | None:
    """Return a video ID only for supported YouTube URL hosts and paths."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None

    if parsed.scheme.lower() not in {"http", "https"} or port is not None:
        return None

    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS or parsed.username or parsed.password:
        return None

    if host in {"youtu.be", "www.youtu.be"}:
        parts = parsed.path.strip("/").split("/")
        video_id = parts[0] if len(parts) == 1 else ""
    elif parsed.path == "/watch":
        values = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
        video_id = values[0] if len(values) == 1 else ""
    else:
        parts = parsed.path.strip("/").split("/")
        video_id = parts[1] if len(parts) == 2 and parts[0] in {
            "embed", "v", "shorts"
        } else ""

    return video_id if VIDEO_ID_PATTERN.fullmatch(video_id) else None


def youtube_watch_url(video_id: str) -> str:
    """Build the canonical download URL for a validated YouTube video ID."""
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID")
    return f"https://www.youtube.com/watch?v={video_id}"
