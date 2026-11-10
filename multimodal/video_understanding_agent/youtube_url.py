"""Validation helpers for YouTube video URLs accepted by the app."""

import re
from urllib.parse import parse_qs, urlsplit


_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com"}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}
_VIDEO_ID = re.compile(r"[\w-]+", re.ASCII)


def is_valid_youtube_url(url: str) -> bool:
    """Return whether ``url`` uses a supported YouTube host and video path."""
    if not isinstance(url, str) or not url.strip():
        return False

    try:
        parsed = urlsplit(url.strip())
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return False

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not host:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if port is not None and port != (80 if scheme == "http" else 443):
        return False

    host = host.lower()
    if host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            ids = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
            return len(ids) == 1 and bool(_VIDEO_ID.fullmatch(ids[0]))

        match = re.fullmatch(r"/shorts/([\w-]+)", parsed.path, re.ASCII)
        return bool(match and _VIDEO_ID.fullmatch(match.group(1)))

    if host in _SHORT_HOSTS:
        match = re.fullmatch(r"/([\w-]+)/?", parsed.path, re.ASCII)
        return bool(match and _VIDEO_ID.fullmatch(match.group(1)))

    return False
