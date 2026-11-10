import re
from urllib.parse import parse_qs, urlsplit


ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"}
VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")


def is_valid_youtube_url(url: str) -> bool:
    """Return whether a URL uses a supported YouTube host and video path."""
    if not isinstance(url, str) or not url.strip():
        return False

    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or hostname not in ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
    ):
        return False

    if hostname in {"youtu.be", "www.youtu.be"}:
        path_parts = [part for part in parsed.path.split("/") if part]
        return len(path_parts) == 1 and VIDEO_ID_PATTERN.fullmatch(path_parts[0]) is not None

    if parsed.path == "/watch":
        video_ids = parse_qs(parsed.query).get("v", [])
        return len(video_ids) == 1 and VIDEO_ID_PATTERN.fullmatch(video_ids[0]) is not None

    path_parts = [part for part in parsed.path.split("/") if part]
    return (
        len(path_parts) == 2
        and path_parts[0] == "shorts"
        and VIDEO_ID_PATTERN.fullmatch(path_parts[1]) is not None
    )
