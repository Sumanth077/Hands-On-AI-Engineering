"""
fetch_rss.py — Fetch and filter articles from RSS feeds listed in sources.json.

Importable as a module:
    from scripts.fetch_rss import fetch_articles

Or run directly:
    python scripts/fetch_rss.py --hours 24 --sources sources.json
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser


FETCH_TIMEOUT = 15  # seconds per feed request


def _dbg(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_sources(sources_path: str | Path) -> list[dict]:
    """Load RSS feed list from a JSON file.

    Expected format: a list of objects with at least a "url" key,
    and an optional "name" key for display purposes.

    Example:
        [{"name": "Andrej Karpathy", "url": "https://karpathy.ai/feed.xml"}, ...]
    """
    path = Path(sources_path)
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        sources = json.load(f)

    if not isinstance(sources, list):
        raise ValueError("sources.json must contain a JSON array")

    return sources


def _parse_entry_date(entry) -> datetime | None:
    """Extract a timezone-aware datetime from a feedparser entry.

    Tries published_parsed first, then updated_parsed.
    Returns None if neither is available or parseable.
    """
    for attr in ("published_parsed", "updated_parsed"):
        struct = getattr(entry, attr, None)
        if struct is not None:
            try:
                # feedparser returns time.struct_time in UTC
                ts = time.mktime(struct)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                continue
    return None


def _extract_content(entry) -> str:
    """Return a short text snippet from the entry body.

    Prefers the full content field, falls back to summary/description.
    Strips to a max of 500 characters so downstream code gets a preview,
    not the full HTML blob.
    """
    # Try content (may be a list of content objects)
    content_list = getattr(entry, "content", None)
    if content_list:
        raw = content_list[0].get("value", "")
    else:
        raw = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""

    # Very light cleanup: collapse whitespace, no HTML stripping here
    text = " ".join(raw.split())
    return text[:500]


def fetch_feed(source: dict, cutoff: datetime) -> list[dict]:
    """Fetch a single RSS/Atom feed and return articles newer than cutoff.

    Args:
        source: dict with at least "url"; optionally "name".
        cutoff: timezone-aware datetime; articles older than this are dropped.

    Returns:
        List of article dicts. Empty list on any error or if no recent articles.
    """
    url = (source.get("url") or source.get("xmlUrl") or "").strip()
    name = source.get("name", url)

    if not url:
        return []

    try:
        # feedparser respects the timeout via socket default — we set it per call
        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(FETCH_TIMEOUT)
        try:
            feed = feedparser.parse(url)
        finally:
            socket.setdefaulttimeout(old_timeout)

        # feedparser doesn't raise on HTTP errors; check bozo flag for malformed feeds
        if feed.bozo and not feed.entries:
            _dbg(f"ERROR  {url} — bozo feed, no entries: {feed.bozo_exception}")
            return []

        articles = []
        for entry in feed.entries:
            pub_date = _parse_entry_date(entry)

            # Skip articles we can't date, or that are older than the cutoff
            if pub_date is None or pub_date < cutoff:
                continue

            articles.append({
                "title":   (getattr(entry, "title", "") or "").strip(),
                "link":    (getattr(entry, "link",  "") or "").strip(),
                "source":  name,
                "pubDate": pub_date.isoformat(),
                "content": _extract_content(entry),
            })

        return articles

    except Exception as exc:
        _dbg(f"ERROR  {url} — {type(exc).__name__}: {exc}")
        # Swallow all errors (network, timeout, parse) so one bad feed
        # never blocks the rest of the pipeline.
        return []


def fetch_articles(
    sources_path: str | Path = "sources.json",
    hours: int = 24,
    max_workers: int = 15,
) -> list[dict]:
    """Fetch articles from all feeds in sources_path published within the last `hours`.

    Args:
        sources_path: Path to the JSON file listing RSS feeds.
        hours:        How far back to look (default 24 hours).
        max_workers:  Number of concurrent fetch threads (default 15).

    Returns:
        List of article dicts sorted by pubDate descending (newest first).
    """
    sources = load_sources(sources_path)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)

    all_articles: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_feed, source, cutoff): source
            for source in sources
        }
        for future in as_completed(futures):
            try:
                articles = future.result()
                all_articles.extend(articles)
            except Exception:
                # future.result() shouldn't raise since fetch_feed catches everything,
                # but guard anyway.
                pass

    # Sort newest-first so callers can easily take the top N recent articles
    all_articles.sort(key=lambda a: a["pubDate"], reverse=True)

    return all_articles


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch recent articles from RSS feeds listed in sources.json."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Only include articles published within this many hours (default: 24).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=str(Path(__file__).parent.parent / "sources.json"),
        help="Path to sources.json (default: ../sources.json relative to this script).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    articles = fetch_articles(sources_path=args.sources, hours=args.hours)

    print(json.dumps(articles, indent=2, ensure_ascii=False))
    print(f"\n# {len(articles)} articles fetched from the last {args.hours} hours.",
          flush=True)
