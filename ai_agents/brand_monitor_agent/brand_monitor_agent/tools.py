"""
Scrapingdog API wrappers.
Low-level functions for Google SERP, web scraping, Google News, and YouTube search.
"""

import html2text
import requests

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

SCRAPE_ENDPOINT = "https://api.scrapingdog.com/scrape"
GOOGLE_ENDPOINT = "https://api.scrapingdog.com/google"
GOOGLE_NEWS_ENDPOINT = "https://api.scrapingdog.com/google_news"
YOUTUBE_ENDPOINT = "https://api.scrapingdog.com/youtube"

_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.ignore_images = True
_h2t.body_width = 0


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _scrape_url(url: str, api_key: str, dynamic: bool = True) -> str:
    """Fetch a URL via Scrapingdog and return cleaned Markdown text."""
    params = {"api_key": api_key, "url": url, "dynamic": str(dynamic).lower()}
    resp = requests.get(SCRAPE_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    return _h2t.handle(resp.text).strip()


def _google_search(query: str, api_key: str, results: int = 10) -> list[dict]:
    """Run a Google SERP query via Scrapingdog and return organic results."""
    params = {"api_key": api_key, "query": query, "results": str(results), "country": "us"}
    resp = requests.get(GOOGLE_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("organic_results") or data.get("organic_data") or []


def _google_news(query: str, api_key: str, results: int = 8) -> list[dict]:
    """Fetch Google News results for a query via Scrapingdog."""
    params = {"api_key": api_key, "query": query, "results": str(results), "tbs": "qdr:m"}
    resp = requests.get(GOOGLE_NEWS_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("news_results", [])


def _youtube_search(query: str, api_key: str) -> list[dict]:
    """Search YouTube via Scrapingdog and return video results."""
    params = {"api_key": api_key, "search_query": query}
    resp = requests.get(YOUTUBE_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.text.strip()
    if not body:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if isinstance(data, list):
        return data
    return data.get("video_results", data.get("results", data.get("organic_results", [])))
