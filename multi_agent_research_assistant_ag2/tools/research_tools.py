# tools/research_tools.py
import json
import urllib.request
import urllib.parse
from datetime import datetime


def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo Instant Answer API.
    Returns a JSON string with top results including titles, URLs, and snippets.

    Args:
        query: Search query string.
        num_results: Number of results to return (default 5).
    """
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []
        # RelatedTopics contains substantive results
        for item in data.get("RelatedTopics", [])[:num_results]:
            if isinstance(item, dict) and "Text" in item:
                results.append({
                    "title": item.get("Text", "")[:100],
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", ""),
                })
        # Abstract is often the best single answer
        if data.get("AbstractText"):
            results.insert(0, {
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"],
            })
        return json.dumps({"query": query, "results": results, "timestamp": datetime.utcnow().isoformat()})
    except Exception as exc:
        return json.dumps({"error": str(exc), "query": query})


def fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and return the text content of a web page.

    Args:
        url: URL of the page to fetch.
        max_chars: Maximum characters to return (default 3000).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        # Strip HTML tags (basic)
        import re
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        return json.dumps({"url": url, "content": text[:max_chars]})
    except Exception as exc:
        return json.dumps({"error": str(exc), "url": url})
