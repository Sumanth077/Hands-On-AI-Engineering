"""
Marketing Strategy Agent - Tools.
"""

from __future__ import annotations

import os

import httpx


def search_web(query: str, num_results: int = 5) -> str:
    """Search the web using Serper and return formatted results."""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return "Web search unavailable - SERPER_API_KEY not set."

    try:
        r = httpx.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": num_results},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        results = []

        if "answerBox" in data:
            ab = data["answerBox"]
            answer = ab.get("answer") or ab.get("snippet", "")
            if answer:
                results.append(f"**Quick Answer:** {answer}")

        for item in data.get("organic", [])[:num_results]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            results.append(f"- **{title}**\n  {snippet}\n  {link}")

        return "\n\n".join(results) if results else "No results found."

    except Exception as exc:
        return f"Search failed: {exc}"
