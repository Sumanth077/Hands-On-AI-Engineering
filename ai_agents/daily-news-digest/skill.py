"""
skill.py — Standalone Daily AI Digest pipeline.

Fetches articles from RSS feeds, calls MiniMax M2.7 to score and select
the top 3, formats the digest, and sends it to Telegram.

Usage:
    py skill.py
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before anything else
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv optional; variables may already be in the environment

# ---------------------------------------------------------------------------
# Fetch articles
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from scripts.fetch_rss import fetch_articles

SOURCES_PATH = Path(__file__).parent / "sources.json"

articles = fetch_articles(sources_path=SOURCES_PATH, hours=24)

if not articles:
    print("No articles fetched in the last 24 hours. Exiting.", file=sys.stderr)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Build article list for the LLM
# ---------------------------------------------------------------------------
today = date.today().strftime("%B %d, %Y").replace(" 0", " ")

article_lines = []
for i, article in enumerate(articles, start=1):
    article_lines.append(f"[{i}] {article['title']}")
    article_lines.append(f"    Source:  {article['source']}")
    article_lines.append(f"    Date:    {article['pubDate']}")
    article_lines.append(f"    URL:     {article['link']}")
    if article.get("content"):
        article_lines.append(f"    Preview: {article['content']}")
    article_lines.append("")

article_block = "\n".join(article_lines)

system_prompt = (
    "You are an AI news curator. Your job is to select the 3 most significant "
    "AI/ML articles from a list and produce a structured JSON response.\n\n"
    "Return ONLY valid JSON with this exact structure:\n"
    "{\n"
    '  "selected": [\n'
    "    {\n"
    '      "title": "...",\n'
    '      "source": "...",\n'
    '      "url": "...",\n'
    '      "category": "Breaking" | "Important" | "Notable",\n'
    '      "summary": "2-3 sentence summary"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Select ONLY the 3 most significant articles. No more than 3.\n"
    "- Score each article 1-10 for AI/ML relevance and quality, then pick the top 3.\n"
    "- Assign each a category: Breaking (major announcements/releases), "
    "Important (significant developments), or Notable (interesting but less urgent).\n"
    "- Write a 2-3 sentence summary per article.\n"
    "- Use the exact URL from the article listing.\n"
    "- Return ONLY the JSON object, no other text."
)

user_prompt = (
    f"Here are today's articles from 92 AI/tech blogs "
    f"({len(articles)} fetched in the last 24 hours).\n\n"
    f"---\nARTICLES:\n\n{article_block}"
)

# ---------------------------------------------------------------------------
# Call MiniMax M2.7
# ---------------------------------------------------------------------------
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
if not MINIMAX_API_KEY:
    print("Error: MINIMAX_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

payload = json.dumps({
    "model": "MiniMax-M2.7",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    "temperature": 0.2,
}).encode("utf-8")

req = urllib.request.Request(
    "https://api.minimaxi.chat/v1/chat/completions",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        response_body = json.loads(resp.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"MiniMax API error {e.code}: {e.read().decode()}", file=sys.stderr)
    sys.exit(1)

raw_content = response_body["choices"][0]["message"]["content"].strip()

# Strip <think>...</think> block if present
if "</think>" in raw_content:
    raw_content = raw_content.split("</think>", 1)[1].strip()

# Strip markdown code fences if present
if raw_content.startswith("```"):
    raw_content = raw_content.split("\n", 1)[1]
    raw_content = raw_content.rsplit("```", 1)[0].strip()

try:
    llm_result = json.loads(raw_content)
    selected = llm_result["selected"]
    required_fields = {"title", "summary", "source", "url", "category"}
    for i, article in enumerate(selected):
        missing = required_fields - article.keys()
        if missing:
            print(f"Article {i} missing fields: {missing}", file=sys.stderr)
            sys.exit(1)
except (json.JSONDecodeError, KeyError) as e:
    print(f"Failed to parse LLM response: {e}", file=sys.stderr)
    sys.exit(1)


def escape_md(text: str) -> str:
    """Escape Markdown special characters for Telegram."""
    for char in ('_', '*', '`', '['):
        text = text.replace(char, '\\' + char)
    return text


# ---------------------------------------------------------------------------
# Format the Telegram message
# ---------------------------------------------------------------------------
CATEGORY_HEADERS = {
    "Breaking": "🔴 BREAKING",
    "Important": "🟡 IMPORTANT",
    "Notable": "🔵 NOTABLE",
}
CATEGORY_EMOJI = {
    "Breaking": "🔴",
    "Important": "🟡",
    "Notable": "🔵",
}
CATEGORY_ORDER = ["Breaking", "Important", "Notable"]

# Group articles by category
grouped: dict[str, list] = {}
for article in selected:
    cat = article.get("category", "Notable")
    grouped.setdefault(cat, []).append(article)

message_parts = [f"🗞️ *Daily AI Digest — {today}*"]

for cat in CATEGORY_ORDER:
    if cat not in grouped:
        continue
    message_parts.append("")
    message_parts.append(f"*{CATEGORY_HEADERS[cat]}*")
    for art in grouped[cat]:
        emoji = CATEGORY_EMOJI[cat]
        title = escape_md(art["title"])
        summary = escape_md(art["summary"])
        source = art["source"]
        url = art["url"]
        message_parts.append(
            f"{emoji} *{title}* — {summary}\n"
            f"Source: {source} | [Read more]({url})"
        )

digest = "\n".join(message_parts)

# ---------------------------------------------------------------------------
# Send to Telegram
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not BOT_TOKEN or not CHAT_ID:
    print("Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set.", file=sys.stderr)
    sys.exit(1)

tg_payload = json.dumps({
    "chat_id": CHAT_ID,
    "text": digest,
    "parse_mode": "Markdown",
    "disable_web_page_preview": False,
}).encode("utf-8")

tg_req = urllib.request.Request(
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    data=tg_payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(tg_req, timeout=30) as resp:
        tg_response = json.loads(resp.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"Telegram API error {e.code}: {e.read().decode()}", file=sys.stderr)
    sys.exit(1)

if not tg_response.get("ok"):
    print(f"Telegram error: {tg_response}", file=sys.stderr)
    sys.exit(1)
