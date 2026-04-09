---
name: daily-ai-news-digest
description: Fetches articles from 92 Karpathy-curated RSS feeds, scores them with an LLM, selects the top 3, and delivers a formatted digest to Telegram every morning.
trigger: "0 8 * * *"
metadata: {"openclaw": {"requires": {"env": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"], "bins": ["py"]}, "primaryEnv": "TELEGRAM_BOT_TOKEN", "emoji": ["📰"]}}
---

# Daily AI News Digest

A morning briefing skill that monitors 92 RSS feeds curated by Andrej Karpathy, surfaces the most important AI and tech stories from the last 24 hours, and delivers a clean digest to Telegram.

## What It Does

Each morning the skill wakes up, pulls fresh articles from a hand-picked list of RSS feeds, uses an LLM to score and rank them by relevance and significance, then writes a concise summary of the top 3 stories and pushes it to a Telegram chat.

## Pipeline

```
92 RSS feeds
    │
    ▼
fetch_rss.py — parallel feed fetching
    │
    ▼
filter — keep only articles published in the last 24 hours
    │
    ▼
LLM scoring — rate each article for relevance, novelty, and impact
    │
    ▼
select top 3 — ranked by score, deduplicated by URL
    │
    ▼
summarise — one-paragraph summary per article via LLM
    │
    ▼
categorise — group into themes (Breaking, Important, Notable)
    │
    ▼
Telegram — formatted message delivered to configured chat
```

## Required Environment Variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token for the Telegram bot that sends the digest |
| `TELEGRAM_CHAT_ID` | ID of the Telegram chat or channel to deliver to |

Copy `.env.example` to `.env` and fill in these values before running.

## Schedule

The skill runs on the cron expression `0 8 * * *` — every day at **08:00 UTC**.

To change the time, update the `trigger` field in the frontmatter above. For example:
- `"0 7 * * 1-5"` — weekdays at 07:00 UTC
- `"0 6 * * *"` — daily at 06:00 UTC

## Running Manually

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python skill.py

# Run only the RSS fetch step
python scripts/fetch_rss.py
```
