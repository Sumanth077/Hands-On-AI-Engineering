# Brand Monitor

![Brand Monitor Demo](assets/demo.gif)

Get a structured intelligence report on any brand across Web, YouTube, Twitter/X, and LinkedIn in a single run. Powered by [Scrapingdog](https://www.scrapingdog.com) for data collection and DeepSeek V4 Flash for analysis.

## Overview

Each run collects platform data via Scrapingdog and passes it to one LLM call per platform — 4 LLM calls total. Results are presented in a Streamlit UI with tabs per platform.

- **Web** — Google SERP + Google News via Scrapingdog
- **YouTube** — YouTube search results, with Google SERP fallback
- **Twitter/X** — Google SERP filtered to twitter.com
- **LinkedIn** — Google SERP filtered to linkedin.com

## Setup

### 1. Prerequisites

- [UV](https://docs.astral.sh/uv/) package manager
- [Scrapingdog API key](https://www.scrapingdog.com)
- DeepSeek V4 Flash API key

### 2. Install dependencies

```bash
cd ai_agents/brand_monitor_agent
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Run

```bash
uv run streamlit run app.py
```

## Project Structure

```
brand_monitor_agent/
├── app.py                    # Streamlit UI
├── pyproject.toml
├── .env.example
└── brand_monitor_agent/
    ├── main.py               # run_brand_monitor() entry point
    ├── tools.py              # Scrapingdog API wrappers
    └── analyzers.py          # Data collection + LLM analysis per platform
```

## How It Works

1. Enter a brand name and API keys in the sidebar
2. Click **Run** — four platform analyzers execute sequentially
3. Each analyzer fetches raw data from Scrapingdog then makes one LLM call to produce a structured brief
4. Results appear in tabs: Web, YouTube, Twitter/X, LinkedIn — each with a sentiment indicator

## Environment Variables

| Variable | Description |
|---|---|
| `SCRAPINGDOG_API_KEY` | Scrapingdog API key for all data collection |
| `ORQ_API_KEY` | API key for LLM analysis |
