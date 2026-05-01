# Startup Analyst

> Elite startup due-diligence agent that scrapes company websites and produces investment-grade reports

![Startup Analyst Demo](assets/demo.gif)

## Overview

Startup Analyst is a single-agent AI system that takes a company name and URL, crawls its website, and generates a structured report covering market position, financials, team, risks, and strategic recommendations.

You give it a prompt like "Analyse Mistral AI (https://mistral.ai)" and it scrapes the site, pulls together everything it can find, and returns a clean report you could hand to an investor.

The agent is powered by **MiniMax M2.5** (via OpenRouter) and uses Firecrawl to extract content from web pages.

## Features

- **Website scraping**: Pulls structured content from specific pages (team, pricing, about, products)
- **Full-site crawling**: Discovers and extracts content across multiple pages in one pass
- **Investment-grade reports**: Covers company profile, market opportunity, financials, risks, and strategic recommendations
- **Streaming UI**: Results appear progressively as the agent works, with a clean Gradio interface
- **Ready-made examples**: One-click analysis for xAI, Perplexity, Cursor, ElevenLabs, and Mistral

## Tech Stack

**Frameworks and Libraries:**
- [Agno](https://github.com/agno-agi/agno) — agent framework
- [Gradio](https://www.gradio.app/) — web UI with streaming support

**Models and APIs:**
- [MiniMax M2.5](https://openrouter.ai/minimax/minimax-m2.5) via [OpenRouter](https://openrouter.ai/) — reasoning and report generation
- [Firecrawl](https://firecrawl.dev/) — website scraping and crawling

## Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for:
  - [ ] OpenRouter — get yours at https://openrouter.ai
  - [ ] Firecrawl — get yours at https://firecrawl.dev

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/startup_analyst
```

### 2. Install Dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

## Usage

```bash
uv run python app.py
```

Then open http://localhost:7860 in your browser.

Enter a prompt like:

```
Perform a comprehensive startup intelligence analysis on Mistral AI (https://mistral.ai)
```

The agent will scrape the site, crawl key pages, and stream back a structured report.

## Project Structure

```
startup_analyst/
├── agent.py          # Agent definition and instructions
├── app.py            # Gradio UI and streaming logic
├── pyproject.toml    # Dependencies
├── .env.example      # Environment variables template
└── README.md         # This file
```

## How It Works

The agent follows a four-part analysis framework:

1. **Foundation analysis** — company name, founding date, location, value proposition, and team
2. **Market intelligence** — target market, competitive positioning, and business model
3. **Financial assessment** — funding history, revenue indicators, and growth metrics
4. **Risk evaluation** — market, technology, team, and financial risks

Firecrawl handles two modes of web access. The scrape tool extracts structured content from a specific URL. The crawl tool walks an entire site and pulls content from every page it finds. The agent decides which to use depending on what it needs.

The Gradio app buffers the model's stream and only starts rendering once it detects the first markdown heading. This suppresses any model narration before the actual report begins.

[Back to Top](#startup-analyst)
