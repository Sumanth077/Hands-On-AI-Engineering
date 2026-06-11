# Marketing Strategy Agent

> Generate a full marketing campaign from a product description and target audience using three specialist AI agents working in sequence.

A multi-agent system that develops comprehensive marketing campaigns from a product description and target audience. Three specialist agents work sequentially, each building on the previous output to produce a full campaign plan.

## Demo

![Demo](assets/demo.gif)

## Overview

Describe your product and target audience, and three AI agents handle the rest. The Market Analyst researches competitors, trends, and audience behaviour using live web search. The Strategy Officer turns those findings into a positioning statement, messaging pillars, channel mix, and 90-day rollout plan. The Creative Director delivers headlines, ad copy, social posts, content ideas, and a launch week playbook, all in one run.

## Features

- Three specialist agents running in sequence: Market Analyst, Strategy Officer, Creative Director
- Serper-powered web search for grounded, current market research
- Output split across three tabs: Market Research, Marketing Strategy, Creative Campaign
- Six built-in sample prompts to get started immediately
- Powered by deepseek-v4-flash via Orq.ai's OpenAI-compatible router

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | deepseek-v4-flash via Orq.ai (`alibaba/deepseek-v4-flash`) |
| Agent Orchestration | OpenAI SDK (sequential pipeline) |
| Web Search | Serper API (via httpx) |
| UI | Gradio |

## Prerequisites

- Python 3.10 or later
- An Orq.ai account and API key at [orq.ai](https://orq.ai) (free tier available)
- A Serper API key at [serper.dev](https://serper.dev) (free tier available)

## Installation

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/marketing_strategy_agent
cp .env.example .env
```

Add your `ORQ_API_KEY` and `SERPER_API_KEY` to `.env`, then install dependencies:

```bash
uv sync
```

## Usage

```bash
uv run python app.py
```

Open `http://localhost:7860`, fill in the product description and target audience, and click Generate.

## Agent Roles

| Agent | Role | Tools |
|---|---|---|
| Market Analyst | Researches market size, competitors, audience pain points, and trends | Serper web search |
| Strategy Officer | Builds positioning, messaging pillars, channel strategy, KPIs, and rollout plan | None |
| Creative Director | Writes campaign name, headlines, ad copy, social posts, and launch playbook | None |

## Environment Variables

| Variable | Description |
|---|---|
| `ORQ_API_KEY` | Orq.ai API key for routing deepseek-v4-flash |
| `SERPER_API_KEY` | Serper API key for web search (Market Analyst) |

## Project Structure

```text
marketing_strategy_agent/
├── marketing_strategy_agent/
│   ├── __init__.py
│   ├── agents.py      # Market Analyst, Strategy Officer, Creative Director
│   ├── pipeline.py    # Sequential runner that chains the three agents
│   └── tools.py       # Serper web search
├── app.py             # Gradio UI
├── pyproject.toml
├── .env.example
└── assets/
    └── demo.gif
```

## How It Works

```
User enters product description + target audience
    │
    ▼
Market Analyst searches the web for competitors, trends, audience data
    │
    ▼
Strategy Officer receives the research and formulates positioning,
messaging pillars, channel mix, KPIs, and a 90-day rollout plan
    │
    ▼
Creative Director receives the strategy and produces headlines,
ad copy, social posts, content ideas, and a launch week playbook
    │
    ▼
Results displayed across three tabs in the Gradio UI
```

---

[⬆ Back to Top](#marketing-strategy-agent)
