# Hotel Finder Agent

> A conversational hotel search agent that takes natural language queries and finds hotels based on location, dates, guest configuration, price range, star rating, and amenities. Powered by qwen3.6-flash via Orq.ai and the Trivago MCP Server.

## Demo

![Demo](assets/demo.gif)

## Overview

Describe where you want to stay in plain English and the agent handles the rest. It uses the Trivago MCP Server to resolve locations, search properties, and return detailed results including pricing, review scores, and booking links. qwen3.6-flash runs through Orq.ai's OpenAI-compatible router, and the agent loop is defined with Google ADK for use with `adk web` or as a Streamlit app.

## Features

- Natural language hotel search across city, country, or region
- Supports check-in/check-out dates, guest count, rooms, star rating, price range, and amenities
- Trivago MCP Server integration via Streamable HTTP transport
- Returns hotel name, star rating, price per night, review score, and booking link
- Conversational multi-turn memory so you can refine your search without repeating yourself
- Google ADK agent definition for `adk web` alongside a Streamlit UI

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | qwen3.6-flash via Orq.ai (`alibaba/qwen3.6-flash`) |
| Agent Orchestration | Google ADK |
| Hotel Data | Trivago MCP Server (`mcp.trivago.com`) |
| MCP Transport | Streamable HTTP (`mcp` Python library) |
| UI | Streamlit |

## Prerequisites

- Python 3.10 or later
- An Orq.ai account and API key at [orq.ai](https://orq.ai)

## Installation

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/hotel_finder_agent
cp .env.example .env
```

Add your Orq.ai API key to `.env`.

## Usage

**Streamlit app:**

```bash
uv run streamlit run app.py
```

**ADK web interface:**

```bash
uv run adk web
```

Open `http://localhost:8501` (Streamlit) or `http://localhost:8000` (ADK) and start searching.

## Example Queries

- "Find hotels in Barcelona for next weekend, 2 adults"
- "Budget hotels under $100 in Amsterdam, check-in June 20 to June 23"
- "5-star hotels in Dubai with a pool for 2 adults"
- "Boutique hotels in Kyoto with breakfast included"
- "Business hotels near Times Square, NYC, 1 room"

## Environment Variables

| Variable | Description |
|---|---|
| `ORQ_API_KEY` | Orq.ai API key for routing qwen3.6-flash |

## Project Structure

```text
hotel_finder_agent/
├── hotel_finder_agent/
│   ├── __init__.py       # Maps ORQ_API_KEY → OPENAI_API_KEY for LiteLlm
│   └── agent.py          # ADK Agent with LiteLlm + McpToolset
├── app.py                # Streamlit UI (OpenAI SDK + mcp directly)
├── pyproject.toml
├── .env.example
└── assets/
    └── demo.png
```

## How It Works

```
User sends a natural language query
    │
    ▼
Streamlit app opens a Streamable HTTP session to mcp.trivago.com
    │
    ▼
Trivago tool schemas are fetched and converted to OpenAI tool format
    │
    ▼
qwen3.6-flash (via Orq.ai) receives the query and tool definitions
    │
    ▼
Agent calls trivago-search-suggestions to resolve the location
    │
    ▼
Agent calls trivago-accommodation-search with location ID and dates
    │
    ▼
Tool results are returned to the model via the MCP session
    │
    ▼
Model synthesises a final response with hotel options and booking links
```
