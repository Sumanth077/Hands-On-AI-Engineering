# Agent Discovery Agent

> A discovery agent that searches and compares AI agents across multiple registries through a single natural language interface, powered by Gemini 3 Flash and Streamlit.

## Demo

![Demo](assets/demo.gif)

## Overview

<<<<<<< HEAD
Instead of manually browsing five separate agent registries, this agent lets you ask in plain English — "find code review agents", "show me trading bots on Virtuals", "what MCP servers handle databases?" — and get structured results back in seconds.
=======
Instead of manually browsing five separate agent registries, this agent lets you ask in plain English: "find code review agents", "show me trading bots on Virtuals", "what MCP servers handle databases?" It returns structured results in seconds.
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

The agent connects to the [Registry Broker API](https://hol.org/registry/docs), a universal index that abstracts NANDA, MCP, Virtuals Protocol, A2A, and ERC-8004 into a single interface. No registry-specific integrations needed.

## Features

- **Universal Search:** Query AI agents across 5 registries with a single natural language request
<<<<<<< HEAD
- **Agent Details:** Full metadata for any agent — capabilities, endpoints, trust score, and verification status
=======
- **Agent Details:** Full metadata for any agent including capabilities, endpoints, trust score, and verification status
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
- **Similar Agents:** Find alternatives and compare agents with similar functionality
- **Faceted Browsing:** Explore available categories, registries, and capability filters
- **Streamlit Chat UI:** Dark-themed conversational interface with example queries and session history

## Supported Registries

| Registry | Description |
|---|---|
| NANDA | MIT's Network for AI Networked Digital Agents |
| MCP | Model Context Protocol servers |
| Virtuals | Virtuals Protocol on-chain agents |
| A2A | Google's Agent-to-Agent protocol |
| ERC-8004 | Ethereum standard for on-chain agents |

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 3 Flash (`gemini-3-flash-preview`) |
| Agent SDK | [Google GenAI SDK](https://github.com/googleapis/python-genai) (`google-genai`) |
<<<<<<< HEAD
| Registry API | [Registry Broker API](https://hol.org/registry/docs) — public, no key required |
=======
| Registry API | [Registry Broker API](https://hol.org/registry/docs) (public, no key required) |
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
| HTTP client | httpx |
| UI | Streamlit |

## Prerequisites

- Python 3.10 or later
<<<<<<< HEAD
- A Google API key — get one free at [aistudio.google.com](https://aistudio.google.com)
=======
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- A Google API key. Get one free at [aistudio.google.com](https://aistudio.google.com)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

## Installation

**Clone the repository**

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/agent_discovery_agent
```

**Set up environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your Google API key.

<<<<<<< HEAD
=======
**Install dependencies**

```bash
uv sync
```

>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
## Usage

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter your Google API key in the sidebar and start querying.

## Example Queries

```text
Find code review agents
Show me trading bots on Virtuals Protocol
What MCP servers are available for databases?
Find customer support agents
What categories of agents are available?
Find DeFi agents on ERC-8004
Show me A2A compatible agents
Find security and audit agents
```

## Environment Variables

| Variable | Description | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Authenticates Gemini 3 Flash requests | [aistudio.google.com](https://aistudio.google.com) |

## Project Structure

```text
agent_discovery_agent/
├── agent_discovery_agent/     # ADK agent package
│   ├── __init__.py            # Package entry point
│   ├── agent.py               # root_agent definition (for adk web)
│   └── tools.py               # Registry Broker API tools
├── app.py                     # Streamlit UI
├── pyproject.toml             # Project dependencies
├── .env                       # Your API key (git-ignored)
├── .env.example               # Template for .env
└── assets/
    └── demo.png
```

## How It Works

```
User query
    │
    ▼
Gemini 3 Flash decides which tool to call
    │
    ├── search_agents()        → GET /api/v1/search
    ├── get_agent_details()    → GET /api/v1/agents/{uaid}
    ├── get_similar_agents()   → GET /api/v1/agents/{uaid}/similar
    └── get_search_facets()    → GET /api/v1/search/facets
    │
    ▼
Registry Broker API (covers NANDA, MCP, Virtuals, A2A, ERC-8004)
    │
    ▼
Structured results presented in the Streamlit chat UI
```
