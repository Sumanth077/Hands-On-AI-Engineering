<a id="top"></a>
# RAG Agent with Database Routing

> A RAG system that intelligently routes queries across three specialized databases covering product information, customer support, and financial data. When no relevant documents are found, a fallback agent performs live web research to still return a useful answer.

## Demo

![Demo](assets/demo.gif)

## Overview

Each user query is classified by a structured-output router (Orq.ai + GLM-5 Turbo) and sent to the most relevant Qdrant collection. The top matching documents are retrieved and a grounded answer is generated via Orq.ai. If nothing clears the relevance threshold, the fallback searches the web with DuckDuckGo and answers via Orq.ai. Every response shows which database was used and exposes the routing reasoning and source documents.

## Features

- Three specialized Qdrant databases: products, customer support, and financial data
- LangChain structured-output router for deterministic database selection
- Score-threshold relevance check: falls back to web search when no documents qualify
- LangGraph ReAct fallback agent with DuckDuckGo search
- Full routing transparency: every answer shows the routing decision and reasoning
- Source documents expandable per response with similarity scores

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GLM-5 Turbo (`zai/glm-5-turbo`) via Orq.ai |
| Embeddings | Gemini Embedding 001 (`google-ai/gemini-embedding-001`) via Orq.ai |
| Router | OpenAI Responses API with JSON structured output |
| Fallback | DuckDuckGo search + Orq.ai generation |
| Vector Store | Qdrant (in-memory) |
| UI | Streamlit |

## Prerequisites

- Python 3.10 or later
- An Orq.ai API key at [orq.ai](https://orq.ai)

## Installation

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/rag_apps/rag_agent_with_database_routing
cp .env.example .env
```

Add your ORQ_API_KEY to `.env`.

## Usage

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`. The three Qdrant collections are seeded with sample data on first run (takes a few seconds).

## Environment Variables

| Variable | Description |
|---|---|
| `ORQ_API_KEY` | Orq.ai key for GLM-5 Turbo and Gemini Embedding 2 |

## Project Structure

```text
rag_agent_with_database_routing/
├── rag_agent/
│   ├── __init__.py
│   ├── databases.py   # Qdrant setup, empty collections, add_documents helper
│   ├── router.py      # Orq.ai router with JSON structured RoutingDecision output
│   ├── retriever.py   # Qdrant similarity search with score threshold
│   ├── fallback.py    # LangGraph ReAct agent with DuckDuckGo
│   └── pipeline.py    # Main orchestration: route, retrieve, generate or fallback
├── app.py             # Streamlit UI
├── pyproject.toml
├── .env.example
└── assets/
    └── demo.gif
```

## How It Works

```
User query
    │
    ▼
Agno router classifies query into: products / support / financial
    │
    ▼
Qdrant retrieves top-k documents from the selected collection
    │
    ├── Documents found (score above threshold)
    │       │
    │       ▼
    │   LangChain generates a grounded answer with citations
    │
    └── No relevant documents found
            │
            ▼
        LangGraph ReAct agent searches the web with DuckDuckGo
            │
            ▼
        Returns a web-sourced answer
    │
    ▼
Streamlit displays the answer with routing badge, source docs, and reasoning
```

[Back to top](#top)
