# Hybrid RAG System

> Combines knowledge graph retrieval and vector retrieval into a unified pipeline for answering complex questions over your own documents.

## Overview

Hybrid RAG System dual-indexes every uploaded document: a knowledge graph captures named entities and their relationships, while a vector store captures semantic meaning at the chunk level. When a question is asked, both retrieval paths run in parallel and their results are fused into a single context before being passed to Mistral Small 4. The answer is displayed alongside clearly labeled graph sources (entities and relationships) and vector sources (ranked text chunks), so you can see exactly what evidence the model used.

## Demo

![Demo](assets/demo.png)

## Features

- **Dual indexing** — every document is indexed into both a knowledge graph and a vector store in a single ingestion pass
- **Parallel retrieval** — graph and vector queries run concurrently via `ThreadPoolExecutor`, minimising latency
- **Context fusion** — graph entities/relationships and ranked vector chunks are merged into one unified prompt
- **Adjustable retrieval** — sliders to tune top-K vector chunks (1–20) and max graph entities (5–50) at query time
- **Document cache** — SHA-256 hash check skips re-indexing files that are already processed
- **Progress tracking** — two-phase progress bar shows entity extraction and embedding status during ingestion
- **Source transparency** — answer panel shows graph sources and vector sources in separate labelled sections with similarity scores
- **Performance metrics** — per-query timing for graph retrieval, vector retrieval, and generation

## Tech Stack

| Layer | Technology |
|---|---|
| Graph indexing & retrieval | GraphRAG + NetworkX |
| Vector retrieval | LangChain + ChromaDB |
| LLM | Mistral Small 4 (`mistral-small-latest`) via Mistral AI API |
| Embeddings | `nomic-embed-text` via Ollama (local, no API key needed) |
| Graph store | NetworkX — persisted as JSON |
| Vector database | ChromaDB — local persistent storage |
| UI | Streamlit |
| Configuration | python-dotenv |

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed and running
- A [Mistral AI](https://platform.mistral.ai) API key

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/rag_apps/hybrid_rag_system
```

**2. Create a virtual environment**

*Windows*
```bash
python -m venv .venv
.venv\Scripts\activate
```

*macOS / Linux*
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your Mistral API key (see [Environment Variables](#environment-variables)).

**5. Pull the embedding model**

```bash
ollama pull nomic-embed-text
```

## Usage

Ensure Ollama is running, then start the app:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

1. Upload a PDF or TXT file using the sidebar uploader
2. Click **Process Document** — the progress bar tracks entity extraction (Phase 1) then chunk embedding (Phase 2)
3. Type a question in the main panel and click **Ask**
4. Review the answer and expand **Graph Sources** and **Vector Sources** to inspect the retrieved evidence

## Environment Variables

Copy `.env.example` to `.env` and fill in the value below.

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | API key from [platform.mistral.ai](https://platform.mistral.ai) |

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Ollama runs locally and requires no API key.

## Project Structure

```text
hybrid-rag-system/
├── app.py                        # Streamlit application — UI, session state, parallel retrieval
├── config.py                     # Constants and automatic data-directory creation
├── src/
│   ├── document_processor.py     # PDF/TXT loading, text chunking, hash-based document cache
│   ├── graph_indexer.py          # Entity/relationship extraction via Mistral → NetworkX graph → JSON
│   ├── vector_indexer.py         # Ollama embedding function, ChromaDB client, chunk indexing
│   ├── graph_retriever.py        # Query-term matching and BFS subgraph expansion
│   ├── vector_retriever.py       # ChromaDB semantic search with similarity scores
│   ├── context_fusion.py         # Formats and merges graph and vector contexts for the prompt
│   └── mistral_client.py         # Mistral Small 4 answer generation
├── data/
│   ├── graphs/                   # Per-document graphs ({hash}.json) and combined_graph.json
│   ├── chroma/                   # ChromaDB persistent storage
│   └── cache/
│       └── processed_docs.json   # Registry of indexed documents
├── assets/
│   └── demo.png                  # Demo screenshot
├── .env.example                  # Environment variable template
└── requirements.txt              # Python dependencies
```
