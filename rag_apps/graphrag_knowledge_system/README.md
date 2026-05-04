# GraphRAG Knowledge System

> Upload documents, build a local knowledge graph, and ask complex questions answered by Mistral AI with entity and relationship context.

## Overview

GraphRAG Knowledge System implements Microsoft's GraphRAG methodology as a fully local Streamlit application. Documents are chunked and processed through an entity-extraction pipeline powered by Mistral Small 4. Extracted entities and relationships are stored in a NetworkX graph persisted to disk, while text embeddings are stored in ChromaDB using Ollama's `nomic-embed-text` model. At query time the app supports two retrieval modes — **Local Search** for pinpointing specific entities and **Global Search** for synthesising broad themes across the entire graph — before generating a grounded answer with supporting knowledge graph context.

## Demo

![Demo](assets/demo.png)

## Features

- **Document ingestion** — upload PDF or TXT files; documents are chunked, entity-extracted, and indexed automatically
- **GraphRAG entity extraction** — extracts typed entities (Person, Organization, Location, Technology, Concept, Event) and weighted relationships from every chunk using Mistral
- **Local Knowledge Graph** — NetworkX `DiGraph` persisted as JSON; no external database required
- **Local embeddings** — `nomic-embed-text` via Ollama runs entirely on your machine; no embedding API key needed
- **Local vector store** — ChromaDB stores chunk embeddings with persistent local storage
- **Dual search modes** — Local Search for entity-centric retrieval, Global Search for community-level thematic synthesis
- **Document caching** — SHA-256 file hashing prevents re-indexing the same file on every run
- **Community detection** — automatic Louvain community detection groups related entities and generates community summaries for Global Search

## Tech Stack

| Component | Technology |
|---|---|
| Graph Indexing & Retrieval | Microsoft GraphRAG methodology + NetworkX |
| Document Ingestion | LangChain |
| LLM | Mistral Small 4 (`mistral-small-latest`) via Mistral AI API |
| Embeddings | `nomic-embed-text` via Ollama (local) |
| Graph Store | NetworkX DiGraph — persisted to `graph_store.json` |
| Vector Database | ChromaDB (local persistent storage in `chroma_db/`) |
| UI | Streamlit |

## Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/download) installed and running
- A [Mistral AI API key](https://platform.mistral.ai)

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/rag_apps/graphrag_knowledge_system
```

**2. Create and activate a virtual environment**

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

**4. Pull the embedding model**

```bash
ollama pull nomic-embed-text
```

**5. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your Mistral API key (see [Environment Variables](#environment-variables)).

## Usage

**Start Ollama** (if it is not already running)

```bash
ollama serve
```

**Launch the app**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Workflow**

1. Upload one or more PDF or TXT files using the sidebar file uploader
2. Click **⚙️ Process Documents** — the app chunks each file, extracts entities and relationships, builds the knowledge graph, and indexes embeddings
3. Type a question in the main panel and choose a search mode:
   - **🔍 Local Search** — best for specific questions about named entities, events, or facts
   - **🌐 Global Search** — best for broad thematic questions that span the whole document set
4. Click **🚀 Ask** — the answer appears as a paragraph with supporting entities and relationships listed below

**Example**

*Question:* `What role does transformer architecture play in modern language models?`

*Local Search output*
```text
Transformer architecture is the foundational design behind modern language models such as GPT
and BERT. It introduces a self-attention mechanism that allows the model to weigh the relevance
of every token relative to every other token in a sequence, enabling parallelised training on
large corpora. Positional encodings preserve sequence order without recurrence, while multi-head
attention captures multiple relationship types simultaneously.

Supporting entities: Transformer, Self-Attention, GPT, BERT, Positional Encoding
Key relationships: Transformer → GPT (weight 0.92), Transformer → BERT (weight 0.89)
```

*Global Search output*
```text
Across the document set, transformer-based architectures emerge as the central theme linking
advances in natural language understanding, code generation, and multimodal reasoning.
Community analysis reveals three clusters: foundational architecture research, fine-tuning and
alignment techniques, and real-world deployment optimisations. The transformer's attention
mechanism is the most-referenced concept across all communities, forming the backbone of every
major model discussed.
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the value below.

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Your Mistral AI API key — get one at [platform.mistral.ai](https://platform.mistral.ai) |

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

> `OLLAMA_BASE_URL` defaults to `http://localhost:11434` and does not need to be set unless you run Ollama on a different host or port.

## Project Structure

```text
graphrag-knowledge-system/
├── app.py                      # Streamlit UI, ingestion pipeline, community detection
├── src/
│   ├── cache_manager.py        # SHA-256 file-hash cache (cache.json)
│   ├── document_processor.py   # LangChain PDF/TXT loader and text splitter
│   ├── entity_extractor.py     # GraphRAG-style entity and relationship extraction via Mistral
│   ├── graph_store.py          # NetworkX DiGraph store persisted to graph_store.json
│   ├── llm_client.py           # Mistral AI chat client with retry logic
│   ├── search_engine.py        # Local search and Global search implementations
│   └── vector_store.py         # ChromaDB collection with Ollama embedding function
├── chroma_db/                  # Auto-created — ChromaDB persistent storage
├── graph_store.json            # Auto-created — serialised NetworkX graph
├── cache.json                  # Auto-created — processed file hashes
├── requirements.txt
├── .env.example
└── README.md
```
