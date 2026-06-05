# Reasoning RAG

> Ask questions against any web source and get cited answers with a live, step-by-step reasoning trace.

## Overview

Reasoning RAG turns any URL into a queryable knowledge base and answers your questions with full transparency. It fetches and embeds the page into a vector store, retrieves the most relevant passages for each question, and reasons over them step by step. The Gradio interface shows the model's reasoning unfold live on the left while the grounded, citation-backed answer builds on the right.

## Demo

![Demo](assets/demo.gif)

## Features

- **One-paste ingestion**: drop in a URL and the app fetches, cleans, chunks, embeds, and indexes the content automatically.
- **Semantic retrieval**: finds the most relevant passages for every question using dense embeddings and cosine similarity.
- **Two-agent pipeline**: a Retriever agent sharpens the search query and a Reasoning agent works through the evidence.
- **Live reasoning trace**: watch the model think step by step as tokens stream in, side by side with the answer.
- **Grounded citations**: every answer is tied to numbered source passages with relevance scores and links back to the origin.
- **Persistent knowledge base**: ingested sources are stored in ChromaDB and remain available across sessions.
- **Configurable model**: swap the underlying LLM with a single environment variable, no code changes required.

## Tech Stack

Frameworks & Libraries:
- [Agno](https://github.com/agno-agi/agno): agent orchestration for the Retriever and Reasoning agents
- [OpenAI Python SDK](https://github.com/openai/openai-python): LLM client pointed at the Orq.ai router
- [mistralai](https://github.com/mistralai/client-python): Mistral Embed embeddings
- [chromadb](https://github.com/chroma-core/chroma): persistent vector storage and similarity search
- [requests](https://github.com/psf/requests) + [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/): URL fetching and HTML cleaning
- [python-dotenv](https://github.com/theskumar/python-dotenv): environment configuration

Additional Tools:
- Vector Database: ChromaDB
- Embeddings: Mistral Embed (`mistral-embed`)
- Web Framework: Gradio

## Prerequisites

- Python 3.10 or higher
- API keys for:
  - Orq.ai (ORQ_API_KEY)
  - Mistral (MISTRAL_API_KEY)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/reasoning-rag
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```env
ORQ_API_KEY=your_orq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
# Optional. Defaults to alibaba/qwen3.6-flash. Must be a provider/model slug.
ORQ_MODEL=alibaba/qwen3.6-flash
```

## Usage

### Running the Application

Start the app from the project directory:

```bash
python app.py
```

Gradio prints a local URL (by default `http://127.0.0.1:7860`). Open it in your browser, add one or more source URLs, then ask questions. The reasoning trace streams on the left and the cited answer appears on the right.

### Example Usage

| Step | Action | Result |
| ---- | ------ | ------ |
| 1 | Paste `https://en.wikipedia.org/wiki/Retrieval-augmented_generation` and click **Add source** | The page is fetched, chunked, embedded, and indexed; a confirmation shows the title and chunk count |
| 2 | Type *"What problem does retrieval-augmented generation solve?"* and click **Ask** | The Retriever pulls the top passages and the Reasoning agent begins streaming |
| 3 | Read the left panel | A step-by-step reasoning trace referencing sources as `[1]`, `[2]`, etc. |
| 4 | Read the right panel | A concise answer with inline citations and a list of source snippets |

### What the Agent Returns

For every question the app produces three coordinated outputs:

- **Reasoning trace**: the agent's step-by-step thinking: which sources are relevant, what each one says, and how they combine to form an answer.
- **Grounded answer**: a concise final response that draws only on the retrieved sources, with inline citations such as `[1]` and `[2]`. If the sources don't contain the answer, the agent says so instead of guessing.
- **Citations**: a numbered list of the exact passages used, each with its source title, link, relevance score, and a snippet.

## Project Structure

```
reasoning-rag/
├── app.py            # Gradio UI with two parallel panels (reasoning trace + cited answer)
├── rag.py            # URL ingestion, chunking, embedding, and ChromaDB storage/retrieval
├── agents.py         # Agno Retriever and Reasoning agents (OpenAI SDK -> Orq.ai router)
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .gitignore        # Ignores .env, .venv/, __pycache__/, and the local vector store
├── README.md         # This file
└── assets/           # Demo GIF and other media
```

## How It Works

1. **Ingestion** (`rag.py`): When you add a URL, the app fetches the page with `requests`, strips non-content elements (scripts, nav, footers) with BeautifulSoup, and normalizes the remaining text.
2. **Chunking**: The cleaned text is split into overlapping word windows (~220 words with ~40-word overlap) so that context is preserved across chunk boundaries.
3. **Embedding**: Each chunk is converted into a dense vector using Mistral Embed (`mistral-embed`), batched for efficiency.
4. **Storage**: Chunks, their embeddings, and metadata (source URL, title, chunk index) are written to a persistent ChromaDB collection configured for cosine similarity.
5. **Retrieval** (`agents.py`): When you ask a question, the **Retriever agent** rewrites it into a focused search query (expanding entities and synonyms). That query is embedded and used to pull the top-k most semantically relevant chunks from ChromaDB.
6. **Reasoning**: The retrieved passages are formatted into a numbered SOURCES block and handed to the **Reasoning agent**, which is instructed to think step by step and then write a grounded answer. It responds in two sections, `### REASONING` and `### ANSWER`, citing sources as `[1]`, `[2]`, etc.
7. **Streaming to the UI** (`app.py`): As the Reasoning agent streams tokens, the app splits the output at the `### ANSWER` marker and routes the parts to the two panels: reasoning on the left, the final cited answer on the right. The citation list is rendered from the exact chunks that were retrieved, so every claim is traceable back to its source.

