# PDF RAG Agent with Milvus, Agno, Qwen and FireCrawl

A Streamlit application demonstrating a Retrieval-Augmented Generation (RAG) workflow using Qwen3:8b, Agno, Milvus, and FireCrawl for crawling through relevant websites.

## Features

- Upload PDFs & process PDF documents.
- Generate BGE embeddings (`bge-small-en-v1.5`).
- Uses Qwen model (via Ollama) for LLM responses.
- Milvus vector store for semantic search.
- Firecrawl API integration for structured and up-to-date web search.
- Interactive Streamlit chat interface.

## Prerequisites

- Python 3.8+
- FireCrawl API Key
- Docker (for Milvus)
- Running Milvus Instance

## Setup

1.  **Clone Repository & Navigate:**
    ```bash
    git clone https://github.com/Sumanth077/awesome-ai-apps-and-agents.git

    cd awesome-ai-apps-and-agents/agentic_rag_with_qwen_and_firecrawl
    ```

2.  **Environment & Dependencies:**
    ```bash
    python3 -m venv venv

    source venv/bin/activate # Or venv\Scripts\activate on Windows

    pip install -r requirements.txt
    ```

3.  **FireCrawl API Key:**
    Set the `FIRECRAWL_API_KEY` environment variable or add it to your Streamlit secrets (`.streamlit/secrets.toml`).

4.  **Start Milvus:**
    Follow the official Milvus guide to start a standalone instance using Docker:
    [https://milvus.io/docs/install_standalone-docker.md](https://milvus.io/docs/install_standalone-docker.md)
    *Ensure it's accessible (usually `http://localhost:19530`).*

## Usage

1.  **Ensure Milvus is running.**
2.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
3.  Open the provided URL (e.g., `http://localhost:8501`).
4.  Upload a PDF, click "Process Document", and start chatting!

## How it Works

- PDF text is chunked and embedded using `bge-small-en-v1.5` from SentenceTransformers.
- Embeddings are stored in Milvus for semantic retrieval.
- User query triggers the Agno Agent:
    - Searches Milvus vector store for relevant context.
    - Falls back to Firecrawl for live web results if context is weak or missing.
    - Uses the locally hosted Qwen model via Ollama to synthesize a final response.

## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details (if one exists). 