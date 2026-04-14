# PDF RAG Agent using Milvus, Agno, OpenAI & DuckDuckGo

A Streamlit application demonstrating a Retrieval-Augmented Generation (RAG) workflow using OpenAI (`gpt-4o-mini`), Agno, Milvus, and DuckDuckGo for web search.

## Features

- Upload & process PDF documents.
- OpenAI embeddings & chat completion (`gpt-4o-mini`).
- Milvus vector store for semantic search.
- DuckDuckGo web search fallback.
- Interactive Streamlit chat interface.

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Docker (for Milvus)
- Running Milvus Instance

## Setup

1.  **Clone Repository & Navigate:**
    ```bash
    git clone https://github.com/Sumanth077/awesome-ai-apps-and-agents.git

    cd awesome-ai-apps-and-agents/agentic_rag_with_o-3-mini_and_duckduckgo
    ```

2.  **Environment & Dependencies:**
    ```bash
    python3 -m venv venv

    source venv/bin/activate # or venv\Scripts\activate on Windows

    pip install -r requirements.txt
    ```
    Ensure ollama is installed, with which you would need to download the corresponding model
    `ollama pull qwen3:1.7b`

3.  **OpenAI API Key:**
    Set the `OPENAI_API_KEY` environment variable or add it to your Streamlit secrets (`.streamlit/secrets.toml`).

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

- PDF text is chunked and embedded using OpenAI.
- Embeddings stored in Milvus.
- User query triggers the Agno Agent:
    - Searches Milvus knowledge base first.
    - Uses DuckDuckGo if knowledge base results are insufficient.
    - `gpt-4o-mini` synthesizes information to generate the final response.

## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details (if one exists). 