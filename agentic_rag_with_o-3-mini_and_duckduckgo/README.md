# Agentic PDF RAG (OpenAI & DuckDuckGo)

A smart document assistant that goes beyond basic search. This agent doesn't just find text—it analyzes your PDFs and proactively searches the web if your documents don't have the answer.

## Principal Functionalities
- **Agentic Knowledge Retrieval**: The agent is programmed to prioritize your local PDF knowledge base but will automatically pivot to **DuckDuckGo** for real-time web context when needed.
- **Deep Document Analysis**: Specifically designed to interpret complex PDF structures, providing citations including page numbers and section headers.
- **Conversational Memory**: Maintains the context of your discussion, allowing for multi-turn questions about specific data points.
- **Integrated Web Search**: Uses the Agno framework to coordinate between local vector data and live web results for a more comprehensive answer.

## Technical Context
- **Models**: OpenAI `o3-mini` (Reasoning) and OpenAI Embeddings.
- **Vector Engine**: **LanceDB** (Serverless/Local). This project uses a local-first database that runs entirely in memory or on disk without requiring an external server setup.
- **Backend**: Agno (formerly Phidata) for agent orchestration.

## Setup & Execution

### 1. Requirements
This project shares the root repository's `requirements.txt` and virtual environment. Ensure you have run:
```bash
pip install -r requirements.txt
```

### 2. Environment
The application loads your `OPENAI_API_KEY` directly from the root `.env` file for a "plug-and-play" experience.

### 3. Launch
```bash
streamlit run app.py
```

## How to Use
1. **Upload**: Drag and drop any PDF into the sidebar.
2. **Profile**: Click "Process Document" to build the local search index.
3. **Query**: Ask any question. Watch the sidebar to see the agent "thinking" and choosing whether to use the PDF or the web for its answer.