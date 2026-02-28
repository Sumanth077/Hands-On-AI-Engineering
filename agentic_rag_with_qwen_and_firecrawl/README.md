# Web-Aware Agentic RAG (OpenAI & Firecrawl)

A research-heavy RAG agent designed to bridge the gap between static documents and the live web. It uses professional-grade web crawling to supplement your PDF data.

## Principal Functionalities
- **Advanced Web Crawling**: Integrated with **Firecrawl**, this agent can perform deep-web searches and scrapes to provide context that your uploaded PDFs might be missing.
- **Intelligent Fallback**: The agent prioritizes your document but is "self-aware"—if it detects an information gap, it will autonomously trigger a web crawl to find the missing facts.
- **Concise Research Reporting**: Optimized for speed and clarity, providing direct answers with clear source attribution.
- **Dynamic Indexing**: Real-time processing of PDFs into a local vector database for immediate querying.

## Technical Context
- **Models**: OpenAI `gpt-4o-mini` for stable reasoning and high-speed response.
- **Web Engine**: **Firecrawl** (for high-fidelity web scraping).
- **Vector Engine**: **LanceDB** (Serverless/Local). Runs entirely within the project without background services.

## Setup & Execution

### 1. Requirements
This project utilizes the shared repository environment. Ensure you have installed the root dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment
The tool automatically pulls `OPENAI_API_KEY` and `FIRECRAWL_API_KEY` from the root `.env` file.

### 3. Launch
```bash
streamlit run app.py
```

## How to Use
1. **Upload**: Use the sidebar to load your technical document or research paper.
2. **Index**: The system will automatically build a searchable index using LanceDB.
3. **Analyze**: Ask questions that require both document knowledge and current event context. The agent will show you when it decides to "crawl the web" to find a more complete answer.