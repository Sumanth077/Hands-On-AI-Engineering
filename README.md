<p align="center">
  <a href="https://aiengineering.beehiiv.com/">
    <img src="assets/theaiengineering_logo.jpeg" alt="Hands-On AI Engineering Banner" width="400">
  </a>
</p>
<div align="center">

# 🚀 Hands-On AI Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

A curated collection of practical, production-ready AI projects across multiple modalities, including language models, multimodal models, OCR systems, RAG pipelines, and AI agents. Each project is designed to help you learn, experiment, and build real-world AI applications.

## 📋 Table of Contents

- [🎯 Why This Repository?](#-why-this-repository)
- [🗂️ Project Categories](#️-project-categories)
- [🚀 Getting Started](#-getting-started)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🎯 Why This Repository?

- **Learn by Doing**: Each project includes complete code, setup instructions, and documentation
- **Production-Ready**: Projects follow best practices and are ready to be adapted for real-world use
- **Diverse Use Cases**: From RAG systems to multi-agent workflows and specialized applications
- **Multiple Model Providers**: Projects use OpenAI, Anthropic, Google, and open-source models
- **Active Community**: Regular updates and new project additions

---

## 🗂️ Project Categories

### 🤖 AI Agents

Intelligent ai agents for various automation tasks.

- [**Multi-Agent Financial Analyst**](./ai_agents/multi_agent_financial_analyst) — Team of specialized agents for comprehensive financial analysis  
- [**FinAgent**](./ai_agents/finagent) — Financial assistant agent for stock market analysis and insights
- [**Daily AI News Digest**](./ai_agents/daily-news-digest) — Automated daily digest from 92 Karpathy-curated tech blogs, delivered to Telegram at 8 AM every morning. MiniMax M2.7 scores every article fetched in the last 24 hours and picks the 3 most significant stories.
- [**Agentic Form Filler**](./ai_agents/agentic-form-filler) — Powerful agentic form-filling application using Landing AI for layout parsing and MiniMax M2.7 for multi-turn conversational data gathering.
- [**AI Travel Planning Agent**](./ai_agents/ai_travel_planning_agent) — Multi-agent travel planner that turns a single natural language request into a complete trip plan with flights, hotels, and a day-by-day itinerary.
- [**Competitive Intelligence Agent**](./ai_agents/competitive_intelligence_agent) — Multi-agent AI system that generates strategic sales battlecards by analyzing competitors through the unique lens of your own business context.
- [**Multi-Agent Research Assistant (AG2)**](./ai_agents/multi_agent_research_assistant_ag2) — Production-grade multi-agent research pipeline using AG2 (formerly AutoGen). Three specialists collaborate under GroupChat with LLM-driven speaker selection to research any topic and produce a structured Markdown report.
- [**Self-Reflective Agentic RAG**](./ai_agents/agentic_rag_system) — LangGraph-driven RAG system that grades retrieved context for relevance and sufficiency, rewrites the query if needed, and only generates an answer once the context passes validation — reducing hallucinations through an iterative retrieval loop.
- [**Agentic SQL Search**](./ai_agents/agentic_sql_search) — Natural language to SQL agent powered by Gemma 4. Ask plain-English questions about an e-commerce database and the agent writes, executes, and explains the SQL query — with full reasoning transparency in the Streamlit UI.
- [**Stock Portfolio Analyst**](./ai_agents/stock_portfolio_analyst) — AI-powered portfolio analysis agent built with Agno and DeepSeek-V4-Flash. Takes ticker symbols, share counts, and purchase prices, fetches live market data via YFinance, and generates a streaming report covering P&L, concentration risk, valuation flags, and rebalancing recommendations.
- [**Eagle Eye**](./ai_agents/eagle_eye) — AI-powered GitHub PR review agent using OpenClaw and Telegram. Fetches pull request diffs via GitHub MCP, performs structured code review with severity ratings, and posts feedback to GitHub only after user approval.
- [**CartMate — AI Customer Support Agent**](./ai_agents/ai_customer_support_agent) — Memory-powered e-commerce support agent built with Mem0 and Mistral Small 4. Remembers customers by name, recalls past orders and reported issues across sessions, and picks up conversations exactly where they left off.
- [**Multi-Agent Coding Assistant**](./ai_agents/multi_agent_coding_assistant) — Four-stage coding pipeline powered by Mistral Small 4 and LangChain. A Planner Agent structures the approach, a Coder Agent writes a first draft, a Reviewer Agent critiques it, and the Coder Agent produces a polished final version — all surfaced in an expandable Streamlit UI.
- [**Startup Analyst**](./ai_agents/startup_analyst) — Elite startup due-diligence agent powered by **MiniMax M2.5** via OpenRouter. Give it a company name and URL and it scrapes the site with Firecrawl, crawls multiple pages, and produces an investment-grade report covering market position, financials, team, risks, and strategic recommendations.
- [**Research Team**](./ai_agents/research_team) — Multi-agent research system powered by **MiniMax M2.5** via OpenRouter. Seek searches the web using DuckDuckGo while Scout navigates internal documents. A team leader coordinates both and synthesises findings into a structured report with sourced key findings and open questions.
- [**GitHub Intelligence Agent**](./ai_agents/github_intelligence_agent) — Conversational GitHub research agent powered by **Gemini 3 Flash** and GitHub's official MCP server. Uses Haystack's SearchableToolset to dynamically discover tools from a catalog of 40+ GitHub API endpoints, keeping context lean and avoiding prompt overflow. Ask anything — trending repos, contributor profiles, issue summaries, codebase exploration.
- [**Smolagents Code Agent**](./ai_agents/smolagents_code_agent) — Real-time agentic task runner powered by **Mistral Small 4** and HuggingFace smolagents. The agent writes and executes Python code at each step using DuckDuckGo and Wikipedia, streaming every Think, Act, and Observe step live in a Gradio UI.

### 📸 OCR

Extracting structure and meaning from visual data and documents.

- [**Image-to-Structured-Data Extractor**](./OCR/image_to_structured_data) — High-fidelity visual OCR using Mistral Large 3 and Instructor to convert images into validated, structured JSON.
- [**LaTeX Formula OCR**](./OCR/latex_formula_ocr) — Local vision-language OCR that extracts math formulas from images/PDFs into LaTeX and renders them instantly with KaTeX.
- [**Medical Prescription Digitizer**](./OCR/medical_prescription_digitizer) — Upload a handwritten or printed prescription and get structured, validated output powered by **Mistral Large 3**. Extracted drug names are checked in real time against the RxNorm database with no API key required.


### 🎧 Audio

Projects for audio understanding and analysis.

- [**Music Explorer**](./audio/music_explorer) — Chat with any audio file or YouTube video using Gemini 3 Flash. Ask for transcriptions, lyrics, emotion analysis, instrument identification, and timestamp-aware track breakdowns.

### 🎬 Multimodal

Projects combining vision, video, and language models.

- [**GLM-OCR Pro**](./multimodal/glm_ocr_pro) — High-performance, local-first Streamlit application for structured document extraction using the GLM-OCR model via Ollama to transform images and PDFs into cleanly formatted Markdown in real-time.
- [**Video Understanding Agent**](./multimodal/video_understanding_agent) — Paste a YouTube URL and get an AI-powered chapter summary, key takeaways, and action items powered by Gemini Flash.

### 📚 RAG Applications

Retrieval-Augmented Generation systems for knowledge-enhanced AI applications.

- [**Agentic RAG with O3-Mini & DuckDuckGo**](./rag_apps/agentic_rag_with_o3_mini_and_duckduckgo) — RAG system using O3-Mini model with DuckDuckGo search integration  
- [**Agentic RAG with Qwen & FireCrawl**](./rag_apps/agentic_rag_with_qwen_and_firecrawl) — Advanced RAG using Qwen models and FireCrawl for web scraping  
- [**Vision RAG**](./rag_apps/vision_rag) — Multimodal RAG system capable of processing and querying visual content  
- [**Clinical RAG with ADE**](./rag_apps/clinical_rag_with_ade) — High-precision RAG system using LandingAI ADE for visual-first parsing and Mistral Large for grounded clinical reasoning.
- [**YouTube Transcript RAG**](./rag_apps/youtube_transcript_rag) — Chat with any YouTube video using local Whisper transcription, ChromaDB semantic search, and Mistral Small 4. Answers are grounded in the video content with clickable timestamp links pointing to the exact moment in the video.
- [**GraphRAG Knowledge System**](./rag_apps/graphrag_knowledge_system) — Upload documents and build a local knowledge graph powered by **Mistral Small 4** and NetworkX. Supports two retrieval modes: Local Search for entity-level queries and Global Search for broad thematic synthesis across the entire graph.
- [**Hybrid RAG System**](./rag_apps/hybrid_rag_system) — Dual-indexes documents into both a knowledge graph and a vector store, then runs both retrieval paths in parallel. **Mistral Small 4** answers questions with fused context from graph entities and ranked vector chunks, with full source transparency.

---

## 🤝 Contributing

We welcome contributions! Whether you're adding new projects, improving existing ones, or fixing bugs, your help makes this repository better for everyone.

### How to Contribute

1. **Read the guidelines**: Check [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions
2. **Create an issue**: Propose your project or improvement
3. **Follow the structure**: Use the appropriate category folder
4. **Submit a PR**: One project per pull request

### Project Structure Requirements

- Each project must be in its own folder within the appropriate category
- Must include a comprehensive `README.md` (use our [template](.github/README_TEMPLATE.md))
- Must include `requirements.txt` or `pyproject.toml`
- Must include `.env.example` for required API keys
- Follow snake_case naming convention

---

## 📜 License

This repository is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

Thank you to all contributors who have helped build this collection of AI engineering projects!

---

<div align="center">

**Built with ❤️ by the [AI Engineering Community](https://aiengineering.beehiiv.com/)**

[⬆ Back to Top](#-hands-on-ai-engineering)

</div>
