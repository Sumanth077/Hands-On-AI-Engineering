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

- [**Multi-Agent Financial Analyst**](./ai_agents/multi_agent_financial_analyst) — Team of specialized agents for comprehensive financial analysis.
- [**FinAgent**](./ai_agents/finagent) — Financial assistant agent for stock market analysis and insights.
- [**Daily AI News Digest**](./ai_agents/daily-news-digest) — Automated daily digest from 92 Karpathy-curated tech blogs delivered to Telegram every morning. MiniMax M2.7 scores articles from the last 24 hours and surfaces the 3 most significant stories.
- [**Agentic Form Filler**](./ai_agents/agentic-form-filler) — Agentic form-filling agent using Landing AI for layout parsing and MiniMax M2.7 for multi-turn data gathering.
- [**AI Travel Planning Agent**](./ai_agents/ai_travel_planning_agent) — Multi-agent travel planner that turns a single natural language request into a complete trip plan with flights, hotels, and a day-by-day itinerary.
- [**Competitive Intelligence Agent**](./ai_agents/competitive_intelligence_agent) — Generates strategic sales battlecards by analyzing competitors through the lens of your own business context.
- [**Multi-Agent Research Assistant (AG2)**](./ai_agents/multi_agent_research_assistant_ag2) — Multi-agent research pipeline using AG2 where three specialists collaborate to research any topic and produce a structured report.
- [**Self-Reflective Agentic RAG**](./ai_agents/agentic_rag_system) — LangGraph RAG system that grades retrieved context, rewrites the query if needed, and generates an answer only once the context passes validation.
- [**Agentic SQL Search**](./ai_agents/agentic_sql_search) — Natural language to SQL agent powered by Gemma 4 that writes, executes, and explains queries against an e-commerce database.
- [**Stock Portfolio Analyst**](./ai_agents/stock_portfolio_analyst) — Portfolio analysis agent built with Agno and DeepSeek-V4-Flash. Fetches live market data via YFinance and generates a report covering P&L, concentration risk, and rebalancing recommendations.
- [**Eagle Eye**](./ai_agents/eagle_eye) — GitHub PR review agent using OpenClaw and Telegram. Fetches diffs via GitHub MCP, performs structured code review with severity ratings, and posts feedback after user approval.
- [**CartMate — AI Customer Support Agent**](./ai_agents/ai_customer_support_agent) — Memory-powered e-commerce support agent built with Mem0 and Mistral Small 4 that remembers customers and picks up conversations where they left off.
- [**Multi-Agent Coding Assistant**](./ai_agents/multi_agent_coding_assistant) — Four-stage coding pipeline powered by Mistral Small 4 and LangChain. A Planner, Coder, and Reviewer agent collaborate to produce a polished final implementation.
- [**Startup Analyst**](./ai_agents/startup_analyst) — Startup due-diligence agent powered by MiniMax M2.5. Scrapes a company's site with Firecrawl and produces an investment-grade report covering market position, financials, team, and risks.
- [**Research Team**](./ai_agents/research_team) — Multi-agent research system powered by MiniMax M2.5. Seek searches the web, Scout navigates internal documents, and a team leader synthesises findings into a structured report.
- [**GitHub Intelligence Agent**](./ai_agents/github_intelligence_agent) — GitHub research agent powered by Gemini 3 Flash and GitHub's official MCP server. Ask anything about repos, contributors, issues, or codebases.
- [**Smolagents Code Agent**](./ai_agents/smolagents_code_agent) — Agentic task runner powered by Mistral Small 4 and HuggingFace smolagents. Writes and executes Python code at each step using DuckDuckGo and Wikipedia.
- [**Agent Discovery Agent**](./ai_agents/agent_discovery_agent) — Searches and compares AI agents across NANDA, MCP, Virtuals Protocol, A2A, and ERC-8004 through a single natural language interface. Powered by Gemini 3 Flash.
- [**Cal Scheduling Agent**](./ai_agents/cal_scheduling_agent) — Conversational scheduling assistant that manages Cal.com appointments through natural language. Book, reschedule, cancel, and check availability with automatic timezone handling.
- [**Hacker News Newsletter Agent**](./ai_agents/hacker_news_newsletter_agent) — Fetches the 10 latest Hacker News stories, scrapes full article content with Trafilatura, generates a structured HTML newsletter with Gemma 4, and delivers it to your inbox via Gmail SMTP.

### 📸 OCR

Extracting structure and meaning from visual data and documents.

- [**Image-to-Structured-Data Extractor**](./OCR/image_to_structured_data) — Converts images into validated, structured JSON using Mistral Large 3 and Instructor.
- [**LaTeX Formula OCR**](./OCR/latex_formula_ocr) — Extracts math formulas from images and PDFs into LaTeX using a local vision-language model.
- [**Medical Prescription Digitizer**](./OCR/medical_prescription_digitizer) — Digitizes handwritten or printed prescriptions into structured output using Mistral Large 3, with real-time drug name validation against RxNorm.


### 🎧 Audio

Projects for audio understanding and analysis.

- [**Music Explorer**](./audio/music_explorer) — Chat with any audio file or YouTube video using Gemini 3 Flash. Ask for transcriptions, emotion analysis, instrument identification, and timestamp-aware breakdowns.

### 🎬 Multimodal

Projects combining vision, video, and language models.

- [**GLM-OCR Pro**](./multimodal/glm_ocr_pro) — Structured document extraction using GLM-OCR via Ollama, transforming images and PDFs into formatted Markdown locally.
- [**Video Understanding Agent**](./multimodal/video_understanding_agent) — Summarizes YouTube videos into chapters, key takeaways, and action items using Gemini Flash.
- [**Multimodal Weather App**](./multimodal/multimodal_weather_app) — Upload a map image and get live weather. Mistral Small 4 identifies the city via vision, then fetches real-time conditions through native tool calling.
- [**Multimodal RAG**](./multimodal/multimodal_rag) — RAG system that ingests text, URLs, PDFs, images, audio, and video into a shared ChromaDB index. Gemini Embedding 2 handles retrieval and Gemini 3 Flash generates grounded answers, passing actual file URIs for media sources.
- [**Image Question Answering**](./multimodal/image_question_answering) — Upload a PDF, select a page, and ask visual questions answered by Gemma 4 with thinking mode. PyMuPDF renders each page to a full-resolution image for grounded reasoning over charts, tables, and figures.

### 📚 RAG Applications

Retrieval-Augmented Generation systems for knowledge-enhanced AI applications.

- [**Agentic RAG with O3-Mini & DuckDuckGo**](./rag_apps/agentic_rag_with_o3_mini_and_duckduckgo) — RAG system using O3-Mini with DuckDuckGo for real-time web search.
- [**Agentic RAG with Qwen & FireCrawl**](./rag_apps/agentic_rag_with_qwen_and_firecrawl) — RAG system using Qwen and FireCrawl for web scraping and retrieval.
- [**Vision RAG**](./rag_apps/vision_rag) — Multimodal RAG system for processing and querying visual content.
- [**Clinical RAG with ADE**](./rag_apps/clinical_rag_with_ade) — High-precision clinical RAG using LandingAI ADE for visual-first document parsing and Mistral Large for grounded reasoning.
- [**YouTube Transcript RAG**](./rag_apps/youtube_transcript_rag) — Chat with any YouTube video using Whisper transcription, ChromaDB retrieval, and Mistral Small 4, with timestamp-linked answers.
- [**GraphRAG Knowledge System**](./rag_apps/graphrag_knowledge_system) — Builds a local knowledge graph from uploaded documents using Mistral Small 4 and NetworkX, supporting both entity-level and thematic queries.
- [**Hybrid RAG System**](./rag_apps/hybrid_rag_system) — Indexes documents into a knowledge graph and a vector store in parallel. Mistral Small 4 answers questions with fused context from both retrieval paths.
- [**HyDE RAG**](./rag_apps/hyde_rag) — RAG pipeline using Hypothetical Document Embeddings. Gemini 3 Flash generates hypothetical answers, Gemini Embedding 2 embeds and averages them, and the result retrieves more precise chunks from ChromaDB.
- [**Rock Music RAG**](./rag_apps/rock_music_rag) — Custom rock music knowledge base built from Wikipedia. Add any band, ask questions across all of them, and get sourced answers powered by BM25 retrieval and Gemma 4.

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
