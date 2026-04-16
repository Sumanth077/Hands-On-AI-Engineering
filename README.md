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
- [**Daily AI News Digest**](./ai_agents/daily_news_digest) — Automated daily digest from 92 Karpathy-curated tech blogs, delivered to Telegram at 8 AM every morning. MiniMax M2.7 scores every article fetched in the last 24 hours and picks the 3 most significant stories.

### 📸 OCR

Extracting structure and meaning from visual data and documents.

- [**Image-to-Structured-Data Extractor**](./OCR/image_to_structured_data) — High-fidelity visual OCR using Mistral Large 3 and Instructor to convert images into validated, structured JSON.


### 🔧 Openclaw

Projects using the Openclaw framework.

- [**Eagle Eye**](./openclaw/eagle_eye) — AI-powered GitHub PR review agent using OpenClaw  


### 📚 RAG Applications

Retrieval-Augmented Generation systems for knowledge-enhanced AI applications.

- [**Agentic RAG with O3-Mini & DuckDuckGo**](./rag_apps/agentic_rag_with_o3_mini_and_duckduckgo) — RAG system using O3-Mini model with DuckDuckGo search integration  
- [**Agentic RAG with Qwen & FireCrawl**](./rag_apps/agentic_rag_with_qwen_and_firecrawl) — Advanced RAG using Qwen models and FireCrawl for web scraping  
- [**Vision RAG**](./rag_apps/vision_rag) — Multimodal RAG system capable of processing and querying visual content  
- [**Clinical RAG with ADE**](./rag_apps/clinical_rag_with_ade) — High-precision RAG system using LandingAI ADE for visual-first parsing and Mistral Large for grounded clinical reasoning

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
