# Research Team

> Multi-agent research system that combines web search and internal document knowledge into structured reports

![Research Team Demo](assets/demo.gif)

## Overview

Research Team is a two-agent AI system that tackles deep research questions by splitting work between specialists. Seek searches the web for external information while Scout looks through local documents. A team leader powered by **MiniMax M2.5** coordinates both, then synthesises their findings into a clean, structured report.

You give it a question like "Research Anthropic — their products, key people, and recent developments" and it comes back with an executive summary, sourced key findings, and open questions.

## Features

- **Seek agent**: Searches the web using DuckDuckGo and reads pages with WebsiteTools to pull external facts
- **Scout agent**: Reads local files from the `knowledge/` directory to surface internal context
- **Team leader**: Delegates work to each specialist, cross-references findings, and produces a unified report
- **Streaming UI**: Report appears progressively in the Gradio interface as the team works
- **File upload**: Upload `.txt`, `.md`, or `.pdf` files directly from the UI — Scout reads them instantly on the next query

## Tech Stack

**Frameworks and Libraries:**
- [Agno](https://github.com/agno-agi/agno) — agent and team framework
- [Gradio](https://www.gradio.app/) — web UI with streaming support

**Models and APIs:**
- [MiniMax M2.5](https://openrouter.ai/minimax/minimax-m2.5) via [OpenRouter](https://openrouter.ai/) — team leader and both agents
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) — web search for Seek

## Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for:
  - [ ] OpenRouter — get yours at https://openrouter.ai

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/research_team
```

### 2. Install Dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. (Optional) Add Internal Knowledge

Drop any `.txt`, `.md`, or `.pdf` files into the `knowledge/` folder. Scout will automatically have access to them when answering research queries.

## Usage

```bash
uv run python app.py
```

Then open http://localhost:7860 in your browser.

Enter a research query like:

```
Research Anthropic — their products, key people, and recent developments
```

The team will search the web, check local docs, and stream back a structured report.

## Project Structure

```
research_team/
├── agents/
│   ├── __init__.py
│   ├── seek.py       # Web research agent (DuckDuckGo + WebsiteTools)
│   └── scout.py      # Internal document agent (FileTools)
├── knowledge/        # Drop internal docs here for Scout to read
├── assets/           # Demo screenshots
├── models.py         # Shared model configuration
├── team.py           # Team definition and coordination logic
├── app.py            # Gradio UI and streaming logic
├── pyproject.toml    # Dependencies
├── .env.example      # Environment variables template
└── README.md         # This file
```

## How It Works

The team leader receives a research query and breaks it into two tracks. It delegates external research to Seek, who runs DuckDuckGo searches and visits the most relevant pages to pull facts and sources. At the same time it delegates internal research to Scout, who lists and reads files in the `knowledge/` directory.

Once both agents respond, the leader cross-references their findings, flags anything contradictory, and writes a final report with an executive summary, sourced key findings, and open questions.

The Gradio app uses the same stream-buffering technique as the other projects in this repo — output is held until the first markdown heading appears, so the UI never shows raw model narration.

[Back to Top](#research-team)
