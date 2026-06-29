# Browser Automation Agent

> An AI agent that takes a natural language instruction and autonomously navigates the web to complete it.

## Overview

Browser Automation Agent is a Gradio app that turns plain-English instructions into real browser actions. Describe a task such as _"Find the latest news about AI agents"_ and the agent plans a sequence of steps, drives a real Chromium browser to carry them out, and returns a structured summary of what it found. It is powered by `browser-use` for autonomous browsing and an LLM served through the Orq.ai AI Router.

## Demo

![Demo](assets/demo.gif)

## Features

- **Natural language tasks:** describe what you want in plain English; no scripting required.
- **Autonomous browsing:** the agent plans and executes real browser actions (navigate, click, read, extract) step by step.
- **Live progress updates:** streams step-by-step progress to the chat as the agent browses, so you can follow along in real time.
- **Structured results:** returns a clear summary plus the pages visited and a collapsible action log.
- **Resilient error handling:** surfaces provider/model errors in the UI and tolerates transient step retries.
- **Configurable model:** point at any Orq.ai-supported model via a single environment variable.
- **Clean chat UI:** a Gradio chat interface with ready-to-run example prompts.

## Tech Stack

Frameworks & Libraries:

- [browser-use](https://github.com/browser-use/browser-use) — autonomous browser agent
- [Playwright](https://playwright.dev/python/) — browser control / Chromium driver
- [LangChain](https://www.langchain.com/) — LLM orchestration
- [Gradio](https://www.gradio.app/) — chat web interface
- [python-dotenv](https://pypi.org/project/python-dotenv/) — environment variable loading

Additional Tools:

- Browser Automation: browser-use
- Web Framework: Gradio

## Prerequisites

- Python 3.10 or higher
- API keys for:
  - Orq.ai (`ORQ_API_KEY`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/browser_automation_agent
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

Then install the Chromium browser that `browser-use` drives:

```bash
python -m playwright install chromium
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set your Orq.ai API key:

```env
ORQ_API_KEY=your_orq_api_key_here
```

Optionally override the model (uses Orq.ai's `provider/model` format):

```env
ORQ_MODEL=alibaba/qwen3.6-flash
```

## Usage

### Running the Application

```bash
gradio app.py
```

Then open the local URL printed in the terminal (usually `http://127.0.0.1:7860`) and start giving the agent tasks.

### Example Instructions

| Instruction | What the Agent Returns |
| --- | --- |
| "Find the latest news about AI agents and summarize the top 3 stories." | A short summary of the three most relevant recent articles, plus the pages it visited. |
| "What is the current weather in Tokyo?" | The current temperature and conditions pulled from a weather source. |
| "Find the top post on Hacker News right now and summarize it." | The title, link, and a brief summary of the current #1 Hacker News post. |
| "Look up the price of the latest iPhone on Apple's website." | The model name and current price found on apple.com. |
| "What is the latest model released by Anthropic?" | The most recent model name, release date, and a one-line description from Anthropic's newsroom. |

## Project Structure

```
browser_automation_agent/
├── app.py            # Gradio UI and agent integration
├── agent.py          # browser-use agent and LLM (Orq.ai) setup
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .env              # Your local secrets (git-ignored)
├── .gitignore
├── README.md
└── assets/           # Demo GIF
```

## How It Works

1. **User input**: The user types a natural language instruction into the Gradio chat interface (`app.py`).
2. **LLM setup**: `agent.py` builds an OpenAI-compatible chat client pointed at the Orq.ai AI Router (`https://api.orq.ai/v3/router`) using the configured model (e.g. `alibaba/qwen3.6-flash`).
3. **Agent planning**: The instruction is handed to a `browser-use` `Agent`, which uses the LLM to reason about the goal and plan a sequence of browser actions.
4. **Browser execution**: `browser-use` drives a real Chromium browser via Playwright, executing each action step by step (navigating, clicking, reading, and extracting content) while observing the page state after every step.
5. **Result extraction**: When the task is done, the agent's history is parsed into a structured result containing the final summary, the URLs visited, and an action log. Provider/model errors are surfaced if the task fails.
6. **Response**: The structured result is rendered back into the Gradio chat as a readable markdown response.
