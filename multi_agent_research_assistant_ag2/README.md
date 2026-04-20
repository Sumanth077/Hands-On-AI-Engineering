# Multi-Agent Research Assistant with AG2

A production-grade multi-agent research pipeline using [AG2](https://github.com/ag2ai/ag2)
(formerly AutoGen). Three specialists collaborate under GroupChat with LLM-driven speaker
selection to research any topic and produce a structured Markdown report.

## Features
- Multi-agent collaboration (researcher, analyst, writer) under GroupChat
- LLM-driven dynamic speaker selection — no hardcoded turn order
- AG2's `register_function(caller=, executor=)` tool registration pattern
- OpenAI-compatible endpoint (works with SambaNova, Azure OpenAI, local models via Ollama)
- Download report as Markdown

## Prerequisites
- Python 3.10+
- OpenAI API key (or compatible endpoint)

## Installation
```bash
cd multi_agent_research_assistant_ag2
pip install -r requirements.txt
cp .env.example .env  # add your API key
```

## Usage
```bash
streamlit run research_assistant.py
```

## How It Works

1. **Researcher** searches the web using DuckDuckGo API and summarises findings
2. **Analyst** critically reviews the research and identifies gaps
3. **Writer** synthesises all inputs into a structured Markdown report
4. **GroupChatManager** uses LLM-based speaker selection to orchestrate the conversation

## AG2 Concepts Demonstrated
- `GroupChat` with `speaker_selection_method="auto"`
- `register_function(caller=, executor=)` — separates tool description from execution
- `UserProxyAgent` with `code_execution_config` — built-in code execution sandbox

> **Security note:** `use_docker=False` in `code_execution_config` means any agent-generated
> code runs directly in your process. For production use, set `use_docker=True` or run in
> an isolated environment.

## Running Tests

Tests are fully mocked — no API keys or network access required.

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
multi_agent_research_assistant_ag2/
├── research_assistant.py   # Agents, GroupChat orchestration, Streamlit UI
├── tools/
│   └── research_tools.py   # web_search (DuckDuckGo API), fetch_page_content
├── tests/
│   ├── conftest.py          # sys.path setup, Streamlit stub
│   ├── test_agent_setup.py  # Agent instantiation, tool registration
│   └── test_research_tools.py  # Mocked tool tests, SSRF validation
├── requirements.txt
└── .env.example
```
