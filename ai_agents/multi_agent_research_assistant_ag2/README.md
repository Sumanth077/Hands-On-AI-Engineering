<<<<<<< HEAD
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
=======
<a id="top"></a>

# Multi-Agent Research Assistant with AG2

A production-grade multi-agent research pipeline using [AG2](https://github.com/ag2ai/ag2)
(formerly AutoGen). Three specialists collaborate under GroupChat to research any topic
and produce a structured Markdown report, powered by Mistral Small 4
(`mistral-small-latest`) via AG2's native Mistral integration.

## Features
- Multi-agent collaboration (researcher, analyst, writer) under GroupChat
- AG2's `register_function(caller=, executor=)` tool registration pattern
- Native Mistral support via AG2's `api_type: "mistral"` config entry
- Download report as Markdown

## Demo

![Demo](assets/demo.gif)

## Prerequisites
- Python 3.10+
- Mistral API key from [console.mistral.ai](https://console.mistral.ai/api-keys)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

## Installation
```bash
cd multi_agent_research_assistant_ag2
pip install -r requirements.txt
<<<<<<< HEAD
cp .env.example .env  # add your API key
=======
cp .env.example .env  # add your Mistral API key
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
```

## Usage
```bash
streamlit run research_assistant.py
```

## How It Works

1. **Researcher** searches the web using DuckDuckGo API and summarises findings
2. **Analyst** critically reviews the research and identifies gaps
3. **Writer** synthesises all inputs into a structured Markdown report
<<<<<<< HEAD
4. **GroupChatManager** uses LLM-based speaker selection to orchestrate the conversation

## AG2 Concepts Demonstrated
- `GroupChat` with `speaker_selection_method="auto"`
- `register_function(caller=, executor=)` — separates tool description from execution
- `UserProxyAgent` with `code_execution_config` — built-in code execution sandbox
=======
4. **GroupChatManager** orchestrates the analyst and writer in round-robin order

## AG2 Concepts Demonstrated
- `GroupChat` with `speaker_selection_method="round_robin"`
- `register_function(caller=, executor=)`, which separates tool description from execution
- `UserProxyAgent` with `code_execution_config`, providing a built-in code execution sandbox
- Native Mistral integration via `{"api_type": "mistral", "model": "mistral-small-latest", ...}` in `LLMConfig`, instead of pointing an OpenAI-compatible `base_url` at Mistral's endpoint
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

> **Security note:** `use_docker=False` in `code_execution_config` means any agent-generated
> code runs directly in your process. For production use, set `use_docker=True` or run in
> an isolated environment.

## Running Tests

<<<<<<< HEAD
Tests are fully mocked — no API keys or network access required.
=======
Tests are fully mocked and require no API keys or network access.
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

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
<<<<<<< HEAD
=======

---

[Back to top](#top)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
