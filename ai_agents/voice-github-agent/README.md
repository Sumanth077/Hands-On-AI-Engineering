# Voice GitHub Agent

> Speak an instruction, and a tool-calling agent inspects your repo and acts on it.

![Voice GitHub Agent Demo](assets/demo.gif)

## Overview

Voice GitHub Agent turns a spoken instruction into real GitHub actions, so you can check in on a repository without switching to a keyboard. You speak a request, it gets transcribed, and a tool-calling agent decides which GitHub tools to run, from reading commits and diffs to opening issues and posting comments, then reports back what it found and did. It is built for developers who want a fast, hands-free way to triage a repo instead of clicking through commits and issues one at a time.

## Features

- **Real voice input**: click a record button in the browser, speak, click again to stop, no terminal and no typing the instruction
- **Sync transcription**: AssemblyAI's Sync API returns a finished transcript in the same HTTP call, no polling
- **A genuine tool-calling agent**: `qwen3-next-80b-a3b` on AssemblyAI's LLM Gateway decides which GitHub tools to call, not a scripted if/else
- **Real GitHub actions**: reads commits, diffs, and issues, and can open issues and post comments through the GitHub REST API
- **Turn-limited agent loop**: capped at 8 tool-calling turns so a confused run fails loud instead of looping forever
- **Formatted markdown output**: the summary and activity feed render through marked.js, so bold text, inline code, and numbered lists show up formatted instead of as raw markdown syntax

## Tech Stack

**Frameworks & Libraries:**
- Flask
- openai (Chat Completions client, pointed at AssemblyAI's LLM Gateway)
- marked.js (markdown rendering in the browser)

**Additional Tools:**
- AssemblyAI Sync API, transcription (Universal-3.5 Pro)
- AssemblyAI LLM Gateway, agent model (qwen3-next-80b-a3b)
- GitHub REST API, commits, diffs, issues, comments
- Browser MediaRecorder / Web Audio API, mic capture and WAV encoding, no server-side audio libraries needed

## Prerequisites

- Python 3.9 or higher
- API keys for:
  - [ ] AssemblyAI (free signup: assemblyai.com/dashboard/signup)
  - [ ] GitHub personal access token (repo scope, or fine-grained with Contents read + Issues read/write)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/voice-github-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```bash
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
GITHUB_TOKEN=your_github_token_here
GITHUB_REPO=owner/name
```

`GITHUB_REPO` is the single repository the agent operates on, as `owner/name`, for example `octocat/hello-world`. Never commit `.env` to version control.

## Usage

### Running the Application

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

### Example Usage

**Input:** (spoken) "Check the latest commits and tell me what changed. If anything looks broken, open an issue for it."

**Output:** The transcript appears in the UI styled like text landing in a notes app, followed by an activity feed listing each GitHub action taken in plain language (for example "Checked the last 5 commits" and "Opened issue #42"), followed by a final written summary of what was checked, found, and done.

## Project Structure

```
voice-github-agent/
├── app.py                  # Flask app: serves the UI and the /api/run endpoint
├── transcribe.py           # AssemblyAI Sync API wrapper
├── agent.py                # LLM Gateway client and tool-calling agent loop
├── github_tools.py         # GitHub REST API tool functions and their schemas
├── templates/
│   └── index.html          # Single-page UI
├── static/
│   ├── app.js               # Recording, client-side WAV encoding, and rendering the result
│   └── style.css            # Visual styling
├── prompts/
│   └── system_prompt.txt    # System prompt for the agent
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## How It Works

**Technical Details:**

The browser records the microphone with `MediaRecorder`, decodes the result with the Web Audio API, and encodes it as a 16-bit PCM mono WAV entirely client-side in `static/app.js`, since AssemblyAI's Sync API requires WAV and this way no server-side audio library is needed. That WAV is POSTed to `/api/run` in `app.py`.

`transcribe.py` sends the WAV to AssemblyAI's Sync API in a single HTTP request and gets a finished transcript back in the same response, with no job to poll. `app.py` hands that transcript to `agent.py`, which runs the tool-calling loop: it sends the transcript, the system prompt from `prompts/system_prompt.txt`, and the GitHub tool schemas to the LLM Gateway on `qwen3-next-80b-a3b`. Whenever the model returns tool calls instead of a final answer, the loop executes them for real against `github_tools.py`, feeds the results back as tool messages, and asks again, up to 8 turns. `run_agent()` returns both the final summary and the full list of tool calls made along the way.

`app.py` returns `{ transcript, tool_calls, summary }` as JSON. `static/app.js` renders the transcript first, then a plain-language activity feed built from the tool calls, then the summary. The summary and activity feed text are escaped before being parsed as markdown with marked.js, so formatting like bold text and numbered lists renders correctly while any injected HTML in the model's output is neutralized instead of executed.

[⬆ Back to Top](#voice-github-agent)
