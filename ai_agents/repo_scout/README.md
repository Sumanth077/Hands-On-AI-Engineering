# RepoScout

> Find and compare open-source projects with OpenAI research decisions and concurrent Apify Server Actor requests.

![RepoScout demo](assets/demo.gif)

## Overview

RepoScout helps developers turn a requirement into an evidence-linked comparison of
open-source repositories. Instead of starting from a fixed list, it discovers public
GitHub repositories, selects plausible candidates, retrieves their documentation,
and assesses each required condition.

Apify **Web Search** discovers projects. Apify **Web Fetch** retrieves repository pages,
READMEs, licence information, documentation, releases, and activity pages through
bounded concurrent **Server Actor** requests. OpenAI plans the research, selects
candidates, assesses retrieved evidence, and explains the shortlist. OpenAI's hosted
web search and URL retrieval tools are not enabled.

The result can include documented matches, exclusions, and unresolved findings.
Every substantive assessment keeps citations back to retrieved source passages.

## Features

- Natural-language requirements, with explicitly requested conditions kept required.
- Agent-driven discovery and candidate selection; unknown details do not prevent inspection.
- Concurrent evidence retrieval with request, model-call, token, and research-round limits.
- Validated structured outputs for planning and assessment through the official OpenAI SDK.
- Comparisons using **Matches**, **Does not match**, and **Needs review**, plus shortlist tradeoffs.
- Licence-scope assessment that distinguishes project code from applicable model or dependency terms.
- Formatted evidence previews, retrieval timestamps, and collapsed unchanged raw sources.
- Saved real sessions, JSON/CSV downloads, and results preserved across ordinary UI reruns.
- Partial-result preservation and explicit stops on authentication, quota, or spending-limit failures.
- A CLI for research and individual retrieval operations.

## Tech Stack

| Layer | Implementation |
|---|---|
| Reasoning | OpenAI Python SDK, Responses API; configured model `gpt-6-luna` |
| Discovery | Apify `apify/web-search` Server Actor |
| Evidence retrieval | Apify `apify/web-fetch` Server Actor; HTTPX asynchronous requests |
| Validation | Pydantic schemas and source-bound citation validation |
| UI | Streamlit |
| Local configuration | `python-dotenv`, environment variables, `.env` |
| Usage protection | Persistent SQLite cost reservations and separate model/retrieval diagnostics |
| Packaging and tests | Python 3.11+, Hatchling, uv/pip, pytest, Ruff |

The optional Gemini adapter is retained for explicit configuration. It is never an
automatic fallback. The default workflow and examples use OpenAI.

## Prerequisites

- Python **3.11 or later**; the application was checked locally with Python 3.12.
- Git and either [uv](https://docs.astral.sh/uv/getting-started/installation/) or pip.
- An OpenAI API key with access and billing/quota for the configured model.
- An Apify API token and sufficient Actor usage credit.
- Network access for new live research. Viewing saved reports needs no API credentials.

API usage may incur charges. A subscription to a chat product does not configure
API access. The application does not upgrade plans or silently change models/providers.

## Installation

### 1. Open the project directory

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/repo_scout
```

Run all remaining commands from this directory. Keep the existing `src/repo_scout/`
package beside `streamlit_app.py`; no nested git repository is needed.

### 2. Install dependencies

With uv (includes test tools):

```bash
uv sync --locked --group dev
```

Or create a virtual environment and install the generated requirements with pip:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install from the project directory:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is exported from `pyproject.toml` and `uv.lock`, with an editable
`-e .` entry that installs the package and the `repo-scout` CLI. Do not maintain a
separate dependency list by hand. After an intentional dependency update, regenerate it:

```bash
uv export --locked --no-dev --no-hashes --output-file requirements.txt
```

### 3. Configure the environment

Create `.env` only if it does not already exist. On macOS/Linux:

```bash
cp -n .env.example .env
```

On Windows PowerShell:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit the local file:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-6-luna
APIFY_TOKEN=your_apify_token
REPOSCOUT_LLM_PROVIDER=openai
REPOSCOUT_ENABLE_LIVE=true
```

Existing environment variables take precedence over `.env`. The example file keeps
live execution disabled until you explicitly enable it. Never commit `.env` or credentials.

## Usage

### Streamlit application

With uv:

```bash
uv run streamlit run streamlit_app.py
```

Or with the activated pip environment:

```bash
python -m streamlit run streamlit_app.py
```

Open the terminal's local URL, normally `http://localhost:8501`.

1. Enter a requirement and choose the candidate limit (default **5**, maximum **20**).
2. Press **Start research** to make live API requests.
3. Follow discovery, retrieval, and assessment progress.
4. Review the comparison, shortlist, and **Project evidence**.
5. Expand evidence and **View raw source**, then download JSON or CSV if needed.

Example request:

> Find self-hosted open-source PDF parsing projects that support table extraction,
> show recent maintenance activity, and use a permissive licence.

Expect a comparison of discovered candidates with criterion verdicts, supporting
source links, explanations, and any remaining review needs. The example does not
promise particular repositories or a nonempty shortlist.

### Saved research sessions

Open **Saved research sessions**, select a report, and press **Open saved session**.
Reports are loaded from `outputs/live-*/result.json`; opening them and downloading
exports makes no API calls, even when live access is disabled. Results retain their
original creation and retrieval timestamps: viewing a saved session is not a fresh
research run. Automated saved-evidence replays have a separate provenance label.
No personal saved reports are included in this project contribution.

**Continue unresolved research** is different: it explicitly starts additional live
work using the saved session's limits and the persistent spending guard. Ordinary
reruns, evidence expansion, and downloads do not launch research.

### CLI

Inspect commands without making API calls:

```bash
uv run repo-scout --help
```

To explicitly authorize live research after configuring `.env`:

```bash
uv run repo-scout --allow-billable research "Find self-hosted PDF parsers with table extraction and a permissive licence" --candidate-limit 5
```

The CLI also provides `search`, `fetch`, `fetch-many`, and `compare`; use `--help` for
their arguments. `compare` makes a separate classic/server retrieval comparison;
normal research uses Server Actors. No benchmark runs automatically.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Unset | Required for new OpenAI research |
| `OPENAI_MODEL` | `gpt-6-luna` | Model ID; currently only this verified configuration is accepted |
| `APIFY_TOKEN` | Unset | Required for new Web Search/Web Fetch requests |
| `APIFY_ACCOUNT_PLAN` | `unknown` | Operator-supplied plan label in diagnostics; does not verify or change an account plan |
| `REPOSCOUT_LLM_PROVIDER` | `openai` | Explicit provider selection; no automatic fallback |
| `REPOSCOUT_ENABLE_LIVE` | `false` | Must be `true` before new API work |
| `REPOSCOUT_MAX_WEB_REQUESTS` | `30` | Combined discovery and fetch request ceiling per research run |
| `REPOSCOUT_MAX_MODEL_CALLS` | `30` | Model generation-call ceiling per run |
| `REPOSCOUT_MAX_ITERATIONS` | `3` | Follow-up research-round ceiling |
| `REPOSCOUT_CONCURRENCY` | `3` | Maximum concurrent independent retrieval requests |
| `REPOSCOUT_MAX_INPUT_CHARS` | `80000` | Structured model input character ceiling |
| `REPOSCOUT_MAX_OUTPUT_CHARS` | `24000` | Model response character ceiling |
| `REPOSCOUT_MAX_OUTPUT_TOKENS` | `4000` | Per-generation maximum output token budget |
| `REPOSCOUT_MAX_SOURCE_CHARS` | `12000` | Selected evidence context per candidate; original retrieved text is retained |
| `REPOSCOUT_RESTORE_RESULT` | Unset | Optional path to a real result inside this project's `outputs/` directory |
| `GEMINI_API_KEY` | Unset | Only needed when explicitly selecting `gemini` |
| `GEMINI_MODEL` | Unset | Optional adapter requires `gemini-3.5-flash-lite` when selected |

The UI and CLI candidate ceiling is chosen at submission. The configured OpenAI
backend also enforces a fixed **$0.25 cumulative local spending guard**, reserving
conservative input/output cost before each generation. It settles known usage and
retains uncertain reservations after transport failures. It does not reset between
sessions. Reaching the cap stops new model work; do not delete the ledger to bypass it.
Other OpenAI model IDs fail closed until their pricing is implemented and verified.
API/model availability, prices, and account quotas can change.

## Project Structure

```text
repo_scout/
├── README.md
├── requirements.txt              # Generated runtime requirements; installs -e .
├── pyproject.toml                # Package/dependency configuration and dev tools
├── uv.lock                       # Locked dependency resolution
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml               # UI theme; Streamlit telemetry disabled
├── streamlit_app.py              # Live research, saved sessions, comparison and evidence UI
├── assets/
│   └── demo.gif
├── src/repo_scout/
│   ├── __init__.py
│   ├── service.py                # Composition and local configuration
│   ├── research.py               # Discovery, selection, planning, assessment and shortlist
│   ├── research_models.py        # Validated decisions, evidence and result schemas
│   ├── apify_client.py           # Server Actors and explicit classic comparison support
│   ├── openai_backend.py         # Responses API and persistent cost reservations
│   ├── gemini.py                 # Optional explicitly selected provider
│   ├── llm.py                    # Structured-output and model-call boundaries
│   ├── models.py                 # Apify input, response and measurement schemas
│   ├── evidence.py               # Original passage indexing and bounded context selection
│   ├── evidence_display.py       # Display-only Markdown cleanup; omits image embeds
│   ├── exports.py                # Comparison rows, JSON and CSV
│   ├── archive.py                # Local reports and sanitized response journals
│   ├── config.py                 # Live-execution gate and Apify settings
│   ├── cli.py                    # repo-scout command
│   ├── reprocess.py              # Saved-evidence utilities
│   ├── demo.py                   # Synthetic transports used only by automated tests
│   └── fixtures/demo.json        # Test fixture, not a user-facing demo mode
└── tests/                        # Offline parser, workflow, budget, evidence and UI checks
```

At runtime, ignored `outputs/` stores reports, retrieved evidence, diagnostics, and
the local cost ledger. It is not part of the contribution. Local environments,
credentials, caches, and recording intermediates also stay outside version control.

## How It Works

1. **Interpret the requirement.** OpenAI produces validated criteria and up to three
   search queries. Requested conditions remain required unless explicitly optional.
2. **Discover projects.** Apify Web Search returns repository URLs, titles, and available
   snippets; null snippets do not automatically exclude a candidate.
3. **Select plausible leads.** OpenAI records reasons to inspect, defer, or exclude.
   Final qualification waits for retrieved evidence.
4. **Retrieve evidence concurrently.** Apify Web Fetch Server Actor requests retrieve
   content and discovered links. The application checks target-page HTTP status,
   preserves failures, and follows only evidence-backed links within the limits.
5. **Assess original passages.** Bounded original passages reach the model in small
   batches. Validated decisions cite source-bound passage IDs; the application resolves
   exact excerpts and keeps required failures separate from missing evidence.
6. **Compare and save.** The app generates a shortlist explanation, stores partial or
   completed results, and serves downloads from that saved result without repeating calls.

Model responses and retrieved web content are treated as untrusted data. Display-only
Markdown repair never changes the source used for citation validation or exports.

## Tests

Install test dependencies with `uv sync --locked --group dev`, then run:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests disable dotenv loading and block real HTTP transports. Synthetic fixtures exercise
parsing, concurrent retrieval, structured decisions, citation boundaries, quota handling,
and UI persistence. They are not substituted for live results and are not accessible
through a fictional sample-data mode in the UI or CLI.

## Official Documentation

- [Apify Web Search](https://apify.com/apify/web-search)
- [Apify Web Fetch](https://apify.com/apify/web-fetch)
- [Apify Server Actors](https://docs.apify.com/actors/running/server-actors)
- [Apify API authentication](https://docs.apify.com/api/v2)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI API pricing](https://openai.com/api/pricing/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

[Back to top](#reposcout)
