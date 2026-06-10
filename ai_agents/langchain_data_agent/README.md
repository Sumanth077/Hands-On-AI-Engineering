# LangChain Data Agent

> Query the Chinook SQLite database in plain English through a conversational Streamlit chat.

## Overview

LangChain Data Agent turns natural language questions into safe, read-only SQL against the bundled Chinook music store database. A LangGraph workflow orchestrates schema discovery, query generation, validation, and execution, then returns a clear natural-language answer with an optional Plotly chart when results are tabular or numeric.

This project is useful for data analysts, engineers, and learners who want to explore SQL databases without writing queries by hand, while still seeing the generated SQL for transparency and learning.

## Demo

![Demo](assets/demo.gif)

## Features

- Conversational Streamlit chat with session-based history
- Natural language to SQL via a LangGraph agent and LangChain SQL toolkit
- Read-only query enforcement (SELECT / WITH only; no INSERT, UPDATE, DELETE, or DDL)
- Automatic Plotly charts for numeric and tabular results
- Expandable SQL panel showing the query the agent ran
- Result tables with real column names from the database (e.g. `ArtistName`, `TrackCount`)

## Tech Stack

**Frameworks & Libraries:**

- LangGraph: agent workflow orchestration
- LangChain: SQL toolkit, messages, and LLM integration
- langchain-openai: ChatOpenAI client for the Orq.ai router
- OpenAI SDK: API client (routed through Orq.ai)
- SQLAlchemy: database engine for SQLite
- Pandas: result parsing and chart data prep

**Additional Tools:**

- Database: SQLite (Chinook)
- Charts: Plotly
- Web Framework: Streamlit

## Prerequisites

- Python 3.10 or higher
- API keys for:
  - Orq.ai (`ORQ_API_KEY`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/langchain_data_agent
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

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set your Orq.ai API key:

```env
ORQ_API_KEY=your_orq_api_key_here
```

Optional LangSmith tracing (uncomment in `.env` if needed):

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Example Usage

Ask questions in the chat like:


| Question                                   | What you get                                          |
| ------------------------------------------ | ----------------------------------------------------- |
| How many artists are in the database?      | A count, the SQL used, and a simple result table      |
| Which artist has the most tracks?          | Ranked results with real column names and a bar chart |
| What are the top 5 best-selling genres?    | Aggregated sales by genre with a chart                |
| How many customers are in each country?    | Grouped counts per country                            |
| What is the average track length by genre? | Aggregated duration stats by genre                    |


The agent lists tables, fetches relevant schemas, generates and checks SQL, runs the query read-only, and summarizes the answer. Follow-up questions use conversation history from the session.

## Project Structure

```
langchain-data-agent/
├── app.py              # Streamlit chat UI, charts, and result tables
├── agent.py            # LangGraph SQL agent and read-only query tool
├── chinook.db          # Chinook sample SQLite database
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
├── README.md
└── assets/
    └── demo.gif        # Application demo recording
```

## How It Works

1. **User input:** You type a question in the Streamlit chat.
2. **List tables:** The agent loads available Chinook tables (Artist, Album, Track, Invoice, etc.).
3. **Get schema:** The LLM calls `sql_db_schema` for tables relevant to your question.
4. **Generate SQL:** The model writes a SQLite `SELECT` query from the schema and question.
5. **Check SQL:** A dedicated step reviews the query for common SQL mistakes.
6. **Execute:** `ReadOnlyQuerySQLDatabaseTool` validates and runs the query; results include column names.
7. **Respond:** The agent returns a natural-language answer; the UI shows the SQL, a data table, and a Plotly chart when appropriate.

The LLM (`kimi-k2.6`) is called through the [Orq.ai](https://orq.ai) router (`https://api.orq.ai/v3/router`). Message history is flattened for compatibility with the model’s tool-calling requirements, while the graph still runs the full LangGraph SQL workflow under the hood.