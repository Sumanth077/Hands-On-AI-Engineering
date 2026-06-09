# Personal Finance Agent

> Upload a bank statement or transaction CSV, categorize your spending, and ask natural language questions about your finances.

## Overview

Personal Finance Agent is a Streamlit application that ingests transaction CSVs, classifies each row into spending categories with pandas, and stores the results in SQLite. Users ask questions in plain English and receive answers from a LangChain tool-calling agent backed by Orq.ai. The agent can summarize spending by category, surface large discretionary purchases, and suggest a weekly budget plan based on historical patterns.

## Demo

![Demo](assets/demo.gif)

## Features

- CSV upload for bank exports with flexible column names (date, description, amount, or debit/credit)
- Automatic categorization into Food, Transport, Entertainment, Utilities, Shopping, Health, and Other
- SQLite persistence for fast queries and agent tool access
- Natural language Q&A via a LangChain agent with pandas summaries and read-only SQL tools
- Weekly budget suggestions derived from your spending patterns
- Bundled sample data (`transactions.csv`) so you can try the app without uploading a bank statement
- Orq.ai model routing through the OpenAI SDK with `deepseek-v4-flash` by default

## Tech Stack

Frameworks & Libraries:

- [LangChain](https://www.langchain.com/) for agent orchestration and tools
- [LangGraph](https://langchain-ai.github.io/langgraph/) for the agent execution graph
- [LangChain OpenAI](https://python.langchain.com/docs/integrations/chat/openai/) for chat models via Orq.ai
- [OpenAI Python SDK](https://github.com/openai/openai-python) (Orq.ai-compatible client)
- [python-dotenv](https://github.com/theskumar/python-dotenv) for environment configuration

Additional Tools:

- Database: SQLite
- Data Processing: pandas
- Web Framework: Streamlit

## Prerequisites

- Python 3.10 or higher
- API keys for:
  - Orq.ai (`ORQ_API_KEY`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/personal_finance_agent
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
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
ORQ_API_KEY=your-orq-api-key-here
```

On Windows Command Prompt, use `copy .env.example .env` instead of `cp`.

## Usage

### Running the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`). In the sidebar, click **Load sample transactions.csv** or upload your own CSV, then ask questions in the chat or use the sample prompt buttons.

### Example Usage

| Question | What the agent returns |
| --- | --- |
| How much did I spend on food last month? | Total Food spending for the prior calendar month with a formatted dollar amount |
| What's my biggest unnecessary expense? | The largest discretionary charges across Entertainment, Shopping, and non-essential Food |
| Break down my spending by category. | Per-category totals and percentage share of overall spending |
| Suggest a weekly budget plan based on my habits. | Proportional weekly budget caps by category with a trim on discretionary spending |
| How much did I spend on transport in the last 7 days? | Transport total for the most recent seven days in the loaded data |

## Project Structure

```
personal_finance_agent/
├── app.py              # Streamlit UI with CSV upload and chat interface
├── agent.py            # LangChain agent, categorization logic, and SQLite tools
├── transactions.csv    # Sample transaction data for immediate testing
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
├── README.md
└── assets/
    └── demo.gif        # Demo recording for the README
```

## How It Works

1. **CSV upload**  
   The user uploads a bank export or loads the bundled `transactions.csv` from the sidebar in `app.py`.

2. **Parse and normalize**  
   `ingest_csv()` in `agent.py` reads the file with pandas, normalizes column names, and parses amounts. The importer supports a single `amount` column or separate `debit` and `credit` columns.

3. **Categorize**  
   Each transaction description is matched against keyword rules and assigned to Food, Transport, Entertainment, Utilities, Shopping, Health, or Other. Income rows (positive amounts with salary or deposit keywords) are stored but excluded from spending summaries.

4. **SQLite storage**  
   Categorized rows are written to `finance.db` in a `transactions` table. Each new import replaces the previous dataset.

5. **Natural language Q&A**  
   The user asks a question in the Streamlit chat. `ask_finance_agent()` runs a LangChain agent with an OpenAI-compatible client pointed at `https://api.orq.ai/v3/router` using the `deepseek-v4-flash` model.

6. **Tool use**  
   The agent selects tools to answer with real numbers from the database: read-only SQL (`query_transactions_sql`), pandas summaries (`summarize_spending`, `get_category_spending`), discretionary analysis (`find_largest_discretionary_expenses`), and budget planning (`suggest_weekly_budget`).

7. **Response**  
   The agent formats tool results into a plain-English reply in the chat. Prior conversation turns are passed on follow-up questions for context.

**Data flow:**

```
CSV upload -> pandas parse and categorize -> SQLite -> LangChain agent and tools -> chat answer
```
