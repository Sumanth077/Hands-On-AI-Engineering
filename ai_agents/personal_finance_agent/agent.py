"""LangChain finance agent with pandas and SQLite tools."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "finance.db"
SAMPLE_CSV = PROJECT_ROOT / "transactions.csv"

CATEGORIES = (
    "Food",
    "Transport",
    "Entertainment",
    "Utilities",
    "Shopping",
    "Health",
    "Other",
)

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Food": (
        "restaurant", "grocery", "starbucks", "mcdonald", "chipotle", "pizza",
        "subway", "taco", "burger", "sushi", "thai", "indian", "olive garden",
        "panera", "domino", "dunkin", "whole foods", "trader joe", "safeway",
        "kroger", "costco", "walmart", "uber eats", "doordash",
    ),
    "Transport": (
        "uber", "lyft", "shell", "chevron", "gas station", "gas ", "metro transit",
        "bus pass", "train ticket", "parking", "transit",
    ),
    "Entertainment": (
        "netflix", "spotify", "hulu", "apple music", "steam", "playstation",
        "video game", "amc", "regal cinema", "concert", "bowling", "festival",
        "theater",
    ),
    "Utilities": (
        "electric", "water utility", "internet provider", "phone bill",
        "gas & electric",
    ),
    "Shopping": (
        "amazon", "target", "best buy", "ikea", "home depot", "nike", "zara",
        "h&m", "old navy", "fashion", "clothing", "electronics", "furniture",
    ),
    "Health": (
        "pharmacy", "cvs", "walgreens", "dentist", "doctor", "planet fitness",
        "gym", "yoga", "optometrist", "copay",
    ),
}

INCOME_KEYWORDS = ("salary", "deposit", "payroll", "income", "refund")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "transaction_date": "date",
        "posted_date": "date",
        "memo": "description",
        "narrative": "description",
        "details": "description",
        "merchant": "description",
        "debit": "amount",
        "credit": "amount",
        "transaction_amount": "amount",
        "value": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def _parse_amount(row: pd.Series) -> float:
    if "amount" in row and pd.notna(row["amount"]):
        return float(row["amount"])
    debit = row.get("debit")
    credit = row.get("credit")
    if pd.notna(debit) and float(debit) != 0:
        return -abs(float(debit))
    if pd.notna(credit) and float(credit) != 0:
        return abs(float(credit))
    raise ValueError("Could not determine transaction amount")


def categorize_transaction(description: str, amount: float) -> str:
    text = (description or "").lower()
    if amount > 0 and any(k in text for k in INCOME_KEYWORDS):
        return "Other"
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return category
    return "Other"


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            is_income INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()


def ingest_csv(file_path: str | Path, replace: bool = True) -> dict[str, Any]:
    """Parse CSV, categorize transactions, and persist to SQLite."""
    path = Path(file_path)
    raw = pd.read_csv(path)
    df = _normalize_columns(raw)

    if "date" not in df.columns or "description" not in df.columns:
        raise ValueError(
            "CSV must include date and description columns "
            "(or common aliases like transaction_date, memo)."
        )

    amounts = []
    categories = []
    for _, row in df.iterrows():
        amount = _parse_amount(row)
        desc = str(row["description"])
        amounts.append(amount)
        categories.append(categorize_transaction(desc, amount))

    df["amount"] = amounts
    df["category"] = categories
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["is_income"] = (df["amount"] > 0).astype(int)
    df = df.dropna(subset=["date"])

    with sqlite3.connect(DB_PATH) as conn:
        _init_db(conn)
        if replace:
            conn.execute("DELETE FROM transactions")
        records = df[["date", "description", "amount", "category", "is_income"]].to_dict(
            "records"
        )
        conn.executemany(
            """
            INSERT INTO transactions (date, description, amount, category, is_income)
            VALUES (:date, :description, :amount, :category, :is_income)
            """,
            records,
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    spending = df[df["amount"] < 0]["amount"].sum()
    by_category = (
        df[df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )
    return {
        "rows_imported": int(count),
        "total_spending": float(abs(spending)),
        "category_breakdown": {k: float(v) for k, v in by_category.items()},
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
    }


def load_transactions_df() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=["id", "date", "description", "amount", "category", "is_income"]
        )
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM transactions ORDER BY date", conn)


def _spending_df() -> pd.DataFrame:
    df = load_transactions_df()
    if df.empty:
        return df
    out = df[df["is_income"] == 0].copy()
    out["spend"] = out["amount"].abs()
    out["date"] = pd.to_datetime(out["date"])
    return out


def _format_weekly_budget() -> str:
    df = _spending_df()
    if df.empty:
        return "No transaction data loaded. Upload a CSV first."

    weeks = max(1, (df["date"].max() - df["date"].min()).days // 7 + 1)
    total_spend = df["spend"].sum()
    weekly_avg = total_spend / weeks

    by_cat = df.groupby("category")["spend"].sum().sort_values(ascending=False)
    lines = [
        f"Based on {weeks} week(s) of data, average weekly spending: ${weekly_avg:,.2f}",
        "",
        "Suggested weekly budget by category (proportional to past spending, 10% trim on discretionary):",
    ]
    discretionary = {"Entertainment", "Shopping", "Food"}
    for cat, total in by_cat.items():
        share = total / total_spend if total_spend else 0
        weekly = weekly_avg * share
        if cat in discretionary:
            weekly *= 0.9
        lines.append(f"- {cat}: ${weekly:,.2f}/week")

    top = by_cat.index[0] if len(by_cat) else "N/A"
    lines.extend(
        [
            "",
            f"Largest category: {top} (${by_cat.iloc[0]:,.2f} total).",
            "Tip: cap Entertainment and Shopping first if you want quick savings.",
        ]
    )
    return "\n".join(lines)


def get_llm():
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        raise ValueError("ORQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatOpenAI(
        model=os.getenv("ORQ_MODEL", "deepseek-v4-flash"),
        openai_api_key=api_key,
        openai_api_base="https://api.orq.ai/v3/router",
        temperature=0.2,
    )


def verify_orq_connection() -> str:
    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        return "ORQ_API_KEY missing"
    client = OpenAI(base_url="https://api.orq.ai/v3/router", api_key=api_key)
    model = os.getenv("ORQ_MODEL", "deepseek-v4-flash")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_tokens=10,
    )
    return response.choices[0].message.content or "OK"


@tool
def query_transactions_sql(sql: str) -> str:
    """Run a read-only SQL SELECT against the transactions table in SQLite.
    Table schema: transactions(id, date, description, amount, category, is_income).
    amount is negative for expenses and positive for income. Only SELECT queries allowed."""
    cleaned = sql.strip().rstrip(";")
    if not re.match(r"^\s*select\b", cleaned, re.IGNORECASE):
        return "Error: only SELECT queries are allowed."
    forbidden = re.search(r"\b(insert|update|delete|drop|alter|create|attach)\b", cleaned, re.I)
    if forbidden:
        return "Error: mutating SQL statements are not allowed."
    try:
        with sqlite3.connect(DB_PATH) as conn:
            result = pd.read_sql_query(cleaned, conn)
        if result.empty:
            return "Query returned no rows."
        return result.to_string(index=False)
    except Exception as exc:
        return f"SQL error: {exc}"


@tool
def summarize_spending(period: str = "all") -> str:
    """Summarize spending with pandas. period: 'all', 'last_month', 'last_7_days', or 'this_month'."""
    df = _spending_df()
    if df.empty:
        return "No spending data in the database. Upload a CSV first."

    subset, label = _filter_by_period(df, period)
    if subset.empty:
        return f"No transactions for {label}."

    total = subset["spend"].sum()
    by_cat = subset.groupby("category")["spend"].sum().sort_values(ascending=False)
    lines = [f"Spending summary ({label}): ${total:,.2f} total", ""]
    for cat, val in by_cat.items():
        pct = 100 * val / total if total else 0
        lines.append(f"- {cat}: ${val:,.2f} ({pct:.1f}%)")
    top_merchants = (
        subset.groupby("description")["spend"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    lines.append("\nTop merchants:")
    for desc, val in top_merchants.items():
        lines.append(f"- {desc}: ${val:,.2f}")
    return "\n".join(lines)


def _filter_by_period(df: pd.DataFrame, period: str) -> tuple[pd.DataFrame, str]:
    now = df["date"].max()
    if period == "last_month":
        start = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        end = now.replace(day=1) - timedelta(days=1)
        mask = (df["date"] >= start) & (df["date"] <= end)
        label = f"last month ({start.date()} to {end.date()})"
    elif period == "this_month":
        start = now.replace(day=1)
        mask = df["date"] >= start
        label = f"this month (from {start.date()})"
    elif period == "last_7_days":
        start = now - timedelta(days=7)
        mask = df["date"] >= start
        label = "last 7 days"
    else:
        mask = pd.Series([True] * len(df), index=df.index)
        label = "all time"
    return df.loc[mask], label


@tool
def get_category_spending(category: str, period: str = "all") -> str:
    """Get total spending for one category (Food, Transport, etc.). period same as summarize_spending."""
    category = category.strip().title()
    if category not in CATEGORIES:
        return f"Unknown category. Use one of: {', '.join(CATEGORIES)}"
    df = _spending_df()
    if df.empty:
        return "No spending data in the database. Upload a CSV first."
    subset, label = _filter_by_period(df, period)
    cat_total = subset.loc[subset["category"] == category, "spend"].sum()
    return f"{category} spending ({label}): ${cat_total:,.2f}"


@tool
def suggest_weekly_budget() -> str:
    """Generate a suggested weekly budget plan from historical spending patterns."""
    return _format_weekly_budget()


@tool
def find_largest_discretionary_expenses(limit: int = 5) -> str:
    """Find the largest individual discretionary expenses (Entertainment, Shopping, Food)."""
    df = _spending_df()
    if df.empty:
        return "No data loaded."
    disc = df[df["category"].isin(["Entertainment", "Shopping", "Food"])]
    top = disc.nlargest(limit, "spend")[["date", "description", "category", "spend"]]
    if top.empty:
        return "No discretionary expenses found."
    lines = ["Largest discretionary expenses:"]
    for _, row in top.iterrows():
        lines.append(
            f"- {row['date'].date()} | {row['description']} | "
            f"{row['category']} | ${row['spend']:,.2f}"
        )
    return "\n".join(lines)


TOOLS = [
    query_transactions_sql,
    summarize_spending,
    get_category_spending,
    suggest_weekly_budget,
    find_largest_discretionary_expenses,
]

SYSTEM_PROMPT = """You are a Personal Finance Agent helping users understand their bank transactions.

You have access to categorized transaction data in SQLite (categories: Food, Transport, Entertainment, Utilities, Shopping, Health, Other).
Expenses have negative amounts; income is positive.

Use the provided tools to answer questions with real numbers from the data. When users ask about budgets, call suggest_weekly_budget.
For vague questions like "unnecessary expenses", use find_largest_discretionary_expenses and highlight Entertainment, Shopping, and non-essential Food.

Be concise, use USD formatting, and cite figures from tool results. If no data is loaded, tell the user to upload a CSV.
Today's reference date for relative periods is {today}.
"""


def build_agent_graph():
    from langchain.agents import create_agent

    llm = get_llm()
    today = datetime.now().strftime("%Y-%m-%d")
    return create_agent(
        llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT.format(today=today),
    )


def _extract_reply(messages: list) -> str:
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = msg.content
            return text if isinstance(text, str) else str(text)
    return "I could not generate a response."


def ask_finance_agent(
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
) -> str:
    from langchain_core.messages import AIMessage, HumanMessage

    graph = build_agent_graph()
    messages: list = []
    if chat_history:
        for human, ai in chat_history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=question))

    result = graph.invoke({"messages": messages})
    return _extract_reply(result.get("messages", []))
