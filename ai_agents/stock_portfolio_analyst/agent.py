"""
Stock Portfolio Analyst — Agno agent powered by DeepSeek-V4-Flash.

Tools:
  • YFinanceTools  — live prices, fundamentals, analyst ratings, company news
  • DuckDuckGoTools — broader market context and recent headlines
  • CalculatorTools — precise P&L and concentration calculations
"""

import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.calculator import CalculatorTools

load_dotenv()

MODEL_ID = "deepseek-ai/deepseek-v4-flash"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

INSTRUCTIONS = """You are an expert financial analyst and portfolio manager.
Your job is to analyze a stock portfolio comprehensively using live market data.

WORKFLOW — follow this order exactly:
1. For EACH holding, use YFinance to fetch:
   - Current stock price
   - Key fundamentals: P/E ratio, P/B ratio, EPS, market cap, sector, industry
   - 52-week high/low range
   - Analyst recommendations summary

2. CALCULATE for each position:
   - Current market value = shares × current price
   - Cost basis = shares × average purchase price
   - Unrealized P&L = current value - cost basis
   - Unrealized P&L % = ((current price - avg cost) / avg cost) × 100
   - Position weight = (current value / total portfolio value) × 100

3. CALCULATE portfolio-level metrics:
   - Total portfolio value
   - Total cost basis
   - Total unrealized P&L and P&L %
   - Sector breakdown and weights

4. Use DuckDuckGo to search for any significant recent news on the holdings.

5. Use Calculator to verify key calculations.

6. IDENTIFY risks:
   - Concentration risk: any position > 20% of total portfolio
   - Valuation risk: P/E > 40 or negative P/E
   - Sentiment risk: recent negative news or analyst downgrades
   - Sector concentration: > 50% in one sector

OUTPUT FORMAT — return a structured markdown report with these sections:

## 📊 Portfolio Summary
A summary table: Total Value, Total Cost, Total P&L, Total P&L%

## 📋 Holdings Analysis
A detailed table with columns:
Ticker | Sector | Shares | Avg Cost | Current Price | Value | P&L | P&L% | Weight%

## 📈 Top Performers & Laggards
Best and worst performing positions with brief commentary.

## ⚠️ Risk Assessment
Bullet list of identified risks with severity (HIGH / MEDIUM / LOW).

## 🔄 Rebalancing Recommendations
Specific, actionable suggestions: what to trim, what to add, what to hold.

## 📰 Recent News Highlights
Key headlines relevant to the portfolio holdings.

Be precise, data-driven, and actionable. Always fetch live data — never estimate prices."""


agent = Agent(
    model=OpenAILike(
        id=MODEL_ID,
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=NVIDIA_BASE_URL,
    ),
    tools=[
        YFinanceTools(
            enable_stock_price=True,
            enable_stock_fundamentals=True,
            enable_key_financial_ratios=True,
            enable_analyst_recommendations=True,
            enable_company_news=True,
            enable_historical_prices=True,
            enable_technical_indicators=True,
        ),
        DuckDuckGoTools(),
        CalculatorTools(),
    ],
    instructions=INSTRUCTIONS,
    markdown=True,
)


def format_prompt(holdings: list[dict], question: str) -> str:
    """Format holdings into a markdown table and append the user's question."""
    rows = "\n".join(
        f"| {h['ticker']} | {h['shares']} | ${h['avg_cost']:.2f} |"
        for h in holdings
    )
    table = f"| Ticker | Shares | Avg Cost (USD) |\n|--------|--------|----------------|\n{rows}"

    question = question.strip() or (
        "Provide a comprehensive portfolio analysis: P&L per position, "
        "portfolio-level metrics, concentration and valuation risks, "
        "and specific rebalancing recommendations."
    )

    return f"Analyze the following stock portfolio:\n\n{table}\n\nQuestion: {question}"


def run_analysis(holdings: list[dict], question: str):
    """Generator — yields streaming text chunks of the analysis report."""
    prompt = format_prompt(holdings, question)
    for chunk in agent.run(prompt, stream=True):
        if chunk.content:
            yield chunk.content
