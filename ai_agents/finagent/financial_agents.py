"""Multi-agent financial analysis pipeline built on Mistral Small 4 and Yahoo Finance market data."""

import os
import re
import json
import yfinance as yf
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

def _normalize_period(period: str) -> str:
    """Normalizes a user-supplied time period string into a yfinance-compatible period code."""
    if not period:
        return '6mo'
    p = str(period).strip().lower()
    m = re.match(r'^(\d+)(m|mo|y|d|w)$', p)
    if m:
        num, unit = m.group(1), m.group(2)
        if unit == 'm':
            unit = 'mo'
        return f"{num}{unit}"
    return p


def _summarize_stock_data(stock_data) -> str:
    """Builds a compact text summary of real OHLCV price data to ground the analysis prompt."""
    first = stock_data.iloc[0]
    latest = stock_data.iloc[-1]
    start_price = float(first['Close'])
    end_price = float(latest['Close'])
    pct_change = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
    period_high = float(stock_data['High'].max())
    period_low = float(stock_data['Low'].min())
    first_date = stock_data.index[0].strftime('%Y-%m-%d')
    latest_date = stock_data.index[-1].strftime('%Y-%m-%d')

    return (
        f"Data range: {first_date} to {latest_date}\n"
        f"Starting close: {start_price:.2f}\n"
        f"Latest close: {end_price:.2f}\n"
        f"Change over period: {pct_change:.2f}%\n"
        f"Period high: {period_high:.2f}\n"
        f"Period low: {period_low:.2f}\n"
        f"Latest volume: {int(latest['Volume'])}"
    )


class MistralAgent:
    """Wraps a single Mistral Small 4 chat role with its own system prompt."""

    def __init__(self, api_key: str, role: str, system_prompt: str):
        self.client = Mistral(api_key=api_key)
        self.model = 'mistral-small-latest'
        self.role = role
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        full_prompt = f"{self.system_prompt}\n\nUser Request: {prompt}"
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
        )
        return response.choices[0].message.content


class FinancialTools:
    """Provides access to the market data needed by the analysis agents."""

    def get_stock_data(self, symbol: str, period: str = "6mo"):
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)


class FinancialAnalysisTeam:
    """Coordinates the query parser and market analyst agents to produce a stock analysis."""

    def __init__(self, mistral_api_key: str):
        self.tools = FinancialTools()
        self.query_parser = MistralAgent(
            mistral_api_key,
            "Query Parser",
            """You are a financial query parser. Extract from the user query:
- Stock symbol (ticker)
- Analysis type (technical, fundamental, comprehensive)
- Time period (like 1d, 1mo, 6mo, 1y)
Return a JSON object with keys: symbol, analysis_type, time_period."""
        )
        self.market_analyst = MistralAgent(
            mistral_api_key,
            "Market Analyst",
            """You are a senior financial analyst. Provide a clear, professional, and actionable analysis for the stock.
Include market trends, price action, risk assessment and investment recommendations."""
        )

    def parse_query(self, query: str):
        """Parses a natural language query into a symbol, analysis type, and time period."""
        response = self.query_parser.generate(query)
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else {}
        except Exception:
            data = {}

        symbol = data.get("symbol")
        if not symbol:
            symbol_match = re.search(r"\b([A-Z]{1,5})\b", query.upper())
            if not symbol_match:
                raise ValueError("No stock ticker symbol found in query.")
            symbol = symbol_match.group(1)
            data = {"symbol": symbol, "analysis_type": "comprehensive", "time_period": "6mo"}

        return data

    def analyze_market(self, query_info, stock_data) -> str:
        """Asks the market analyst agent for a written analysis grounded in the fetched stock data."""
        data_summary = _summarize_stock_data(stock_data)
        prompt = (
            f"Analyze the stock symbol {query_info['symbol']} for the period "
            f"{query_info.get('time_period', '6mo')} using the following real market data:\n\n"
            f"{data_summary}\n\n"
            f"Base your trends, risk factors, and recommendations only on this data. "
            f"Do not assume any other price information."
        )
        return self.market_analyst.generate(prompt)

    def analyze(self, query: str) -> str:
        """Runs the full parse, fetch, and analyze pipeline for a query and returns the analysis text."""
        query_info = self.parse_query(query)
        period = _normalize_period(query_info.get("time_period", "6mo"))
        stock_data = self.tools.get_stock_data(query_info["symbol"], period)
        if stock_data.empty:
            return f"No data available for symbol {query_info['symbol']} in period {period}."

        analysis = self.analyze_market(query_info, stock_data)
        return analysis


def run_financial_analysis(query: str) -> str:
    """Builds a FinancialAnalysisTeam from the configured Mistral API key and runs the given query."""
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        return "Error: MISTRAL_API_KEY environment variable is not set."
    team = FinancialAnalysisTeam(mistral_api_key)
    try:
        return team.analyze(query)
    except Exception as e:
        return f"Error during analysis: {e}"
