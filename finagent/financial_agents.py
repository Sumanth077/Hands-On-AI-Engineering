import os
import re
import json
import yfinance as yf
import google.generativeai as genai


def _normalize_period(period: str) -> str:
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


class GeminiAgent:
    def __init__(self, api_key: str, role: str, system_prompt: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.role = role
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        full_prompt = f"{self.system_prompt}\n\nUser Request: {prompt}"
        response = self.model.generate_content(full_prompt)
        return response.text


class FinancialTools:
    def get_stock_data(self, symbol: str, period: str = "6mo"):
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)


class FinancialAnalysisTeam:
    def __init__(self, gemini_api_key: str):
        self.tools = FinancialTools()
        self.query_parser = GeminiAgent(
            gemini_api_key,
            "Query Parser",
            """You are a financial query parser. Extract from the user query:
- Stock symbol (ticker)
- Analysis type (technical, fundamental, comprehensive)
- Time period (like 1d, 1mo, 6mo, 1y)
Return a JSON object with keys: symbol, analysis_type, time_period."""
        )
        self.market_analyst = GeminiAgent(
            gemini_api_key,
            "Market Analyst",
            """You are a senior financial analyst. Provide a clear, professional, and actionable analysis for the stock.
Include market trends, price action, risk assessment and investment recommendations."""
        )

    def parse_query(self, query: str):
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
        prompt = (
            f"Analyze the stock symbol {query_info['symbol']} for the period "
            f"{query_info.get('time_period', '6mo')}. Use recent price data and technical indicators "
            f"to provide market trends, risk factors, and recommendations."
        )
        return self.market_analyst.generate(prompt)

    def analyze(self, query: str) -> str:
        query_info = self.parse_query(query)
        period = _normalize_period(query_info.get("time_period", "6mo"))
        stock_data = self.tools.get_stock_data(query_info["symbol"], period)
        if stock_data.empty:
            return f"No data available for symbol {query_info['symbol']} in period {period}."

        analysis = self.analyze_market(query_info, stock_data)
        return analysis


def run_financial_analysis(query: str) -> str:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return "Error: GEMINI_API_KEY environment variable is not set."
    team = FinancialAnalysisTeam(gemini_api_key)
    try:
        return team.analyze(query)
    except Exception as e:
        return f"Error during analysis: {e}"
