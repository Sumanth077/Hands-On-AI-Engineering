import os
import re
import json
import yfinance as yf
from dotenv import load_dotenv
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

load_dotenv()

def _normalize_period(period: str) -> str:
    if not period:
        return "6mo"
    p = str(period).strip().lower()
    m = re.match(r"^(\d+)(m|mo|y|d|w)$", p)
    if m:
        num, unit = m.group(1), m.group(2)
        if unit == "m":
            unit = "mo"
        return f"{num}{unit}"
    return p

# Gemini Agent wrapper
class GeminiAgent:
    def __init__(self, api_key: str, role: str, system_prompt: str):
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
        }
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=generation_config,
        )
        self.role = role
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        full_prompt = f"{self.system_prompt}\n\nUser Request: {prompt}"
        response = self.model.generate_content(full_prompt)
        return response.text.strip()

@tool("Fetch Financial Data")
def financial_data_tool(symbol: str, period: str = "6mo") -> str:
    """Fetches historical stock data for a given symbol and time period using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return json.dumps({"error": f"No data available for {symbol}."})
        # Return data summary as JSON (example: number of rows)
        return json.dumps({"symbol": symbol, "period": period, "data_points": len(data)})
    except Exception as e:
        return json.dumps({"error": f"Error fetching data for {symbol}: {str(e)}"})

class FinancialAnalysisTeam:
    def __init__(self, gemini_api_key: str):
        self.tools = [financial_data_tool]
        self.query_parser_llm = GeminiAgent(
            gemini_api_key,
            "Query Parser",
            (
                "You are a financial query parser. Extract from the user query:"
                "\n- Stock symbol (ticker)"
                "\n- Analysis type (technical, fundamental, comprehensive)"
                "\n- Time period (like 1d, 1mo, 6mo, 1y)"
                "\nRespond ONLY in valid JSON format."
            ),
        )
        self.market_analyst_llm = GeminiAgent(
            gemini_api_key,
            "Market Analyst",
            (
                "You are a senior financial analyst. Provide a clear, professional, and actionable analysis "
                "for the stock. Include market trends, price action, risk assessment, and investment recommendations. "
                "Respond ONLY in valid JSON format."
            ),
        )
        self.parser_agent = Agent(
            role="Query Parser",
            goal="Extract symbol, analysis type, and time period from a user query.",
            backstory="Expert at understanding financial user queries and returning structured data.",
            tools=self.tools,
            verbose=True,
        )
        self.analyst_agent = Agent(
            role="Market Analyst",
            goal="Analyze fetched market data and provide insights.",
            backstory="Experienced financial professional who writes insightful reports.",
            tools=self.tools,
            verbose=True,
        )

    def parse_query(self, query: str):
        """Parse query to extract symbol, analysis type, and time period using regex."""
        symbol_match = re.search(r"\b([A-Z]{2,5})\b", query.upper())
        if not symbol_match:
            raise ValueError(
                "No stock ticker symbol found in query. Please include a ticker symbol (e.g., AAPL, GOOGL)."
            )
        symbol = symbol_match.group(1)
        analysis_type = "comprehensive"
        if re.search(r"\btechnical\b", query, re.IGNORECASE):
            analysis_type = "technical"
        elif re.search(r"\bfundamental\b", query, re.IGNORECASE):
            analysis_type = "fundamental"
        time_period = "6mo"
        period_match = re.search(
            r"\b(\d+)\s*(day|week|month|year|d|w|mo|m|y)s?\b", query, re.IGNORECASE
        )
        if period_match:
            num = period_match.group(1)
            unit = period_match.group(2).lower()
            if unit in ["day", "d"]:
                time_period = f"{num}d"
            elif unit in ["week", "w"]:
                time_period = f"{num}wk"
            elif unit in ["month", "mo", "m"]:
                time_period = f"{num}mo"
            elif unit in ["year", "y"]:
                time_period = f"{num}y"
        return {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "time_period": time_period,
        }

    def analyze_market(self, query_info, stock_data) -> str:
        prompt = (
            f"Analyze the stock symbol {query_info['symbol']} for the period "
            f"{query_info.get('time_period', '6mo')}. "
            f"Use the historical price data to identify trends, risks, and opportunities."
        )
        return self.market_analyst_llm.generate(prompt)

    def analyze(self, query: str) -> str:
        query_info = self.parse_query(query)
        period = _normalize_period(query_info.get("time_period", "6mo"))
        ticker = yf.Ticker(query_info["symbol"])
        stock_data = ticker.history(period=period)
        if stock_data.empty:
            return json.dumps({"error": f"No data available for {query_info['symbol']} ({period})."})
        analysis = self.analyze_market(query_info, stock_data)
        # Ensure we return valid JSON from analysis
        try:
            return json.dumps(json.loads(analysis), indent=2)
        except Exception:
            # Fallback: wrap output in JSON object if not valid JSON
            return json.dumps({"analysis": analysis})

def run_financial_analysis(query: str) -> str:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return json.dumps({"error": "GEMINI_API_KEY environment variable is not set."})
    team = FinancialAnalysisTeam(gemini_api_key)
    try:
        return team.analyze(query)
    except Exception as e:
        return json.dumps({"error": f"Error during analysis: {str(e)}"})
