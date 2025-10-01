#!/usr/bin/env python3

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
import sys

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    import yfinance as yf
    import pandas as pd
except ImportError as e:
    print(f"Required packages not installed: {e}\nInstall with: pip install google-generativeai yfinance pandas", file=sys.stderr)
    raise

logger = logging.getLogger(__name__)

@dataclass
class QueryInfo:
    symbol: str
    analysis_type: str = "comprehensive"
    time_period: str = "6m"
    additional_symbols: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)

@dataclass
class NewsItem:
    title: str
    summary: str
    url: str

@dataclass
class AnalysisResult:
    query: QueryInfo
    insights: str
    code: str
    recommendations: str
    news: List[NewsItem] = field(default_factory=list)

class FinancialTools:
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.firecrawl_api_key = firecrawl_api_key
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        try:
            return yf.Ticker(symbol).history(period=period)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        try:
            return yf.Ticker(symbol).info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def get_stock_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        try:
            ticker = yf.Ticker(symbol)
            news = getattr(ticker, 'news', [])
            return [{'title': item.get('title', item.get('summary', '')), 
                    'url': item.get('link') or item.get('url', ''),
                    'content': item.get('summary') or item.get('content', '')} 
                   for item in news[:limit] if isinstance(item, dict)]
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

def _normalize_period(period: str) -> str:
    if not period:
        return '6mo'
    p = str(period).strip().lower()
    m = re.match(r'^(\d+)(m|mo|y|d)$', p)
    return f"{m.group(1)}{'mo' if m.group(2) == 'm' else m.group(2)}" if m else p

class GeminiAgent:
    def __init__(self, api_key: str, role: str, system_prompt: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.role = role
        self.system_prompt = system_prompt
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                f"{self.system_prompt}\n\nUser Request: {prompt}",
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=2000)
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Unable to generate response - {e}"

class FinancialAnalysisTeam:
    def __init__(self, firecrawl_api_key: Optional[str] = None, gemini_api_key: str = None):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")
            
        self.tools = FinancialTools(firecrawl_api_key)
        
        self.query_parser = GeminiAgent(gemini_api_key, "Query Parser",
            """You are a financial query parser. Extract key information from user queries:
- Stock symbol (primary focus)
- Analysis type (technical, fundamental, comprehensive)
- Time period (1d, 1w, 1m, 3m, 6m, 1y, 2y)
- Additional symbols for comparison
- Technical indicators requested

Return a JSON object with: symbol, analysis_type, time_period, additional_symbols, indicators.
If information is unclear, use reasonable defaults.""")
        
        self.code_generator = GeminiAgent(gemini_api_key, "Code Generator",
            """You are a Python code generator for financial analysis. Generate clean, executable Python code using:
- yfinance for data
- pandas for analysis
- matplotlib for visualization
- numpy for calculations

Requirements:
- Include proper imports
- Add error handling
- Create informative plots
- Use professional styling
- Include data validation
- Add comments explaining key sections

Focus on creating practical, working code that provides valuable insights.""")
        
        self.market_analyst = GeminiAgent(gemini_api_key, "Market Analyst",
            """You are a senior financial analyst. Provide expert market insights and investment recommendations based on:
- Technical analysis patterns
- Market trends and sentiment
- Risk assessment
- Investment opportunities

Be professional, balanced, and provide actionable insights. Always include risk warnings and mention this is not financial advice.""")
    
    def parse_query(self, query: str) -> QueryInfo:
        try:
            response = None
            try:
                response = self.query_parser.generate(query)
            except Exception as e:
                logger.warning(f"Gemini generation failed, falling back to heuristic parsing: {e}")

            data = None
            if response:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except Exception:
                        logger.warning("Model output contained invalid JSON")

            if not data:
                symbol_match = re.search(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,4})?)\b", query.upper())
                if not symbol_match:
                    raise ValueError("No ticker symbol found in query. Please include a valid stock ticker.")
                
                symbol = symbol_match.group(1)
                info = yf.Ticker(symbol).info
                if not info or info.get('regularMarketPrice') is None:
                    raise ValueError(f"Ticker '{symbol}' not found or has no market data.")
                
                data = {"symbol": symbol, "analysis_type": "comprehensive", "time_period": "6m", 
                       "additional_symbols": [], "indicators": []}
            
            symbol = data.get("symbol")
            if not symbol:
                raise ValueError("Parser did not return a ticker symbol.")

            return QueryInfo(symbol=symbol, analysis_type=data.get("analysis_type", "comprehensive"),
                           time_period=data.get("time_period", "6m"),
                           additional_symbols=data.get("additional_symbols", []),
                           indicators=data.get("indicators", []))
        except Exception as e:
            logger.error(f"Error parsing query: {e}", exc_info=True)
            raise
    
    def generate_code(self, query_info: QueryInfo, stock_data: pd.DataFrame) -> str:
        prompt = f"""Generate Python code for financial analysis:

Stock: {query_info.symbol}
Analysis Type: {query_info.analysis_type}
Time Period: {query_info.time_period}
Data Shape: {stock_data.shape if not stock_data.empty else "No data"}

Create code that:
1. Fetches data using yfinance
2. Performs technical analysis
3. Creates professional visualizations
4. Calculates key metrics
5. Handles errors gracefully

Make it production-ready and well-documented."""
        return self.code_generator.generate(prompt)
    
    def analyze_market(self, query_info: QueryInfo, stock_data: pd.DataFrame, news: List[Dict] = None) -> tuple:
        data_summary = "No data available"
        try:
            if not stock_data.empty and 'Close' in stock_data.columns and len(stock_data['Close']) >= 1:
                latest = stock_data['Close'].iloc[-1]
                change = ((latest / stock_data['Close'].iloc[0]) - 1) * 100 if len(stock_data['Close']) >= 2 and stock_data['Close'].iloc[0] != 0 else 0.0
                vol = stock_data['Close'].pct_change().std() * 100 if len(stock_data['Close']) >= 2 else 0.0
                data_summary = f"Latest Price: ${latest:.2f}, Change: {change:.1f}%, Volatility: {vol:.1f}%"
        except Exception as e:
            logger.warning(f"Failed to compute data summary: {e}")
        
        news_summary = "Recent news: " + "; ".join([item.get('title', '')[:50] for item in news[:3]]) if news else ""
        
        insights = self.market_analyst.generate(f"""Analyze this stock data and provide expert insights:

Stock: {query_info.symbol}
Analysis Period: {query_info.time_period}
Data Summary: {data_summary}
{news_summary}

Provide professional market insights covering:
- Current market position
- Technical patterns observed
- Market sentiment
- Key risk factors
- Investment thesis""")
        
        recommendations = self.market_analyst.generate(f"""Based on the analysis of {query_info.symbol}, provide investment recommendations:

Data: {data_summary}
{news_summary}

Provide:
- Clear investment recommendation (Buy/Hold/Sell with rationale)
- Risk assessment
- Price targets or key levels
- Time horizon considerations
- Portfolio allocation suggestions

Remember to include appropriate disclaimers.""")
        
        return insights, recommendations
    
    def analyze(self, query: str) -> AnalysisResult:
        try:
            try:
                query_info = self.parse_query(query)
            except Exception as e:
                return AnalysisResult(QueryInfo(symbol="INVALID"), f"Input error: {e}",
                                    "# No code generated due to input error",
                                    "Please provide a valid stock ticker in your query.", [])

            logger.info(f"Analyzing {query_info.symbol}")
            period = _normalize_period(query_info.time_period)
            stock_data = self.tools.get_stock_data(query_info.symbol, period)
            news_data = self.tools.get_stock_news(query_info.symbol, 5)

            if stock_data.empty:
                return AnalysisResult(query_info, "No market data could be retrieved for the provided ticker.",
                                    "# No code generated because historical price data is unavailable",
                                    "Verify the ticker symbol and try again.", [])
            
            code = self.generate_code(query_info, stock_data)
            insights, recommendations = self.analyze_market(query_info, stock_data, news_data)
            
            news_items = [NewsItem(item.get('title', 'Market Update'), item.get('content', '')[:200],
                                  item.get('url', '')) for item in news_data[:3]]
            
            return AnalysisResult(query_info, insights, code, recommendations, news_items)
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return AnalysisResult(QueryInfo(symbol="ERROR"), f"Analysis failed: {e}",
                                "# Error: Unable to generate code",
                                "Unable to provide recommendations due to error.")