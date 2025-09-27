#!/usr/bin/env python3

"""
Financial Analysis Team using Gemini AI
"""

import json
import re
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import logging
import sys

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Google Generative AI not installed. Install with: pip install google-generativeai", file=sys.stderr)
    raise

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Required packages not installed. Install with: pip install yfinance pandas", file=sys.stderr)
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
    """Financial data and news tools"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.firecrawl_api_key = firecrawl_api_key
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.history(period=period)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get basic stock information"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def get_stock_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Get stock news - simplified version"""
        try:
            ticker = yf.Ticker(symbol)
            news = getattr(ticker, 'news', None)
            if not news:
                # Some versions of yfinance may expose news via an attribute or method
                news = []

            parsed = []
            for item in news[:limit]:
                # yfinance news items can vary across versions; be defensive
                title = item.get('title') if isinstance(item, dict) and 'title' in item else item.get('summary', '') if isinstance(item, dict) else ''
                url = item.get('link') or item.get('url') or '' if isinstance(item, dict) else ''
                content = item.get('summary') or item.get('content') or '' if isinstance(item, dict) else ''
                parsed.append({'title': title, 'url': url, 'content': content})

            return parsed
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []


def _normalize_period(period: str) -> str:
    """Normalize common period shorthand to yfinance-compatible period strings.

    Examples:
    - '6m' -> '6mo'
    - '1y' -> '1y'
    - '1d' -> '1d'
    - '3m' -> '3mo'
    If period is already yfinance-friendly, return it unchanged.
    """
    if not period:
        return '6mo'
    p = str(period).strip().lower()
    # Accept formats like '6m', '6mo', '1y', '1d'
    m = re.match(r'^(\d+)(m|mo|y|d)$', p)
    if m:
        num, unit = m.groups()
        if unit == 'm':
            unit = 'mo'
        return f"{num}{unit}"
    return p

class GeminiAgent:
    """Base agent using Gemini AI"""
    
    def __init__(self, api_key: str, role: str, system_prompt: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.role = role
        self.system_prompt = system_prompt
        
        # Configure safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def generate(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            full_prompt = f"{self.system_prompt}\n\nUser Request: {prompt}"
            response = self.model.generate_content(
                full_prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Unable to generate response - {str(e)}"

class FinancialAnalysisTeam:
    """Multi-agent financial analysis team using Gemini AI"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None, gemini_api_key: str = None):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")
            
        self.tools = FinancialTools(firecrawl_api_key)
        
        # Initialize agents
        self.query_parser = GeminiAgent(
            api_key=gemini_api_key,
            role="Query Parser",
            system_prompt="""You are a financial query parser. Extract key information from user queries:
- Stock symbol (primary focus)
- Analysis type (technical, fundamental, comprehensive)
- Time period (1d, 1w, 1m, 3m, 6m, 1y, 2y)
- Additional symbols for comparison
- Technical indicators requested

Return a JSON object with: symbol, analysis_type, time_period, additional_symbols, indicators.
If information is unclear, use reasonable defaults."""
        )
        
        self.code_generator = GeminiAgent(
            api_key=gemini_api_key,
            role="Code Generator",
            system_prompt="""You are a Python code generator for financial analysis. Generate clean, executable Python code using:
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

Focus on creating practical, working code that provides valuable insights."""
        )
        
        self.market_analyst = GeminiAgent(
            api_key=gemini_api_key,
            role="Market Analyst",
            system_prompt="""You are a senior financial analyst. Provide expert market insights and investment recommendations based on:
- Technical analysis patterns
- Market trends and sentiment
- Risk assessment
- Investment opportunities

Be professional, balanced, and provide actionable insights. Always include risk warnings and mention this is not financial advice."""
        )
    
    def parse_query(self, query: str) -> QueryInfo:
        """Parse natural language query"""
        try:
            response = None
            try:
                response = self.query_parser.generate(query)
            except Exception as gen_err:
                # Log and continue to heuristic fallback below
                logger.warning(f"Gemini generation failed, falling back to heuristic parsing: {gen_err}")

            data = None
            if response:
                # Extract JSON from response if present
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except Exception:
                        # Invalid JSON in model output; ignore and use heuristic fallback
                        logger.warning("Model output contained invalid JSON — falling back to heuristic parsing")

            # If model didn't provide valid JSON, fall back to heuristic extraction + yfinance validation
            if not data:
                # Try to heuristically extract a ticker symbol from the query
                symbol_match = re.search(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,4})?)\b", query.upper())
                if not symbol_match:
                    raise ValueError("No ticker symbol found in query. Please include a valid stock ticker.")

                symbol = symbol_match.group(1)

                # Validate the extracted symbol with yfinance to ensure real data exists
                try:
                    info = yf.Ticker(symbol).info
                    if not info or info.get('regularMarketPrice') is None:
                        raise ValueError(f"Ticker '{symbol}' not found or has no market data.")
                except Exception as e:
                    raise ValueError(f"Ticker validation failed for '{symbol}': {e}")

                data = {"symbol": symbol, "analysis_type": "comprehensive", "time_period": "6m", "additional_symbols": [], "indicators": []}
            
            # Ensure the parser returned a real symbol; do not silently
            # fall back to any hard-coded ticker.
            symbol = data.get("symbol")
            if not symbol:
                raise ValueError("Parser did not return a ticker symbol.")

            return QueryInfo(
                symbol=symbol,
                analysis_type=data.get("analysis_type", "comprehensive"),
                time_period=data.get("time_period", "6m"),
                additional_symbols=data.get("additional_symbols", []),
                indicators=data.get("indicators", [])
            )
        except Exception as e:
            # Log the parsing/validation error and re-raise so the higher-level
            # workflow can produce an informative AnalysisResult instead of
            # silently using mock data.
            logger.error(f"Error parsing query: {e}", exc_info=True)
            raise
    
    def generate_code(self, query_info: QueryInfo, stock_data: pd.DataFrame) -> str:
        """Generate Python analysis code"""
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
        """Generate market insights and recommendations"""
        # Prepare data summary with defensive checks
        data_summary = "No data available"
        try:
            if not stock_data.empty and 'Close' in stock_data.columns and len(stock_data['Close']) >= 1:
                latest_price = stock_data['Close'].iloc[-1]

                # Compute price change only if we have at least two data points and non-zero first value
                if len(stock_data['Close']) >= 2 and stock_data['Close'].iloc[0] != 0:
                    price_change = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
                else:
                    price_change = 0.0

                # Volatility requires at least two points
                if len(stock_data['Close']) >= 2:
                    volatility = stock_data['Close'].pct_change().std() * 100
                else:
                    volatility = 0.0

                data_summary = f"Latest Price: ${latest_price:.2f}, Change: {price_change:.1f}%, Volatility: {volatility:.1f}%"
        except Exception as e:
            logger.warning(f"Failed to compute data summary: {e}")
        
        news_summary = ""
        if news:
            news_summary = "Recent news: " + "; ".join([item.get('title', '')[:50] for item in news[:3]])
        
        insights_prompt = f"""Analyze this stock data and provide expert insights:

Stock: {query_info.symbol}
Analysis Period: {query_info.time_period}
Data Summary: {data_summary}
{news_summary}

Provide professional market insights covering:
- Current market position
- Technical patterns observed
- Market sentiment
- Key risk factors
- Investment thesis"""
        
        recommendations_prompt = f"""Based on the analysis of {query_info.symbol}, provide investment recommendations:

Data: {data_summary}
{news_summary}

Provide:
- Clear investment recommendation (Buy/Hold/Sell with rationale)
- Risk assessment
- Price targets or key levels
- Time horizon considerations
- Portfolio allocation suggestions

Remember to include appropriate disclaimers."""
        
        insights = self.market_analyst.generate(insights_prompt)
        recommendations = self.market_analyst.generate(recommendations_prompt)
        
        return insights, recommendations
    
    def analyze(self, query: str) -> AnalysisResult:
        """Complete financial analysis workflow"""
        try:
            # Parse query (may raise ValueError if no valid ticker found)
            try:
                query_info = self.parse_query(query)
            except Exception as e:
                # Return a structured AnalysisResult indicating the input error
                return AnalysisResult(
                    query=QueryInfo(symbol="INVALID"),
                    insights=f"Input error: {str(e)}",
                    code="# No code generated due to input error",
                    recommendations="Please provide a valid stock ticker in your query.",
                    news=[]
                )

            logger.info(f"Analyzing {query_info.symbol}")
            # Normalize period for yfinance compatibility and fetch data
            period = _normalize_period(query_info.time_period)
            stock_data = self.tools.get_stock_data(query_info.symbol, period)
            news_data = self.tools.get_stock_news(query_info.symbol, 5)

            if stock_data.empty:
                return AnalysisResult(
                    query=query_info,
                    insights="No market data could be retrieved for the provided ticker.",
                    code="# No code generated because historical price data is unavailable",
                    recommendations="Verify the ticker symbol and try again. If the ticker is correct, the data source may be temporarily unavailable.",
                    news=[]
                )
            
            # Generate analysis components
            code = self.generate_code(query_info, stock_data)
            insights, recommendations = self.analyze_market(query_info, stock_data, news_data)
            
            # Format news
            news_items = []
            for item in news_data[:3]:
                news_items.append(NewsItem(
                    title=item.get('title', 'Market Update'),
                    summary=item.get('content', '')[:200],
                    url=item.get('url', '')
                ))
            
            return AnalysisResult(
                query=query_info,
                insights=insights,
                code=code,
                recommendations=recommendations,
                news=news_items
            )
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            # Return error result
            return AnalysisResult(
                query=QueryInfo(symbol="ERROR"),
                insights=f"Analysis failed: {str(e)}",
                code="# Error: Unable to generate code",
                recommendations="Unable to provide recommendations due to error."
            )
