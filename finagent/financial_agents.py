#!/usr/bin/env python3

import json
import re
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    import yfinance as yf
    import pandas as pd
except ImportError as e:
    print(f"Required packages not installed: {e}\nInstall with: pip install google-generativeai yfinance pandas", file=sys.stderr)
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class QueryInfo:
    symbol: str
    analysis_type: str = "comprehensive"
    time_period: str = "6mo"
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
        """Fetch historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_stock_info(self, symbol: str) -> Dict:
        """Fetch stock metadata and information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info if info else {}
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}

    def get_stock_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Fetch recent news for a stock symbol"""

        def _normalize_url(u: Optional[str]) -> str:
            if not u:
                return ''
            u = str(u).strip()
            if u.startswith('//'):
                return 'https:' + u
            if u.startswith('http://') or u.startswith('https://'):
                return u
            if u.startswith('/'):
                return 'https://finance.yahoo.com' + u
            return ''

        def _safe_string(value, default: str = '') -> str:
            """Safely convert value to string, handling None cases"""
            if value is None:
                return default
            return str(value)

        try:
            ticker = yf.Ticker(symbol)
            news = getattr(ticker, 'news', []) or []

            out: List[Dict] = []
            for item in news:
                if not isinstance(item, dict):
                    continue

                title = _safe_string(
                    item.get('title') or
                    item.get('summary') or
                    item.get('headline'),
                    'Market Update'
                )

                raw_url = _safe_string(
                    item.get('link') or
                    item.get('url') or
                    item.get('linkUrl')
                )

                url = _normalize_url(raw_url)

                content = _safe_string(
                    item.get('summary') or
                    item.get('content') or
                    item.get('snippet')
                )

                out.append({
                    'title': title,
                    'url': url,
                    'content': content
                })

                if len(out) >= limit:
                    break

            return out

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []


def _normalize_period(period: str) -> str:
    """Normalize time period strings to yfinance format"""
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
    """Agent that uses Google's Gemini API for text generation"""

    def __init__(self, api_key: str, role: str, system_prompt: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.role = role
        self.system_prompt = system_prompt
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def generate(self, prompt: str) -> str:
        """Generate a response using Gemini"""
        try:
            full_prompt = f"{self.system_prompt}\n\nUser Request: {prompt}"
            response = self.model.generate_content(
                full_prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating response for {self.role}: {e}")
            return f"Error: Unable to generate response - {e}"


class FinancialAnalysisTeam:
    """Orchestrates multiple AI agents for comprehensive financial analysis"""

    def __init__(self, firecrawl_api_key: Optional[str] = None, gemini_api_key: str = None):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")

        self.tools = FinancialTools(firecrawl_api_key)

        self.query_parser = GeminiAgent(
            gemini_api_key,
            "Query Parser",
            """You are a financial query parser. Extract key information from user queries:
- Stock symbol (primary focus) - must be a valid ticker
- Analysis type (technical, fundamental, comprehensive)
- Time period (1d, 1w, 1m, 3m, 6m, 1y, 2y, 5y)
- Additional symbols for comparison
- Technical indicators requested (RSI, MACD, Moving Averages, etc.)

Return a JSON object with: symbol, analysis_type, time_period, additional_symbols, indicators.
If information is unclear, use reasonable defaults. Always extract at least the stock symbol."""
        )

        self.code_generator = GeminiAgent(
            gemini_api_key,
            "Code Generator",
            """You are a Python code generator for financial analysis. Generate clean, executable Python code using:
- yfinance for data retrieval
- pandas for data manipulation
- matplotlib/seaborn for visualization
- numpy for numerical calculations

Requirements:
- Include all necessary imports at the top
- Add comprehensive error handling
- Create clear, informative plots with proper labels
- Use professional styling (grid, legends, titles)
- Include data validation checks
- Add comments explaining key sections
- Make code self-contained and executable

Focus on creating practical, working code that provides valuable financial insights."""
        )

        self.market_analyst = GeminiAgent(
            gemini_api_key,
            "Market Analyst",
            """You are a senior financial analyst with expertise in market analysis. Provide expert insights based on:
- Technical analysis patterns (support/resistance, trends, momentum)
- Market trends and sentiment
- Price action and volume analysis
- Risk assessment and volatility
- Investment opportunities

Be professional, balanced, and provide actionable insights. Structure your analysis clearly.
Always include risk warnings and explicitly state "This is not financial advice. Do your own research before investing."""""
        )

    def parse_query(self, query: str) -> QueryInfo:
        """Parse user query to extract structured information"""
        try:
            response = None
            try:
                response = self.query_parser.generate(query)
            except Exception as e:
                logger.warning(f"Gemini parsing failed, using fallback: {e}")

            data = None
            if response:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in model response")

            # Fallback parsing if Gemini fails
            if not data:
                symbol_match = re.search(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,4})?)\b", query.upper())
                if not symbol_match:
                    raise ValueError("No ticker symbol found in query. Please include a valid stock ticker (e.g., AAPL, MSFT).")

                symbol = symbol_match.group(1)

                # Validate symbol
                try:
                    info = yf.Ticker(symbol).info
                    if not info or 'regularMarketPrice' not in info:
                        raise ValueError(f"Ticker '{symbol}' not found or has no market data.")
                except Exception:
                    raise ValueError(f"Unable to verify ticker '{symbol}'. Please check the symbol.")

                data = {
                    "symbol": symbol,
                    "analysis_type": "comprehensive",
                    "time_period": "6mo",
                    "additional_symbols": [],
                    "indicators": []
                }

            symbol = data.get("symbol")
            if not symbol:
                raise ValueError("Could not extract ticker symbol from query.")

            return QueryInfo(
                symbol=symbol,
                analysis_type=data.get("analysis_type", "comprehensive"),
                time_period=data.get("time_period", "6mo"),
                additional_symbols=data.get("additional_symbols", []),
                indicators=data.get("indicators", [])
            )

        except Exception as e:
            logger.error(f"Error parsing query: {e}", exc_info=True)
            raise

    def generate_code(self, query_info: QueryInfo, stock_data: pd.DataFrame) -> str:
        """Generate Python code for financial analysis"""
        data_info = "No data available"
        if not stock_data.empty:
            data_info = f"{len(stock_data)} rows, columns: {', '.join(stock_data.columns[:5])}"

        prompt = f"""Generate Python code for financial analysis:

Stock Symbol: {query_info.symbol}
Analysis Type: {query_info.analysis_type}
Time Period: {query_info.time_period}
Data Available: {data_info}
Requested Indicators: {', '.join(query_info.indicators) if query_info.indicators else 'Standard technical indicators'}

Create complete, executable code that:
1. Imports required libraries (yfinance, pandas, matplotlib, numpy)
2. Fetches stock data for the specified period
3. Performs technical analysis (moving averages, RSI, volume analysis)
4. Creates professional visualizations with:
   - Price chart with moving averages
   - Volume chart
   - Technical indicators
5. Calculates key metrics (returns, volatility, trends)
6. Includes proper error handling
7. Uses professional styling

Make the code self-contained and production-ready with clear comments."""

        return self.code_generator.generate(prompt)

    def analyze_market(self, query_info: QueryInfo, stock_data: pd.DataFrame, 
                      news: List[Dict] = None) -> tuple:
        """Generate market insights and recommendations"""
        data_summary = "No data available"

        try:
            if not stock_data.empty and 'Close' in stock_data.columns:
                close_prices = stock_data['Close'].dropna()
                if len(close_prices) >= 2:
                    latest = close_prices.iloc[-1]
                    first = close_prices.iloc[0]
                    change = ((latest / first) - 1) * 100 if first != 0 else 0.0
                    volatility = close_prices.pct_change().std() * 100

                    data_summary = (
                        f"Latest Price: ${latest:.2f}, "
                        f"Period Change: {change:+.1f}%, "
                        f"Volatility: {volatility:.1f}%"
                    )
        except Exception as e:
            logger.warning(f"Failed to compute data summary: {e}")

        news_summary = ""
        if news:
            news_titles = []
            for item in news[:3]:
                title = item.get('title', '') if item.get('title') is not None else ''
                if title:
                    news_titles.append(str(title)[:60])

            if news_titles:
                news_summary = "Recent headlines: " + "; ".join(news_titles)

        insights_prompt = f"""Analyze this stock and provide expert insights:

Stock: {query_info.symbol}
Analysis Period: {query_info.time_period}
Data Summary: {data_summary}
{news_summary}

Provide a comprehensive analysis covering:
1. Current market position and price action
2. Technical patterns and trends observed
3. Market sentiment indicators
4. Key support and resistance levels
5. Risk factors to consider

Be specific, professional, and actionable."""

        insights = self.market_analyst.generate(insights_prompt)

        recommendations_prompt = f"""Based on the analysis of {query_info.symbol}, provide investment recommendations:

Data: {data_summary}
{news_summary}

Provide clear guidance including:
1. Investment recommendation (Buy/Hold/Sell) with detailed rationale
2. Risk assessment (Low/Medium/High with explanation)
3. Price targets or key technical levels to watch
4. Time horizon considerations (short/medium/long term)
5. Portfolio allocation suggestions (if buying)

End with: "⚠️ Disclaimer: This is not financial advice. Always do your own research and consult with a financial advisor before making investment decisions."""

        recommendations = self.market_analyst.generate(recommendations_prompt)

        return insights, recommendations

    def analyze(self, query: str) -> AnalysisResult:
        """Main analysis pipeline"""
        try:
            # Parse the query
            try:
                query_info = self.parse_query(query)
            except Exception as e:
                logger.error(f"Query parsing failed: {e}")
                return AnalysisResult(
                    query=QueryInfo(symbol="INVALID"),
                    insights=f"Input error: {e}",
                    code="# No code generated due to input error",
                    recommendations="Please provide a valid stock ticker in your query.",
                    news=[]
                )

            logger.info(f"Analyzing {query_info.symbol} for period {query_info.time_period}")

            # Fetch data
            period = _normalize_period(query_info.time_period)
            stock_data = self.tools.get_stock_data(query_info.symbol, period)
            news_data = self.tools.get_stock_news(query_info.symbol, 5)

            # Check if we have data
            if stock_data.empty:
                logger.warning(f"No data available for {query_info.symbol}")
                return AnalysisResult(
                    query=query_info,
                    insights="No market data could be retrieved for the provided ticker.",
                    code="# No code generated because historical price data is unavailable",
                    recommendations="Please verify the ticker symbol and try again.",
                    news=[]
                )

            # Generate analysis components
            code = self.generate_code(query_info, stock_data)
            insights, recommendations = self.analyze_market(query_info, stock_data, news_data)

            # Process news items with safe string handling
            news_items = []
            for item in news_data[:3]:
                title = item.get('title', '') if item.get('title') is not None else 'Market Update'
                content = item.get('content', '') if item.get('content') is not None else ''
                url = item.get('url', '') if item.get('url') is not None else ''

                # Safe string slicing
                summary = str(content)[:200] if len(str(content)) > 200 else str(content)

                news_items.append(NewsItem(
                    title=str(title),
                    summary=summary,
                    url=str(url)
                ))

            return AnalysisResult(
                query=query_info,
                insights=insights,
                code=code,
                recommendations=recommendations,
                news=news_items
            )

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}", exc_info=True)
            return AnalysisResult(
                query=QueryInfo(symbol="ERROR"),
                insights=f"Analysis failed: {str(e)}",
                code="# Error: Unable to generate code",
                recommendations="Unable to provide recommendations due to error.",
                news=[]
            )
