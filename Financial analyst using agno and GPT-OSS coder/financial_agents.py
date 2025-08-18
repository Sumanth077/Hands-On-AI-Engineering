#!/usr/bin/env python3

"""
Financial Analysis Agents using Agno Framework with GPT-OSS
Separate agentic logic for the MCP financial analyst
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Firecrawl imports
try:
    from firecrawl import FirecrawlApp
except ImportError:
    print("Firecrawl not installed. Install with: pip install firecrawl-py")

# Agno imports
try:
    from agno import Agent, Team
    from agno.llm.ollama import OllamaLLM
    from agno.tools import Tool
except ImportError:
    print("Agno not installed. Install with: pip install agno")

logger = logging.getLogger(__name__)

# Pydantic models
class StockQuery(BaseModel):
    """Structured stock query model"""
    symbol: str = Field(..., description="Stock ticker symbol")
    analysis_type: str = Field(..., description="Type of analysis")
    time_period: str = Field(default="1y", description="Time period")
    additional_symbols: Optional[List[str]] = Field(default=None)
    indicators: Optional[List[str]] = Field(default=None)

class MarketNews(BaseModel):
    """Market news structure"""
    title: str
    summary: str
    url: str
    published: str
    relevance_score: float

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    query: StockQuery
    code: str
    insights: str
    news: List[MarketNews]
    recommendations: str

class GPTOSSInterface:
    """Interface to GPT-OSS model via Ollama"""
    
    def __init__(self, model_name: str = "gpt-oss", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def query(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Query GPT-OSS model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_ctx": kwargs.get("num_ctx", 4096)
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Error querying GPT-OSS: {e}")
            return f"Error: {str(e)}"

class FinancialTools:
    """Financial analysis tools for agents"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.firecrawl = None
        if firecrawl_api_key:
            self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
    
    def get_stock_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Fetch recent news for a stock using Firecrawl"""
        try:
            if not self.firecrawl:
                return []
                
            # Search for news on financial websites
            news_urls = [
                f"https://finance.yahoo.com/quote/{symbol}/news/",
                f"https://www.marketwatch.com/investing/stock/{symbol}",
                f"https://seekingalpha.com/symbol/{symbol}/news"
            ]
            
            news_items = []
            for url in news_urls[:2]:  # Limit to avoid rate limits
                try:
                    result = self.firecrawl.scrape_url(
                        url, 
                        params={
                            "formats": ["markdown"],
                            "onlyMainContent": True
                        }
                    )
                    
                    if result and result.get('markdown'):
                        news_items.append({
                            'url': url,
                            'content': result['markdown'][:1000],  # Limit content
                            'title': f"Latest {symbol} News"
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    continue
                    
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get basic market data (fallback if yfinance not available)"""
        try:
            if not self.firecrawl:
                return {}
                
            url = f"https://finance.yahoo.com/quote/{symbol}"
            result = self.firecrawl.scrape_url(
                url,
                params={
                    "formats": ["markdown"],
                    "onlyMainContent": True
                }
            )
            
            if result and result.get('markdown'):
                return {
                    'symbol': symbol,
                    'data': result['markdown'][:500],
                    'source': url
                }
                
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            
        return {}

class QueryParserAgent:
    """Agent for parsing natural language queries"""
    
    def __init__(self, gpt_interface: GPTOSSInterface):
        self.gpt = gpt_interface
        
        # Create Agno agent
        self.agent = Agent(
            name="Query Parser",
            role="Financial Query Analyzer",
            goal="Parse natural language financial queries into structured format",
            backstory="Expert in understanding financial terminology and extracting structured data from user requests",
            llm=self._create_ollama_llm(),
            verbose=True
        )
    
    def _create_ollama_llm(self):
        """Create Ollama LLM interface for Agno"""
        try:
            return OllamaLLM(
                model="gpt-oss",
                base_url="http://localhost:11434"
            )
        except:
            # Fallback to direct interface
            return None
    
    def parse_query(self, user_query: str) -> StockQuery:
        """Parse natural language query"""
        system_prompt = """You are a financial query parser. Extract structured information from user queries.
        
        Return ONLY a valid JSON object with these fields:
        - symbol: main stock ticker symbol (uppercase)
        - analysis_type: one of [price, trend, comparison, forecast, performance, news]
        - time_period: one of [1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
        - additional_symbols: array of additional ticker symbols if mentioned
        - indicators: array of technical indicators if requested [sma, ema, rsi, macd, bollinger]"""
        
        prompt = f"""Parse this financial query into structured JSON: "{user_query}"
        
        Examples:
        Query: "Show me Apple stock performance over 6 months"
        Output: {{"symbol": "AAPL", "analysis_type": "performance", "time_period": "6mo"}}
        
        Query: "Compare Tesla vs Ford with moving averages"
        Output: {{"symbol": "TSLA", "analysis_type": "comparison", "time_period": "1y", "additional_symbols": ["F"], "indicators": ["sma"]}}"""
        
        response = self.gpt.query(prompt, system_prompt, temperature=0.1)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return StockQuery(**data)
        except Exception as e:
            logger.warning(f"Failed to parse structured query: {e}")
        
        # Fallback parsing
        return self._fallback_parse(user_query)
    
    def _fallback_parse(self, query: str) -> StockQuery:
        """Fallback parsing logic"""
        query_upper = query.upper()
        
        # Common stock symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "F"]
        symbol = "AAPL"  # default
        
        for sym in symbols:
            if sym in query_upper:
                symbol = sym
                break
        
        # Analysis type detection
        analysis_type = "price"
        if any(word in query_upper for word in ["COMPARE", "VS", "VERSUS"]):
            analysis_type = "comparison"
        elif any(word in query_upper for word in ["TREND", "TRENDING"]):
            analysis_type = "trend"
        elif any(word in query_upper for word in ["NEWS", "LATEST"]):
            analysis_type = "news"
        elif any(word in query_upper for word in ["FORECAST", "PREDICT"]):
            analysis_type = "forecast"
        
        # Time period detection
        time_period = "1y"
        if "6 MONTH" in query_upper or "6MO" in query_upper:
            time_period = "6mo"
        elif "3 MONTH" in query_upper or "3MO" in query_upper:
            time_period = "3mo"
        elif "1 MONTH" in query_upper or "1MO" in query_upper:
            time_period = "1mo"
        elif "2 YEAR" in query_upper or "2Y" in query_upper:
            time_period = "2y"
        
        return StockQuery(
            symbol=symbol,
            analysis_type=analysis_type,
            time_period=time_period
        )

class CodeGeneratorAgent:
    """Agent for generating Python analysis code"""
    
    def __init__(self, gpt_interface: GPTOSSInterface):
        self.gpt = gpt_interface
        
        self.agent = Agent(
            name="Code Generator",
            role="Python Financial Code Developer",
            goal="Generate clean, executable Python code for financial analysis",
            backstory="Senior Python developer specializing in financial data analysis with pandas, matplotlib, and yfinance",
            llm=self._create_ollama_llm(),
            verbose=True
        )
    
    def _create_ollama_llm(self):
        """Create Ollama LLM interface"""
        try:
            return OllamaLLM(model="gpt-oss", base_url="http://localhost:11434")
        except:
            return None
    
    def generate_code(self, query: StockQuery, news_data: List[Dict] = None) -> str:
        """Generate Python code for analysis"""
        
        system_prompt = """You are a senior Python developer creating financial analysis code.
        
        Requirements:
        - Use yfinance for data fetching
        - Use pandas for data manipulation
        - Use matplotlib for visualization
        - Include error handling
        - Create professional-looking plots
        - Add proper titles, labels, and legends
        - Return clean, executable code without markdown formatting"""
        
        news_context = ""
        if news_data:
            news_context = f"\nRecent news context: {json.dumps([n.get('title', '') for n in news_data[:2]])}"
        
        prompt = f"""Generate Python code for this financial analysis:
        
        Symbol: {query.symbol}
        Analysis Type: {query.analysis_type}
        Time Period: {query.time_period}
        Additional Symbols: {query.additional_symbols or []}
        Technical Indicators: {query.indicators or []}
        {news_context}
        
        Code requirements:
        1. Fetch data using yfinance
        2. Create meaningful visualizations
        3. Calculate relevant metrics
        4. Handle errors gracefully
        5. Include comments explaining the analysis
        6. Use proper plot formatting with titles and labels
        
        Return only the Python code, no explanations or markdown."""
        
        code = self.gpt.query(prompt, system_prompt, temperature=0.2)
        
        # Clean the response
        return self._clean_code_response(code, query)
    
    def _clean_code_response(self, response: str, query: StockQuery) -> str:
        """Clean and validate code response"""
        lines = response.split('\n')
        clean_lines = []
        in_code_block = False
        
        for line in lines:
            if '```python' in line.lower():
                in_code_block = True
                continue
            elif '```' in line and in_code_block:
                break
            elif line.strip():
                # Keep lines that look like Python code
                if (in_code_block or 
                    line.startswith('import ') or 
                    line.startswith('from ') or 
                    any(keyword in line for keyword in ['yf.', 'plt.', 'pd.', 'np.']) or
                    line.startswith('    ') or  # Indented lines
                    '=' in line or
                    line.startswith('#')):
                    clean_lines.append(line)
        
        if clean_lines:
            return '\n'.join(clean_lines)
        
        # Fallback code generation
        return self._generate_fallback_code(query)
    
    def _generate_fallback_code(self, query: StockQuery) -> str:
        """Generate fallback code if AI generation fails"""
        additional_plots = ""
        if query.additional_symbols:
            for symbol in query.additional_symbols[:2]:  # Limit to 2 additional
                additional_plots += f"""
# Add {symbol} for comparison
{symbol.lower()}_data = yf.Ticker('{symbol}').history(period='{query.time_period}')
if not {symbol.lower()}_data.empty:
    plt.plot({symbol.lower()}_data.index, {symbol.lower()}_data['Close'], 
             linewidth=2, label='{symbol} Close Price')
"""
        
        indicators_code = ""
        if query.indicators:
            if 'sma' in [i.lower() for i in query.indicators]:
                indicators_code += """
# Simple Moving Average
data['SMA_20'] = data['Close'].rolling(window=20).mean()
plt.plot(data.index, data['SMA_20'], '--', label='SMA 20', alpha=0.7)
"""
        
        return f"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

try:
    # Fetch stock data
    print(f"Fetching data for {query.symbol}...")
    ticker = yf.Ticker('{query.symbol}')
    data = ticker.history(period='{query.time_period}')
    
    if data.empty:
        print(f"No data available for {query.symbol}")
        exit()
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Main stock price
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], linewidth=2, label=f'{query.symbol} Close Price', color='blue')
    {additional_plots}
    {indicators_code}
    
    plt.title(f'{query.symbol} Stock Analysis - {query.analysis_type.title()} ({query.time_period.upper()})', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Volume subplot
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['Volume'], alpha=0.7, color='gray', label='Volume')
    plt.title('Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    current_price = data['Close'].iloc[-1]
    start_price = data['Close'].iloc[0]
    total_return = ((current_price / start_price) - 1) * 100
    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
    
    print(f"\\n=== {query.symbol} Analysis Results ===")
    print(f"Period: {query.time_period.upper()}")
    print(f"Start Price: ${start_price:.2f}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Volatility: {volatility:.2f}%")
    print(f"Average Volume: {data['Volume'].mean():,.0f}")
    
except Exception as e:
    print(f"Error in analysis: {{e}}")
    print("Please check the stock symbol and try again.")
""".strip()

class MarketAnalystAgent:
    """Agent for market analysis and insights"""
    
    def __init__(self, gpt_interface: GPTOSSInterface, tools: FinancialTools):
        self.gpt = gpt_interface
        self.tools = tools
        
        self.agent = Agent(
            name="Market Analyst",
            role="Financial Market Expert",
            goal="Provide comprehensive market insights and analysis",
            backstory="Experienced financial analyst with deep knowledge of market trends, technical analysis, and economic indicators",
            llm=self._create_ollama_llm(),
            verbose=True
        )
    
    def _create_ollama_llm(self):
        """Create Ollama LLM interface"""
        try:
            return OllamaLLM(model="gpt-oss", base_url="http://localhost:11434")
        except:
            return None
    
    def analyze_stock(self, query: StockQuery, news_data: List[Dict] = None) -> str:
        """Generate market insights"""
        
        news_summary = ""
        if news_data:
            news_items = [item.get('content', '')[:200] for item in news_data[:3]]
            news_summary = f"\n\nRecent News Context:\n" + "\n".join(f"- {item}" for item in news_items if item)
        
        system_prompt = """You are a professional financial analyst providing market insights.
        
        Provide analysis covering:
        - Current market sentiment
        - Technical outlook
        - Key factors affecting the stock
        - Risk assessment
        - Investment considerations
        
        Keep insights concise but informative (200-300 words)."""
        
        prompt = f"""Provide market analysis for {query.symbol} stock:
        
        Analysis Focus: {query.analysis_type}
        Time Period: {query.time_period}
        Comparison Stocks: {query.additional_symbols or 'None'}
        Technical Indicators: {query.indicators or 'None'}
        {news_summary}
        
        Include:
        1. Current market position assessment
        2. Technical analysis perspective
        3. Key catalysts or concerns
        4. Risk factors to consider
        5. Brief outlook summary"""
        
        insights = self.gpt.query(prompt, system_prompt, temperature=0.3)
        
        return insights

class FinancialAnalysisTeam:
    """Orchestrate the financial analysis team using Agno"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.gpt_interface = GPTOSSInterface()
        self.tools = FinancialTools(firecrawl_api_key)
        
        # Initialize agents
        self.query_parser = QueryParserAgent(self.gpt_interface)
        self.code_generator = CodeGeneratorAgent(self.gpt_interface)
        self.market_analyst = MarketAnalystAgent(self.gpt_interface, self.tools)
        
        logger.info("Financial Analysis Team initialized with GPT-OSS and Firecrawl")
    
    def analyze(self, user_query: str) -> AnalysisResult:
        """Perform complete financial analysis"""
        try:
            # Step 1: Parse the query
            logger.info(f"Parsing query: {user_query}")
            parsed_query = self.query_parser.parse_query(user_query)
            
            # Step 2: Gather news and market data
            logger.info(f"Gathering news for {parsed_query.symbol}")
            news_data = self.tools.get_stock_news(parsed_query.symbol)
            
            # Step 3: Generate analysis code
            logger.info("Generating analysis code")
            analysis_code = self.code_generator.generate_code(parsed_query, news_data)
            
            # Step 4: Generate market insights
            logger.info("Generating market insights")
            market_insights = self.market_analyst.analyze_stock(parsed_query, news_data)
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(parsed_query, market_insights)
            
            # Step 6: Format news data
            formatted_news = []
            for news_item in news_data[:3]:
                formatted_news.append(MarketNews(
                    title=news_item.get('title', 'Market News'),
                    summary=news_item.get('content', '')[:200],
                    url=news_item.get('url', ''),
                    published=datetime.now().isoformat(),
                    relevance_score=0.8
                ))
            
            return AnalysisResult(
                query=parsed_query,
                code=analysis_code,
                insights=market_insights,
                news=formatted_news,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            # Return minimal result on error
            return AnalysisResult(
                query=StockQuery(symbol="AAPL", analysis_type="price"),
                code="# Error generating code",
                insights=f"Error in analysis: {str(e)}",
                news=[],
                recommendations="Please check the query and try again."
            )
    
    def _generate_recommendations(self, query: StockQuery, insights: str) -> str:
        """Generate investment recommendations"""
        system_prompt = """Generate brief investment recommendations based on the analysis.
        Include risk level and key actionable points. Keep it concise (100-150 words)."""
        
        prompt = f"""Based on this analysis for {query.symbol}:

        Analysis Type: {query.analysis_type}
        Market Insights: {insights[:300]}...
        
        Provide investment recommendations including:
        1. Risk assessment (Low/Medium/High)
        2. Key action items
        3. Timeline considerations
        4. Important disclaimers"""
        
        recommendations = self.gpt_interface.query(prompt, system_prompt, temperature=0.2)
        
        return recommendations + "\n\n⚠️ Disclaimer: This is not financial advice. Consult with a financial advisor before making investment decisions."