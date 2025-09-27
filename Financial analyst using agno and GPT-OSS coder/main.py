#!/usr/bin/env python3

"""
MCP Financial Analyst Server with Gemini AI
Using FastMCP for modern MCP server implementation
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import subprocess
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
# Load environment variables from .env file
load_dotenv()

# MCP imports - Using FastMCP
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("MCP not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import agentic system
try:
    from financial_agents import FinancialAnalysisTeam, AnalysisResult, FinancialTools
    print("✅ Successfully imported financial_agents.py", file=sys.stderr)
except ImportError as e:
    print(f"❌ Error importing financial_agents.py: {str(e)}", file=sys.stderr)
    print("Make sure financial_agents.py is in the same directory and all dependencies are installed.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-financial")

class FinancialAnalysisServer:
    """Modern MCP Server for Financial Analysis with Gemini AI using FastMCP"""
    
    def __init__(self, gemini_api_key: str, firecrawl_api_key: Optional[str] = None):
        # Create FastMCP server
        self.mcp = FastMCP("financial-analyst-gemini")
        self.analysis_team = FinancialAnalysisTeam(firecrawl_api_key, gemini_api_key)
        self.setup_tools()
        
    def setup_tools(self):
        """Setup MCP tools using FastMCP decorators"""
        
        @self.mcp.tool()
        def analyze_stock(query: str) -> str:
            """Analyze stock using Gemini AI with news integration"""
            try:
                result: AnalysisResult = self.analysis_team.analyze(query)
                
                news_section = ""
                if result.news:
                    news_section = "\n### 📰 Latest News:\n"
                    for i, news in enumerate(result.news[:3], 1):
                        news_section += f"{i}. **{news.title}**\n   {news.summary[:100]}...\n\n"
                
                response = f"""# 📈 Stock Analysis - {result.query.symbol}

## 💡 AI Insights
{result.insights}

## 📊 Analysis Code
```python
{result.code}
```
{news_section}
## 🎯 Recommendations
{result.recommendations}

---
*Powered by Gemini AI*
"""
                return response
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}", exc_info=True)
                return f"❌ Analysis Error: {str(e)}"
        
        @self.mcp.tool()
        def execute_code(code: str, timeout: int = 30) -> str:
            """Execute generated Python analysis code"""
            execution_code = f"""
import sys
import warnings
warnings.filterwarnings('ignore')

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(execution_code)
                temp_file = f.name
            
            try:
                # Use UTF-8 encoding and replace errors to avoid 'charmap' codec issues on Windows
                result = subprocess.run([sys.executable, temp_file], 
                                      capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
                os.unlink(temp_file)
                
                if result.returncode == 0:
                    output = result.stdout or "Code executed successfully!"
                    # Remove or replace non-printable / non-ascii characters that may break MCP clients
                    safe_output = ''.join((ch if ord(ch) < 0x110000 else '?' for ch in output))
                    return f"**Success**\n```\n{safe_output}\n```"
                else:
                    safe_err = ''.join((ch if ord(ch) < 0x110000 else '?' for ch in (result.stderr or 'Error')))
                    return f"**Error**\n```\n{safe_err}\n```"
                    
            except subprocess.TimeoutExpired:
                os.unlink(temp_file)
                return f"⏰ Execution timeout after {timeout}s"
            except Exception as e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return f"❌ Execution error: {str(e)}"
        
        @self.mcp.tool()
        def get_news(symbol: str, limit: int = 5) -> str:
            """Fetch latest market news for a stock"""
            try:
                # Create FinancialTools instance to get news
                tools = FinancialTools()
                news_data = tools.get_stock_news(symbol, limit)
                
                if not news_data:
                    return f"📰 No news found for {symbol}"
                
                news_text = f"# 📰 News for {symbol}\n\n"
                for i, item in enumerate(news_data[:limit], 1):
                    title = item.get('title', 'Update')
                    content = item.get('content', '')[:200] + "..." if len(item.get('content', '')) > 200 else item.get('content', '')
                    url = item.get('url', '')
                    news_text += f"## {i}. {title}\n{content}\n[Read more]({url})\n\n"
                
                return news_text
                
            except Exception as e:
                logger.error(f"News fetch error: {str(e)}", exc_info=True)
                return f"❌ News fetch error: {str(e)}"
    
    def run(self):
        """Run the MCP server"""
        self.mcp.run()

def main():
    """Initialize and run the MCP server"""
    
    # Check for required packages
    missing_packages = []
    try:
        import google.generativeai as genai
    except ImportError:
        missing_packages.append("google-generativeai")
    
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        missing_packages.append("yfinance pandas")
    
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        missing_packages.append("mcp")
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}", file=sys.stderr)
        print("Install with: pip install " + " ".join(missing_packages), file=sys.stderr)
        sys.exit(1)
    
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY required. Get one from: https://makersuite.google.com/app/apikey", file=sys.stderr)
        print(f"❌ Debug: API Key found = {'Yes' if gemini_api_key else 'No'}", file=sys.stderr)
        print("❌ Make sure your .env file is in the same directory as main.py", file=sys.stderr)
        print("❌ .env file format should be: GEMINI_API_KEY=your_actual_key_here", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Gemini API Key loaded successfully (first 10 chars): {gemini_api_key[:10]}...", file=sys.stderr)

    
    firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
    if firecrawl_api_key:
        print("✅ Firecrawl API Key also loaded", file=sys.stderr)
    else:
        print("⚠️  Firecrawl API Key not found (optional)", file=sys.stderr)
    
    # Create and run server
    try:
        server = FinancialAnalysisServer(gemini_api_key, firecrawl_api_key)
        logger.info(f"✅ Financial Analysis Server initialized successfully")
        server.run()
    except Exception as e:
        print(f"❌ Failed to start server: {str(e)}", file=sys.stderr)
        logger.error(f"Server startup error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()#!/usr/bin/env python3
