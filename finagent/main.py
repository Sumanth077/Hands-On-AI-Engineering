#!/usr/bin/env python3

import logging
import os
import sys
import tempfile
import subprocess
from dotenv import load_dotenv
from typing import Optional

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    from financial_agents import FinancialAnalysisTeam, AnalysisResult, FinancialTools
except ImportError as e:
    print(f"Import error: {e}\nInstall with: pip install mcp", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger("mcp-financial")

class FinancialAnalysisServer:
    def __init__(self, gemini_api_key: str, firecrawl_api_key: Optional[str] = None):
        self.mcp = FastMCP("financial-analyst-gemini")
        self.analysis_team = FinancialAnalysisTeam(firecrawl_api_key, gemini_api_key)
        self.setup_tools()
        
    def setup_tools(self):
        @self.mcp.tool()
        def analyze_stock(query: str) -> str:
            try:
                result: AnalysisResult = self.analysis_team.analyze(query)
                news_section = "\n### Latest News:\n" + "\n".join(
                    f"{i}. **{n.title}**\n   {n.summary[:100]}...\n" 
                    for i, n in enumerate(result.news[:3], 1)
                ) if result.news else ""
                
                return f"""# Stock Analysis - {result.query.symbol}

## AI Insights
{result.insights}

## Analysis Code
```python
{result.code}
```
{news_section}
## Recommendations
{result.recommendations}

---
*Powered by Gemini AI*
"""
            except Exception as e:
                logger.error(f"Analysis error: {e}", exc_info=True)
                return f"Analysis Error: {e}"
        
        @self.mcp.tool()
        def execute_code(code: str, timeout: int = 30) -> str:
            exec_code = f"import sys\nimport warnings\nwarnings.filterwarnings('ignore')\n\ntry:\n{chr(10).join('    ' + line for line in code.split(chr(10)))}\nexcept Exception as e:\n    print(f'Error: {{e}}')\n    sys.exit(1)"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(exec_code)
                temp_file = f.name
            
            try:
                result = subprocess.run([sys.executable, temp_file], capture_output=True, 
                                      text=True, timeout=timeout, encoding='utf-8', errors='replace')
                os.unlink(temp_file)
                output = result.stdout or "Code executed successfully!" if result.returncode == 0 else result.stderr or 'Error'
                safe_output = ''.join(ch if ord(ch) < 0x110000 else '?' for ch in output)
                status = "Success" if result.returncode == 0 else "Error"
                return f"**{status}**\n```\n{safe_output}\n```"
            except subprocess.TimeoutExpired:
                os.unlink(temp_file)
                return f"Execution timeout after {timeout}s"
            except Exception as e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return f"Execution error: {e}"
        
        @self.mcp.tool()
        def get_news(symbol: str, limit: int = 5) -> str:
            try:
                news_data = FinancialTools().get_stock_news(symbol, limit)
                if not news_data:
                    return f"No news found for {symbol}"
                
                return f"# News for {symbol}\n\n" + "\n".join(
                    f"## {i}. {item.get('title', 'Update')}\n{item.get('content', '')[:200]}{'...' if len(item.get('content', '')) > 200 else ''}\n[Read more]({item.get('url', '')})\n"
                    for i, item in enumerate(news_data[:limit], 1)
                )
            except Exception as e:
                logger.error(f"News fetch error: {e}", exc_info=True)
                return f"News fetch error: {e}"
    
    def run(self):
        self.mcp.run()

def main():
    required = ["google-generativeai", "yfinance pandas", "mcp"]
    missing = []
    
    for pkg in required:
        try:
            if pkg == "google-generativeai":
                import google.generativeai
            elif pkg == "yfinance pandas":
                import yfinance, pandas
            else:
                from mcp.server.fastmcp import FastMCP
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}\nInstall with: pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("GEMINI_API_KEY required. Get one from: https://makersuite.google.com/app/apikey", file=sys.stderr)
        sys.exit(1)
    
    try:
        server = FinancialAnalysisServer(gemini_api_key, os.getenv('FIRECRAWL_API_KEY'))
        logger.info("Financial Analysis Server initialized successfully")
        server.run()
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        logger.error(f"Server startup error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()