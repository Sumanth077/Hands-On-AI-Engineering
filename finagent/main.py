#!/usr/bin/env python3

import logging
import os
import sys
import tempfile
import subprocess
from dotenv import load_dotenv
from typing import Optional

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-financial")

# Load environment variables
load_dotenv()

# Import required modules with error handling
try:
    from mcp.server.fastmcp import FastMCP
    from financial_agents import FinancialAnalysisTeam, AnalysisResult, FinancialTools
except ImportError as e:
    print(f"Import error: {e}\nInstall with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Create the FastMCP instance
mcp = FastMCP("financial-analyst-gemini")

# Initialize API keys from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

if not GEMINI_API_KEY:
    print(
        "ERROR: GEMINI_API_KEY environment variable is required.\n"
        "Get your API key from: https://makersuite.google.com/app/apikey\n"
        "Set it with: export GEMINI_API_KEY='your-key-here'",
        file=sys.stderr
    )
    sys.exit(1)

if not FIRECRAWL_API_KEY:
    logger.info("FIRECRAWL_API_KEY not set (optional)")

# Instantiate the analysis team object
analysis_team = FinancialAnalysisTeam(FIRECRAWL_API_KEY, GEMINI_API_KEY)


@mcp.tool()
def analyze_stock(query: str) -> str:
    """
    Comprehensive stock analysis using AI agents.
    Args:
        query: Natural language query (e.g., "analyze AAPL" or "technical analysis of MSFT over 6 months")
    Returns:
        Formatted analysis report with insights, code, and recommendations
    """
    try:
        result: AnalysisResult = analysis_team.analyze(query)
        news_section = ""
        if result.news:
            news_section = "\n### Latest News:\n"
            for i, news_item in enumerate(result.news[:3], 1):
                summary = str(news_item.summary) if news_item.summary else ""
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                url_part = f"\n [Read more]({news_item.url})" if news_item.url else ""
                news_section += f"{i}. **{news_item.title}**\n {summary}{url_part}\n\n"

        report = f"""# Stock Analysis - {result.query.symbol}

## Market Insights

{result.insights}

## Analysis Code

{result.code}


{news_section}

## Investment Recommendations

{result.recommendations}

---

*Analysis powered by Gemini AI | Period: {result.query.time_period}*
"""
        return report

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return (
            f"# Analysis Error\n\n"
            f"An unexpected error occurred while analyzing the query.\n\n"
            f"**Error details:** {str(e)}\n\n"
            f"Please check:\n"
            f"- The ticker symbol is valid\n"
            f"- Your API keys are configured correctly\n"
            f"- You have an active internet connection\n\n"
            f"Check server logs for full traceback."
        )


@mcp.tool()
def execute_code(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in a sandboxed environment.
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
    Returns:
        Execution output or error message
    """
    exec_code = (
        "import sys\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "try:\n"
    )

    for line in code.split('\n'):
        exec_code += f"    {line}\n"

    exec_code += (
        "except Exception as e:\n"
        "    print(f'Execution Error: {e}', file=sys.stderr)\n"
        "    import traceback\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(exec_code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )

        os.unlink(temp_file)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            error_msg = stderr or stdout or 'Unknown error'
            safe_output = ''.join(ch if ord(ch) < 0x110000 else '?' for ch in error_msg)
            return f"**Execution Failed**\n``````"
        else:
            output = stdout or 'Code executed successfully!'
            safe_output = ''.join(ch if ord(ch) < 0x110000 else '?' for ch in output)
            return f"**Success**\n``````"

    except subprocess.TimeoutExpired:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return f"**Timeout Error**\nExecution exceeded {timeout} seconds limit."

    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        logger.exception("Code execution error")
        return f"**Execution Error**\n``````"


@mcp.tool()
def get_news(symbol: str, limit: int = 5) -> str:
    """
    Fetch latest news for a stock symbol.
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
        limit: Maximum number of news items to return (default: 5)
    Returns:
        Formatted news items with titles, summaries, and links
    """
    try:
        news_data = FinancialTools().get_stock_news(symbol, limit)

        if not news_data:
            return f"# No News Found\n\nNo recent news articles found for {symbol.upper()}."

        news_report = f"# Latest News for {symbol.upper()}\n\n"

        for i, item in enumerate(news_data[:limit], 1):
            title = str(item.get('title', 'Market Update'))
            content = str(item.get('content', ''))
            url = str(item.get('url', ''))

            if len(content) > 200:
                content = content[:200] + '...'

            news_report += f"## {i}. {title}\n"
            if content:
                news_report += f"{content}\n"
            if url:
                news_report += f"[Read full article]({url})\n"
            news_report += "\n"

        return news_report

    except Exception as e:
        logger.error(f"News fetch error: {e}", exc_info=True)
        return f"# Error Fetching News\n\nFailed to retrieve news for {symbol}: {str(e)}"


def check_dependencies() -> bool:
    """Check if all required packages are installed"""
    required_packages = {
        "google-generativeai": "google.generativeai",
        "yfinance": "yfinance",
        "pandas": "pandas",
        "mcp": "mcp.server.fastmcp"
    }

    missing = []
    for pkg_name, import_path in required_packages.items():
        try:
            parts = import_path.split('.')
            module = __import__(parts[0])
            for part in parts[1:]:
                module = getattr(module, part)
        except (ImportError, AttributeError):
            missing.append(pkg_name)

    if missing:
        print(
            f"Missing required packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}",
            file=sys.stderr
        )
        return False

    return True


if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)

    logger.info("Starting Financial Analysis MCP server...")
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error starting server: {e}", exc_info=True)
        print(f"Failed to start server: {e}", file=sys.stderr)
        sys.exit(1)
