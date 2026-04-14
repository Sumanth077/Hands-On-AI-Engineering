from mcp.server.fastmcp import FastMCP
from financial_agents import run_financial_analysis


# Create FastMCP instance
mcp = FastMCP("financial-analyst")


@mcp.tool()
def analyze_stock(query: str) -> str:
    """
    Performs a natural language stock market analysis based on the query.
    Returns a human-readable analysis summary and recommendations.

    The query is a string including stock ticker symbol (e.g., TSLA, AAPL),
    time period (1d, 1mo, 1y), and analysis request (e.g., analyze, compare).

    Args:
        query (str): The stock market query.

    Returns:
        str: Plain text stock market analysis report.
    """
    try:
        result = run_financial_analysis(query)
        return result
    except Exception as e:
        return f"Error: {e}"


# Removed save_code and run_code_and_show_plot tools because analysis does not return code


# Run the server locally
if __name__ == "__main__":
    mcp.run(transport='stdio')
