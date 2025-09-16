from mcp.server.fastmco import FastMCP
from financial_aagents import analyze_stock_data

mcp = FastMCP("financial_analysis_mcp")

@mcp.tool()
def analyze_stock(query: str) -> str:
    """
    Analyze stock data based on the user query.
    """
    try:
        result = analyze_stock_data(query)
        return result
    except Exception as e:
        return f"Error: {str(e)}"
    

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8000)
    