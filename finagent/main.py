from mcp.server.fastmcp import FastMCP
from financial_agents import run_financial_analysis
import json

# Create FastMCP instance
mcp = FastMCP("financial-agent")  # keep name consistent with your logs

@mcp.tool()
def analyze_stock(query: str) -> str:
    """
    Analyzes stock market data based on the query and generates insights.
    Returns valid JSON string response.
    """
    try:
        result = run_financial_analysis(query)

        if isinstance(result, (dict, list)):
            # Return JSON formatted string if result is dict or list
            return json.dumps(result, indent=2)
        elif isinstance(result, str):
            # Try to parse string as JSON
            try:
                obj = json.loads(result)
                return json.dumps(obj, indent=2)
            except Exception:
                # If not valid JSON string, wrap it safely inside JSON object
                return json.dumps({"result": result.strip()}, indent=2)
        else:
            # Convert other types to string inside JSON object
            return json.dumps({"result": str(result).strip()})

    except Exception as e:
        # Return error message in JSON format
        return json.dumps({"error": str(e)})

@mcp.tool()
def save_code(code: str) -> str:
    """
    Saves generated Python code to 'stock_analysis.py'.
    Ensures the file is valid and ready to execute.
    """
    try:
        if not code.strip():
            return json.dumps({"error": "No code provided."})
        with open("stock_analysis.py", "w", encoding="utf-8") as f:
            f.write(code)
        return json.dumps({"message": "✅ Code saved to stock_analysis.py"})
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def run_code_and_show_plot() -> str:
    """
    Runs 'stock_analysis.py' and generates any resulting plots.
    """
    try:
        with open("stock_analysis.py", "r", encoding="utf-8") as f:
            exec(f.read(), {})
        return json.dumps({"message": "✅ Code executed successfully."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Run the server locally with stdio transport
if __name__ == "__main__":
    mcp.run(transport="stdio")
