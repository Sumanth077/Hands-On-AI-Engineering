import asyncio
from mcp.server.fastmcp import FastMCP
from agents import final_research

# Create FastMCP instance
mcp = FastMCP("deep-researcher-agent")


@mcp.tool()
async def deep_research_tool(query: str) -> str:
    """
    Run the deep research agent on the provided query.
    
    Arg: user query
    
    Returns: The research response from the deep research agent."""
    return await final_research(query)

# run the FastMCP server
if __name__ == "__main__":
    mcp.run()

