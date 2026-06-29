"""
Google ADK agent definition for the Hotel Finder Agent.

Uses LiteLlm to route through Orq.ai's qwen3.6-flash and
McpToolset to connect to the Trivago MCP server.

Run with: uv run adk web
"""

from datetime import date

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StreamableHTTPConnectionParams

TRIVAGO_MCP_URL = "https://mcp.trivago.com/mcp"

INSTRUCTION = f"""You are a helpful hotel search assistant powered by Trivago.

Today's date is {date.today().strftime("%B %d, %Y")}.

Help users find hotels based on natural language queries. Extract:
- Location (city, region, or country)
- Check-in and check-out dates
- Number of adults and rooms
- Price range, star rating, and amenities if mentioned

Tool usage:
1. Use trivago-search-suggestions first to resolve a location name to the correct ID or coordinates
2. Use trivago-accommodation-search for the main hotel search by location and dates
3. Use trivago-accommodation-radius-search to search near specific GPS coordinates

When presenting results:
- Show hotel name, star rating, price per night, and review score
- Include a booking link where available
- Highlight standout amenities (pool, breakfast, free cancellation, parking)
- Present the top 3-5 options clearly using markdown

If the user has not provided check-in and check-out dates, ask them before searching.
If they give only a check-in date, default to a one-night stay.
"""

root_agent = Agent(
    name="hotel_finder_agent",
    model=LiteLlm(model="openai/alibaba/qwen3.6-flash"),
    description="Finds hotels on Trivago based on natural language queries for location, dates, guests, price, and amenities.",
    instruction=INSTRUCTION,
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=TRIVAGO_MCP_URL,
            )
        )
    ],
)
