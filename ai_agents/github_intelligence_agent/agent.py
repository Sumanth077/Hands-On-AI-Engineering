"""
GitHub Intelligence Agent
=========================
Research anything on GitHub through natural language.
Powered by Gemini 3 Flash + GitHub MCP Server via Haystack.
"""

import os
from textwrap import dedent

from haystack.components.agents import Agent
from haystack.tools import SearchableToolset
from haystack.utils import Secret
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
MODEL_ID = "gemini-3-flash-preview"

SYSTEM_PROMPT = dedent("""
    You are a GitHub Intelligence Agent with access to GitHub's full API through MCP tools.

    You can help with:
    - Searching and discovering repositories, users, code, issues, and pull requests
    - Profiling GitHub users and their most notable projects
    - Summarising open issues, pull requests, and releases for any repository
    - Analysing trends across GitHub (top repos, active projects, popular topics)
    - Exploring codebases and understanding project structure and activity

    When given a task:
    1. Use search_tools to dynamically find the right tools for the job
    2. Call the appropriate tools with precise, well-formed parameters
    3. Synthesise the results into a clear, well-structured response

    If a tool call fails, analyse the cause, adapt your approach, and retry with
    a different strategy. Persist until the task is complete.

    Do not narrate your process. Output only the final answer.
""")


def build_agent(github_pat: str, gemini_api_key: str, streaming_callback=None) -> Agent:
    """Build and return a GitHub Intelligence Agent.

    Args:
        github_pat: GitHub Personal Access Token (read-only permissions are enough)
        gemini_api_key: Google AI Gemini API key
        streaming_callback: Optional callable for streaming chunks to the UI
    """
    os.environ["GITHUB_PAT"] = github_pat
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    mcp_server_info = StreamableHttpServerInfo(
        url=GITHUB_MCP_URL,
        token=Secret.from_env_var("GITHUB_PAT"),
    )

    toolset = MCPToolset(server_info=mcp_server_info)
    searchable_toolset = SearchableToolset(catalog=toolset)

    generator = GoogleGenAIChatGenerator(model=MODEL_ID)

    agent = Agent(
        system_prompt=SYSTEM_PROMPT,
        chat_generator=generator,
        tools=searchable_toolset,
        streaming_callback=streaming_callback,
    )

    return agent
