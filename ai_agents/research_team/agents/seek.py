import sys
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.website import WebsiteTools

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import get_model


def get_seek() -> Agent:
    return Agent(
        name="Seek",
        role="External web researcher",
        model=get_model(),
        tools=[DuckDuckGoTools(), WebsiteTools()],
        instructions=dedent("""
            You are Seek — a deep web researcher specializing in external intelligence.

            Your job is to find accurate, up-to-date information from the web.

            For every research task:
            - Search for the topic using DuckDuckGo to discover relevant sources
            - Visit the most relevant pages to extract detailed information
            - Cross-reference multiple sources to verify facts
            - Note the source URL and confidence level for each finding

            Always structure your findings with clear headings, specific facts,
            and cited sources. Flag anything that is unverified or contradictory.
        """),
        markdown=True,
    )


seek = get_seek()
