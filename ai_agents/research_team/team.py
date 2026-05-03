"""
Research Team
=============
Seek + Scout working together on deep research tasks.

The team leader coordinates two agents:
- Seek: External web researcher using DuckDuckGo and WebsiteTools
- Scout: Internal knowledge navigator using local files

Usage:
    uv run python app.py
"""

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from textwrap import dedent

from agno.team.team import Team

from agents.seek import seek
from agents.scout import scout
from models import get_model

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


def _has_knowledge() -> bool:
    """Return True if the knowledge directory has at least one real file."""
    return any(
        f for f in KNOWLEDGE_DIR.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )


def build_instructions() -> str:
    if _has_knowledge():
        scout_note = (
            "- Scout: Internal knowledge navigator. Use Scout to search the "
            "knowledge directory for any internal documents relevant to the query."
        )
        workflow = (
            "1. Delegate external research to Seek and internal research to Scout\n"
            "        2. Wait for both to respond\n"
            "        3. Synthesise all findings into the final report"
        )
    else:
        scout_note = (
            "- Scout: Internal knowledge navigator. "
            "The knowledge base is currently empty — do NOT delegate to Scout."
        )
        workflow = (
            "1. Delegate the full research task to Seek\n"
            "        2. Synthesise Seek's findings into the final report"
        )

    return dedent(f"""
        You lead a research team with two specialists:
        - Seek: Deep web researcher. Use for external research, company analysis,
          people research, and topic deep-dives.
        {scout_note}

        For every research task:
        {workflow}
        4. Cross-reference findings to identify patterns and contradictions

        Always produce a structured report with:
        - Executive Summary
        - Key Findings (organized by dimension)
        - Sources and confidence levels
        - Open questions and recommended next steps

        Do not narrate your process or announce tool calls. Output only the
        final structured report.
    """)


research_team = Team(
    id="research-team",
    name="Research Team",
    model=get_model(),
    members=[seek, scout],
    instructions=build_instructions(),
    show_members_responses=False,
    markdown=True,
    add_datetime_to_context=True,
)
