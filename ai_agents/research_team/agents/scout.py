import sys
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.tools.file import FileTools

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import get_model

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"


def get_scout() -> Agent:
    return Agent(
        name="Scout",
        role="Internal knowledge navigator",
        model=get_model(),
        tools=[FileTools(base_dir=KNOWLEDGE_DIR)],
        instructions=dedent("""
            You are Scout — an internal knowledge navigator specializing in
            finding information from local documents and files.

            Your job is to search the knowledge directory for relevant documents
            and extract useful information from them.

            For every research task:
            - List available files in the knowledge directory
            - Read files that are relevant to the research question
            - Extract key facts, data points, and context
            - Note which file each finding came from

            If no relevant files are found, clearly state that no internal
            knowledge is available on the topic.
        """),
        markdown=True,
    )


scout = get_scout()
