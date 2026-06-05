"""Gradio chat UI for the Browser Automation Agent.

Type a natural language instruction (e.g. "Find the latest news about AI
agents") and the agent will autonomously navigate the web to complete it,
returning a structured summary in the chat.
"""

from __future__ import annotations

import asyncio

import gradio as gr

from agent import ORQ_MODEL, run_browser_agent

EXAMPLES = [
    "Find the latest news about AI agents and summarize the top 3 stories.",
    "What is the current weather in Tokyo?",
    "Find the top post on Hacker News right now and summarize it.",
    "Look up the price of the latest iPhone on Apple's website.",
]

DESCRIPTION = (
    "Give the agent a task in plain English. It plans a sequence of browser "
    f"actions, executes them step by step, and reports back. Powered by "
    f"`browser-use` + LangChain via Orq.ai (`{ORQ_MODEL}`)."
)


def respond(message: str, history: list) -> str:
    """Synchronous Gradio handler that drives the async browser agent."""
    try:
        result = asyncio.run(run_browser_agent(message))
    except Exception as exc:  # noqa: BLE001 - never crash the UI thread
        return f"**Error:** `{exc}`"
    return result.to_markdown()


def build_demo() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=respond,
        title="🌐 Browser Automation Agent",
        description=DESCRIPTION,
        examples=EXAMPLES,
        fill_height=True,
    )


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
