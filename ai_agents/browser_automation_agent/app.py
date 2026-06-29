"""Gradio chat UI for the Browser Automation Agent.

Type a natural language instruction (e.g. "Find the latest news about AI
agents") and the agent will autonomously navigate the web to complete it,
returning a structured summary in the chat.
"""

from __future__ import annotations

import asyncio
import queue
import threading

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


def respond(message: str, history: list):
    """Streaming Gradio handler that yields step updates then the final result."""
    step_queue: queue.Queue = queue.Queue()

    def on_step(state, output, step_num: int) -> None:
        url = getattr(state, "url", "") or ""
        msg = f"Step {step_num}: Navigating to {url}..." if url else f"Step {step_num}: Processing..."
        step_queue.put(("step", msg))

    def run_agent() -> None:
        try:
            result = asyncio.run(run_browser_agent(message, on_step=on_step))
            step_queue.put(("done", result))
        except Exception as exc:  # noqa: BLE001 - never crash the UI thread
            step_queue.put(("error", str(exc)))

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    yield "Starting browser agent..."

    while True:
        kind, value = step_queue.get()
        if kind == "step":
            yield value
        elif kind == "done":
            yield value.to_markdown()
            break
        else:
            yield f"**Error:** `{value}`"
            break


def build_demo() -> gr.ChatInterface:
    """Construct and return the Gradio ChatInterface for the browser automation agent."""
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
