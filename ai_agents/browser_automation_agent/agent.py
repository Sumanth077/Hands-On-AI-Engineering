"""Browser automation agent powered by browser-use + LangChain.

This module wires a LangChain `ChatOpenAI` client (pointed at the Orq.ai
router) into a `browser-use` Agent. The agent receives a natural language
instruction, autonomously plans and executes browser actions, and returns a
structured summary of what it did and what it found.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from dotenv import load_dotenv

try:
    # Recent browser-use ships its own chat-model classes. Its Agent reads
    # `llm.provider`, so it requires a browser-use ChatOpenAI rather than a raw
    # LangChain model. browser-use's ChatOpenAI accepts the same
    # base_url/api_key/model arguments, so the Orq.ai routing is unchanged.
    from browser_use import Agent, ChatOpenAI
except ImportError as exc:  # pragma: no cover - surfaced to the user at runtime
    raise ImportError(
        "browser-use is not installed. Run `pip install -r requirements.txt`."
    ) from exc


load_dotenv()

ORQ_API_KEY = os.getenv("ORQ_API_KEY")
ORQ_BASE_URL = os.getenv("ORQ_BASE_URL", "https://api.orq.ai/v3/router")
# Orq.ai routes models in `provider/model` format (e.g. alibaba/qwen3.6-flash).
ORQ_MODEL = os.getenv("ORQ_MODEL", "alibaba/qwen3.6-flash")

# Hard cap on the number of browser steps a single instruction may take.
DEFAULT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "25"))


@dataclass
class AgentResult:
    """Structured result returned to the UI layer."""

    instruction: str
    success: bool
    summary: str
    visited_urls: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_markdown(self) -> str:
        """Render the result as a chat-friendly markdown block."""
        status = "✅ Completed" if self.success else "⚠️ Did not complete"
        lines = [f"**{status}**", "", self.summary.strip() or "_No summary produced._"]

        if self.visited_urls:
            lines += ["", "**Pages visited:**"]
            lines += [f"- {url}" for url in self.visited_urls[:10]]

        if self.actions:
            lines += ["", "<details><summary>Action log</summary>", ""]
            lines += [f"{i + 1}. {action}" for i, action in enumerate(self.actions[:30])]
            lines += ["", "</details>"]

        if self.error:
            lines += ["", f"**Error:** `{self.error}`"]

        return "\n".join(lines)


def get_llm() -> ChatOpenAI:
    """Build the LangChain chat model pointed at the Orq.ai router."""
    if not ORQ_API_KEY:
        raise ValueError(
            "ORQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return ChatOpenAI(
        base_url=ORQ_BASE_URL,
        api_key=ORQ_API_KEY,
        model=ORQ_MODEL,
        temperature=0.0,
    )


def _extract_history(history) -> AgentResult:
    """Best-effort extraction of useful fields from a browser-use history object.

    The browser-use API surface has shifted across versions, so each accessor is
    guarded to keep the agent resilient.
    """
    summary = ""
    visited_urls: List[str] = []
    actions: List[str] = []
    success = True

    # Final natural-language answer.
    try:
        summary = history.final_result() or ""
    except Exception:  # noqa: BLE001 - defensive across versions
        summary = ""

    # Whether the agent considered the task done.
    try:
        success = bool(history.is_done())
    except Exception:  # noqa: BLE001
        success = bool(summary)

    # Surface any step errors (e.g. a model 404 from the LLM provider) instead
    # of silently reporting "no summary".
    step_errors: List[str] = []
    try:
        step_errors = [str(e) for e in history.errors() if e]
    except Exception:  # noqa: BLE001
        step_errors = []

    # URLs the agent navigated to.
    try:
        visited_urls = [u for u in history.urls() if u]
    except Exception:  # noqa: BLE001
        visited_urls = []

    # Human-readable model "thoughts"/actions, if available.
    try:
        for thought in history.model_thoughts():
            text = getattr(thought, "next_goal", None) or str(thought)
            if text:
                actions.append(text)
    except Exception:  # noqa: BLE001
        actions = []

    # Only surface errors when the task did NOT produce an answer. browser-use
    # retries transient step failures (e.g. a model returning an empty/malformed
    # response), so a recovered error alongside a valid summary is just noise.
    error_msg: Optional[str] = None
    task_failed = not summary and not success
    if step_errors and task_failed:
        # De-duplicate identical errors (browser-use retries a lot).
        unique_errors = list(dict.fromkeys(step_errors))
        error_msg = unique_errors[-1]
        success = False

    if not summary:
        if error_msg:
            summary = "The agent could not complete the task due to an error."
        else:
            summary = (
                "The agent finished but did not return a textual summary. "
                "Check the action log for details."
            )

    # De-duplicate visited URLs while preserving order.
    seen = set()
    deduped_urls = []
    for url in visited_urls:
        if url not in seen:
            seen.add(url)
            deduped_urls.append(url)

    return AgentResult(
        instruction="",
        success=success,
        summary=summary,
        visited_urls=deduped_urls,
        actions=actions,
        error=error_msg,
    )


async def run_browser_agent(
    instruction: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_step: Optional[Callable] = None,
) -> AgentResult:
    """Run the browser-use agent for a single natural language instruction.

    Args:
        instruction: The user's natural language task.
        max_steps: Safety cap on the number of browser steps.

    Returns:
        An :class:`AgentResult` describing the outcome.
    """
    instruction = (instruction or "").strip()
    if not instruction:
        return AgentResult(
            instruction=instruction,
            success=False,
            summary="Please provide an instruction for the agent to carry out.",
        )

    try:
        llm = get_llm()
    except ValueError as exc:
        return AgentResult(
            instruction=instruction,
            success=False,
            summary="Configuration error.",
            error=str(exc),
        )

    try:
        agent = Agent(task=instruction, llm=llm, register_new_step_callback=on_step)
        history = await agent.run(max_steps=max_steps)
        result = _extract_history(history)
        result.instruction = instruction
        return result
    except Exception as exc:  # noqa: BLE001 - report any runtime failure to the UI
        return AgentResult(
            instruction=instruction,
            success=False,
            summary="The agent hit an error while browsing.",
            error=str(exc),
        )


if __name__ == "__main__":
    import asyncio

    demo_task = "Find the latest news about AI agents and summarize the top 3 stories."
    outcome = asyncio.run(run_browser_agent(demo_task))
    print(outcome.to_markdown())
