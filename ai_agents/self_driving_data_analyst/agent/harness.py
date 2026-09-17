"""
The investigation loop itself, this is the actual project.

Understand -> choose what to investigate -> run an analysis -> observe ->
decide what's next -> repeat -> produce final findings. The result of each
tool call determines the next one; nothing here is a fixed pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import duckdb
import pandas as pd

from agent.client import LinerClient
from agent.loop_detector import is_looping
from agent.prompts import LOOP_WARNING, SYSTEM_PROMPT
from agent.state import InvestigationState
from agent.tool_schemas import TOOLS
from tools.charts import create_chart
from tools.dataset import inspect_dataset
from tools.findings import save_finding
from tools.python_tool import run_python
from tools.sql import run_sql

DEPTH_TO_REASONING_EFFORT = {
    "Quick": "low",
    "Standard": "medium",
    "Deep": "high",
}


@dataclass
class InvestigationEvent:
    type: str  # "tool_call" | "loop_warning" | "finished" | "budget_exhausted"
    payload: dict[str, Any] = field(default_factory=dict)


def _prune_messages(messages: list[dict[str, Any]], context_window_turns: int) -> list[dict[str, Any]]:
    """Keeps messages[0] (system) and messages[1] (the original objective)
    always, plus only the most recent `context_window_turns` assistant/tool
    exchanges. Only cuts at assistant-message boundaries so a tool_call_id
    pairing between an assistant's tool_calls and the following tool results
    never gets split."""
    if len(messages) <= 2:
        return messages

    head, tail = messages[:2], messages[2:]
    turn_starts = [i for i, m in enumerate(tail) if m.get("role") == "assistant"]

    if len(turn_starts) <= context_window_turns:
        return messages

    cutoff = turn_starts[-context_window_turns]
    return head + tail[cutoff:]


class InvestigationHarness:
    def __init__(
        self,
        client: LinerClient,
        con: duckdb.DuckDBPyConnection,
        dataframes: dict[str, pd.DataFrame],
        max_steps: int = 25,
        max_tool_calls: int = 40,
        context_window_turns: int = 6,
    ):
        self.client = client
        self.con = con
        self.dataframes = dataframes
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.context_window_turns = context_window_turns
        self.charts: list[Any] = []  # Plotly figures, kept out of the Pydantic state on purpose

    def _dispatch_tool(self, name: str, arguments: dict, state: InvestigationState) -> dict:
        if name == "inspect_dataset":
            return inspect_dataset(self.con)
        if name == "run_sql":
            return run_sql(self.con, arguments.get("query", ""))
        if name == "run_python":
            return run_python(arguments.get("code", ""), self.dataframes)
        if name == "create_chart":
            result = create_chart(
                rows=arguments.get("rows", []),
                chart_type=arguments.get("chart_type", "bar"),
                x=arguments.get("x"),
                y=arguments.get("y"),
                color=arguments.get("color"),
                title=arguments.get("title"),
            )
            if result.get("success") and "figure" in result:
                self.charts.append(result.pop("figure"))
                result["chart_index"] = len(self.charts) - 1
            return result
        if name == "save_finding":
            return save_finding(
                state,
                finding=arguments.get("finding", ""),
                importance=arguments.get("importance", "medium"),
                evidence=arguments.get("evidence"),
            )
        if name == "update_hypothesis":
            state.upsert_hypothesis(
                hypothesis=arguments.get("hypothesis", ""),
                status=arguments.get("status", "untested"),
                confidence=arguments.get("confidence"),
                evidence=arguments.get("evidence"),
            )
            return {"success": True}
        if name == "finish_investigation":
            state.root_cause = arguments.get("root_cause")
            state.confidence = arguments.get("confidence")
            state.recommended_actions = arguments.get("recommended_actions") or []
            state.final_report = arguments.get("summary_markdown", "")
            return {"success": True}
        return {"success": False, "error": f"Unknown tool '{name}'"}

    def investigate(self, objective: str, depth: str = "Standard") -> Iterator[InvestigationEvent]:
        reasoning_effort = DEPTH_TO_REASONING_EFFORT.get(depth, "medium")
        state = InvestigationState(objective=objective, reasoning_effort=reasoning_effort)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(state_summary=state.summary_for_prompt())},
            {"role": "user", "content": objective},
        ]

        while True:
            if state.step_count >= self.max_steps or len(state.tool_calls) >= self.max_tool_calls:
                state.is_finished = True
                state.stop_reason = "budget_exhausted"
                state.final_report = (
                    "Investigation stopped after reaching the step/tool-call budget "
                    "before the agent concluded on its own. Findings and hypotheses "
                    "gathered so far are included below."
                )
                yield InvestigationEvent("budget_exhausted", {"state": state})
                return

            # Keep the system message's state summary current every turn.
            messages[0]["content"] = SYSTEM_PROMPT.format(state_summary=state.summary_for_prompt())
            messages = _prune_messages(messages, self.context_window_turns)

            response = self.client.chat(
                messages=messages,
                tools=TOOLS,
                reasoning_effort=reasoning_effort,
            )

            state.step_count += 1
            state.usage.model_calls += 1
            state.usage.prompt_tokens += response.prompt_tokens
            state.usage.completion_tokens += response.completion_tokens
            state.usage.cached_tokens += response.cached_tokens

            if not response.tool_calls:
                state.is_finished = True
                state.stop_reason = "concluded"
                state.final_report = response.content or "(no content returned)"
                yield InvestigationEvent("finished", {"state": state})
                return

            messages.append(response.message)

            for tool_call in response.tool_calls:
                fn_name = tool_call["function"]["name"]
                try:
                    fn_args = json.loads(tool_call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                result = self._dispatch_tool(fn_name, fn_args, state)
                is_error = isinstance(result, dict) and result.get("success") is False

                record = state.add_tool_call(
                    tool_name=fn_name,
                    arguments=fn_args,
                    result_summary=json.dumps(result, default=str),
                    model_call_index=state.step_count,
                    is_error=is_error,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, default=str),
                    }
                )

                yield InvestigationEvent("tool_call", {"record": record, "state": state})

                if fn_name == "finish_investigation":
                    state.is_finished = True
                    state.stop_reason = "concluded"
                    yield InvestigationEvent("finished", {"state": state})
                    return

            if is_looping(state):
                messages.append({"role": "user", "content": LOOP_WARNING})
                yield InvestigationEvent("loop_warning", {"state": state})
