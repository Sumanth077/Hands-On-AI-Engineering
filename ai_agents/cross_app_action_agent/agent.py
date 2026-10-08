"""LangGraph workflow for approved cross-application actions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from mcp_client import LinerActionsClient


SYSTEM_PROMPT = """You are a careful cross-app action agent.
You work only through the Liner Actions MCP tools supplied to you.
For every request, first search for an Action that can complete the next step.
If Liner says an account is missing, call connect_account, show the returned link,
and wait for the user to say they have finished. Then call complete_connection and continue.
Use one tool call at a time. Never claim that a message, issue, page, or other change
happened until execute_action returns a successful result. Explain what completed and
what still needs user action in your final response.
"""


class ActionState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    tools: list[dict[str, Any]]
    approved_call: dict[str, Any]
    activity: list[dict[str, Any]]


def needs_approval(tool_call: dict[str, Any]) -> bool:
    """Require review for the tool that performs an external Action.

    Liner's execute_action can represent a read or a write. Requiring approval for
    every execution avoids guessing a provider operation's impact from its name.
    """
    return tool_call.get("name") == "execute_action"


def build_graph(client: LinerActionsClient, model_name: str, base_url: str, api_key: str):
    """Build a graph whose tools come from the signed-in Liner MCP session."""
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        max_tokens=384,
        timeout=60,
        max_retries=2,
    )

    def call_model(state: ActionState) -> dict[str, Any]:
        bound_model = model.bind_tools(state["tools"])
        response = bound_model.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def route_after_model(state: ActionState) -> Literal["review", "execute", "end"]:
        message = state["messages"][-1]
        calls = getattr(message, "tool_calls", [])
        if not calls:
            return "end"
        return "review" if needs_approval(calls[0]) else "execute"

    def review_action(state: ActionState) -> dict[str, Any]:
        call = state["messages"][-1].tool_calls[0]
        decision = interrupt(
            {
                "kind": "action_approval",
                "tool_name": call["name"],
                "arguments": call["args"],
                "message": "Approve this Liner Action?",
            }
        )
        if decision.get("decision") == "approve":
            return {"approved_call": call}
        rejection = ToolMessage(
            content="The user rejected this external action. Explain the outcome and offer a revision.",
            tool_call_id=call["id"],
        )
        return {"messages": [rejection], "activity": [{"status": "rejected", "call": call}]}

    def route_after_review(state: ActionState) -> Literal["execute", "agent"]:
        return "execute" if state.get("approved_call") else "agent"

    def execute_tool(state: ActionState) -> dict[str, Any]:
        message = state["messages"][-1]
        call = state.get("approved_call") or message.tool_calls[0]
        try:
            output = client.call_sync(call["name"], call["args"])
            content = client.format_result(output)
            status = "completed"
        except Exception as error:  # Surface remote errors to the model and user.
            content = f"Action failed: {error}"
            status = "failed"
        tool_message = ToolMessage(content=content, tool_call_id=call["id"])
        return {
            "messages": [tool_message],
            "approved_call": {},
            "activity": [{"status": status, "call": call, "result": content}],
        }

    graph = StateGraph(ActionState)
    graph.add_node("agent", call_model)
    graph.add_node("review", review_action)
    graph.add_node("execute", execute_tool)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_model,
        {"review": "review", "execute": "execute", "end": END},
    )
    graph.add_conditional_edges("review", route_after_review, {"execute": "execute", "agent": "agent"})
    graph.add_edge("execute", "agent")
    return graph.compile(checkpointer=MemorySaver())


def user_message(text: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Create the first graph update for a chat submission."""
    return {"messages": [HumanMessage(content=text)], "tools": tools}

