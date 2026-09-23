import json
import os
import uuid
from collections.abc import Generator

import ollama
from dotenv import load_dotenv

from agent.llm_client import (
    _configured_base_url,
    _configured_model,
    assert_local_only,
    get_client,
    health_check as llm_health_check,
)
from agent.prompts import SYSTEM_PROMPT, build_kickoff_message
from agent.state import InvestigationResult, InvestigationState
from agent.tools import AgentTools
from memory.vectorai_client import health_check as vectorai_health_check
from simulator.history import History
from simulator.machine import Machine

load_dotenv()

DEFAULT_MAX_AGENT_ROUNDS = 8
DEFAULT_MAX_TOOL_CALLS = 20


def _configured_max_rounds() -> int:
    return int(os.getenv("MAX_AGENT_ROUNDS", DEFAULT_MAX_AGENT_ROUNDS))


def _configured_max_tool_calls() -> int:
    return int(os.getenv("MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS))


def _unreachable_result() -> dict:
    return InvestigationResult(
        likely_cause="Undetermined — investigation could not start",
        confidence=0.0,
        recommendation="Manual inspection recommended.",
        supporting_evidence=[],
        retrieved_incident_ids=[],
        manual_sources=[],
    ).model_dump()


def _fallback_result(state: InvestigationState) -> dict:
    return InvestigationResult(
        likely_cause="Undetermined — investigation did not reach a conclusion",
        confidence=0.0,
        recommendation="Manual inspection recommended.",
        supporting_evidence=list(state.findings),
        retrieved_incident_ids=[
            item["id"] for item in state.retrieved_incidents if item.get("id")
        ],
        manual_sources=[
            item["section_title"]
            for item in state.retrieved_manual_chunks
            if item.get("section_title")
        ],
    ).model_dump()


def run_investigation(
    machine: Machine,
    history: History,
    objective: str,
    max_rounds: int | None = None,
    max_tool_calls: int | None = None,
) -> Generator[InvestigationState, None, None]:
    if max_rounds is None:
        max_rounds = _configured_max_rounds()
    if max_tool_calls is None:
        max_tool_calls = _configured_max_tool_calls()

    investigation_id = str(uuid.uuid4())
    current_readings = machine.get_current_readings().model_dump(mode="json")

    llm_status = llm_health_check()
    if not llm_status.get("reachable") or not llm_status.get("model_available"):
        state = InvestigationState(
            investigation_id=investigation_id,
            objective=objective,
            current_readings=current_readings,
        )
        if not llm_status.get("reachable"):
            state.errors.append("Ollama is not reachable — is it running?")
        else:
            model = _configured_model()
            state.errors.append(
                f"model {model} is not pulled — run `ollama pull {model}`."
            )
        state.final_result = _unreachable_result()
        yield state
        return

    vectorai_status = vectorai_health_check()
    if not vectorai_status.get("reachable"):
        state = InvestigationState(
            investigation_id=investigation_id,
            objective=objective,
            current_readings=current_readings,
        )
        state.errors.append(
            "Actian VectorAI DB is not reachable — is the container running "
            "(docker compose up -d)?"
        )
        state.final_result = _unreachable_result()
        yield state
        return

    state = InvestigationState(
        investigation_id=investigation_id,
        objective=objective,
        current_readings=current_readings,
    )
    tools = AgentTools(machine=machine, history=history, state=state)
    dispatch = {fn.__name__: fn for fn in tools.tool_list()}

    yield state

    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_kickoff_message(objective, state.current_readings)},
    ]

    base_url = _configured_base_url()
    model = _configured_model()
    assert_local_only(base_url, model)
    client = get_client()

    executed_signatures: set[str] = set()
    total_tool_calls = 0
    tool_call_limit_reached = False
    nudge_used = False

    def process_tool_calls(tool_calls):
        nonlocal total_tool_calls, tool_call_limit_reached
        for call in tool_calls:
            if total_tool_calls >= max_tool_calls:
                state.errors.append("max_tool_calls_reached")
                tool_call_limit_reached = True
                break

            name = call.function.name
            args = dict(call.function.arguments or {})
            # args can contain lists (e.g. finish_investigation's
            # supporting_evidence), which aren't hashable, so use a stable
            # JSON string as the canonical signature instead of a tuple.
            signature = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"

            if signature in executed_signatures:
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": "Skipped: identical call already made this investigation.",
                    }
                )
                total_tool_calls += 1
                yield state
                continue

            executed_signatures.add(signature)
            fn = dispatch.get(name)
            try:
                if fn is None:
                    raise KeyError(f"Unknown tool '{name}'")
                result = fn(**args)
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})
            except Exception as exc:
                state.errors.append(f"tool_error:{name}:{exc}")
                messages.append(
                    {"role": "tool", "tool_name": name, "content": f"Tool error: {exc}"}
                )

            total_tool_calls += 1
            yield state

    def chat_or_none():
        """Call the model; on a mid-investigation connection loss, record a
        clear error and return None instead of letting the exception
        propagate out of run_investigation() and crash the caller (e.g. the
        Gradio UI)."""
        try:
            return client.chat(
                model=model, messages=messages, tools=list(dispatch.values()), think=False
            )
        except (ConnectionError, ollama.RequestError) as exc:
            state.errors.append(f"llm_connection_lost_mid_investigation: {exc}")
            return None

    for _round in range(max_rounds):
        response = chat_or_none()
        if response is None:
            break
        messages.append(response.message)

        tool_calls = response.message.tool_calls
        if not tool_calls:
            content = response.message.content or ""
            state.errors.append(f"model_ended_without_tool_call: {content[:300]}")

            if not nudge_used:
                nudge_used = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You didn't call a tool. If your investigation is complete, "
                            "call finish_investigation now summarizing your conclusion "
                            "using the evidence you've already gathered. If you still "
                            "need more information, call the appropriate tool."
                        ),
                    }
                )
                response = chat_or_none()
                if response is None:
                    break
                messages.append(response.message)
                tool_calls = response.message.tool_calls
                if not tool_calls:
                    content = response.message.content or ""
                    state.errors.append(f"model_ended_without_tool_call: {content[:300]}")

            if not tool_calls:
                state.errors.append("model_ended_without_finish_investigation")
                break

        yield from process_tool_calls(tool_calls)

        if tool_call_limit_reached:
            break

        if state.final_result is not None:
            break
    else:
        if state.final_result is None:
            state.errors.append("max_rounds_reached")

    if state.final_result is None:
        state.final_result = _fallback_result(state)
        yield state
