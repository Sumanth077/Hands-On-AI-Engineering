import json
from unittest.mock import MagicMock

import duckdb
import pandas as pd

from agent.client import LinerResponse
from agent.harness import InvestigationHarness


def _fake_response(content=None, tool_calls=None, prompt_tokens=100, completion_tokens=50):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return LinerResponse(
        raw={
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    )


def _tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_harness_runs_tool_then_concludes():
    con = duckdb.connect(database=":memory:")
    df = pd.DataFrame({"category": ["Fashion", "Electronics"], "revenue": [100, 500]})
    con.register("orders", df)

    client = MagicMock()
    client.chat.side_effect = [
        _fake_response(
            tool_calls=[_tool_call("call_1", "inspect_dataset", {})],
        ),
        _fake_response(content="Fashion revenue is much lower than Electronics. Investigation complete."),
    ]

    harness = InvestigationHarness(client=client, con=con, dataframes={"orders": df}, max_steps=10, max_tool_calls=10)

    events = list(harness.investigate(objective="Why is revenue low?", depth="Standard"))

    event_types = [e.type for e in events]
    assert event_types == ["tool_call", "finished"]

    final_state = events[-1].payload["state"]
    assert final_state.is_finished is True
    assert final_state.stop_reason == "concluded"
    assert "Investigation complete" in final_state.final_report
    assert final_state.usage.model_calls == 2
    assert final_state.usage.total_tokens == 300  # two calls x (100 + 50)


def test_harness_stops_at_step_budget():
    con = duckdb.connect(database=":memory:")
    df = pd.DataFrame({"x": [1, 2, 3]})
    con.register("t", df)

    client = MagicMock()
    # Always returns another tool call, never concludes on its own.
    client.chat.return_value = _fake_response(
        tool_calls=[_tool_call("call_x", "inspect_dataset", {})]
    )

    harness = InvestigationHarness(client=client, con=con, dataframes={"t": df}, max_steps=3, max_tool_calls=100)

    events = list(harness.investigate(objective="loop forever", depth="Quick"))

    assert events[-1].type == "budget_exhausted"
    final_state = events[-1].payload["state"]
    assert final_state.stop_reason == "budget_exhausted"
    assert final_state.step_count == 3


def test_tool_calls_share_model_call_index_within_a_round():
    con = duckdb.connect(database=":memory:")
    df = pd.DataFrame({"x": [1, 2, 3]})
    con.register("t", df)

    client = MagicMock()
    client.chat.side_effect = [
        # Round 1: two parallel tool calls in a single model response.
        _fake_response(
            tool_calls=[
                _tool_call("call_1", "inspect_dataset", {}),
                _tool_call("call_2", "inspect_dataset", {}),
            ]
        ),
        # Round 2: a separate while-loop iteration, one more tool call.
        _fake_response(tool_calls=[_tool_call("call_3", "inspect_dataset", {})]),
        _fake_response(content="Done."),
    ]

    harness = InvestigationHarness(client=client, con=con, dataframes={"t": df}, max_steps=10, max_tool_calls=10)
    events = list(harness.investigate(objective="test", depth="Standard"))

    final_state = events[-1].payload["state"]
    tool_calls = final_state.tool_calls
    assert len(tool_calls) == 3

    # Calls dispatched from the same parallel batch share a model_call_index.
    assert tool_calls[0].model_call_index == tool_calls[1].model_call_index
    # A call from a later while-loop iteration gets a different one.
    assert tool_calls[2].model_call_index != tool_calls[0].model_call_index


def test_harness_save_finding_persists():
    con = duckdb.connect(database=":memory:")
    df = pd.DataFrame({"x": [1]})
    con.register("t", df)

    client = MagicMock()
    client.chat.side_effect = [
        _fake_response(
            tool_calls=[
                _tool_call(
                    "call_1",
                    "save_finding",
                    {"finding": "Revenue dropped due to Fashion discounts", "importance": "high"},
                )
            ]
        ),
        _fake_response(content="Done."),
    ]

    harness = InvestigationHarness(client=client, con=con, dataframes={"t": df}, max_steps=10, max_tool_calls=10)
    events = list(harness.investigate(objective="test", depth="Standard"))

    final_state = events[-1].payload["state"]
    assert len(final_state.findings) == 1
    assert final_state.findings[0].importance == "high"
