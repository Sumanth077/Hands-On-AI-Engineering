from types import SimpleNamespace

import agent.harness as harness_module
from agent.tools import AgentTools
from simulator.machine import Machine


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _response(tool_calls, content=""):
    return SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    )


class ScriptedChat:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def __call__(self, model, messages, tools, think=False):
        self.call_count += 1
        if not self._responses:
            raise AssertionError(
                f"chat() called more times than scripted (call #{self.call_count})"
            )
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _patch_healthy_services(monkeypatch):
    monkeypatch.setattr(
        harness_module,
        "llm_health_check",
        lambda: {"reachable": True, "model_available": True, "local_models": []},
    )
    monkeypatch.setattr(
        harness_module, "vectorai_health_check", lambda: {"reachable": True}
    )


def _patch_chat(monkeypatch, responses):
    scripted = ScriptedChat(responses)
    fake_client = SimpleNamespace(chat=scripted)
    monkeypatch.setattr(harness_module, "get_client", lambda: fake_client)
    return scripted


def _patch_search_manual(monkeypatch, results=None):
    if results is None:
        results = [
            {
                "id": "chunk-1",
                "score": 0.9,
                "text": "Bearing guidance text.",
                "section_title": "Bearing Troubleshooting",
                "source_filename": "troubleshooting_guide.md",
                "component": "bearing",
            }
        ]
    call_log: list[dict] = []

    def search_manual(self, query, top_k=3):
        call_log.append({"query": query, "top_k": top_k})
        self.state.retrieved_manual_chunks.extend(results)
        return results

    monkeypatch.setattr(AgentTools, "search_manual", search_manual)
    return call_log


def _new_machine_and_history():
    machine = Machine(seed=1)
    return machine, machine.history


def test_happy_path_finishes_and_stops_after_finish_investigation(monkeypatch):
    _patch_healthy_services(monkeypatch)
    manual_log = _patch_search_manual(monkeypatch)

    final_args = dict(
        likely_cause="worn bearing",
        confidence=0.8,
        recommendation="Inspect the bearing assembly.",
        supporting_evidence=["vibration elevated"],
        retrieved_incident_ids=[],
        manual_sources=["Bearing Troubleshooting"],
    )

    responses = [
        _response([_tool_call("get_current_readings", {})]),
        _response([_tool_call("search_manual", {"query": "bearing", "top_k": 3})]),
        _response([_tool_call("finish_investigation", final_args)]),
    ]
    scripted = _patch_chat(monkeypatch, responses)

    machine, history = _new_machine_and_history()

    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    final_state = states[-1]
    assert final_state.final_result == final_args
    assert scripted.call_count == 3
    assert len(manual_log) == 1


def test_duplicate_call_detection_does_not_re_execute_underlying_tool(monkeypatch):
    _patch_healthy_services(monkeypatch)
    manual_log = _patch_search_manual(monkeypatch)

    duplicate_call = _tool_call("search_manual", {"query": "bearing", "top_k": 3})
    final_args = dict(
        likely_cause="worn bearing",
        confidence=0.7,
        recommendation="Inspect the bearing assembly.",
        supporting_evidence=[],
        retrieved_incident_ids=[],
        manual_sources=[],
    )

    responses = [
        _response([duplicate_call]),
        _response([duplicate_call]),
        _response([_tool_call("finish_investigation", final_args)]),
    ]
    scripted = _patch_chat(monkeypatch, responses)

    machine, history = _new_machine_and_history()

    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    final_state = states[-1]
    assert final_state.final_result == final_args
    assert scripted.call_count == 3
    # The underlying AgentTools.search_manual must have run only once, even
    # though the model requested the identical call twice.
    assert len(manual_log) == 1


def test_max_rounds_fallback_when_model_never_finishes(monkeypatch):
    _patch_healthy_services(monkeypatch)

    responses = [
        _response([_tool_call("save_finding", {"finding": "f1", "evidence": "e1"})]),
        _response([_tool_call("save_finding", {"finding": "f2", "evidence": "e2"})]),
    ]
    scripted = _patch_chat(monkeypatch, responses)

    machine, history = _new_machine_and_history()

    states = list(
        harness_module.run_investigation(
            machine, history, "Investigate bearing fault", max_rounds=2
        )
    )

    final_state = states[-1]
    assert scripted.call_count == 2
    assert "max_rounds_reached" in final_state.errors
    assert final_state.final_result is not None
    assert final_state.final_result["confidence"] == 0.0
    assert final_state.final_result["likely_cause"].startswith("Undetermined")


def test_tool_exception_is_recorded_and_loop_continues(monkeypatch):
    _patch_healthy_services(monkeypatch)

    def search_past_incidents(self, query, top_k=3):
        raise RuntimeError("boom")

    monkeypatch.setattr(AgentTools, "search_past_incidents", search_past_incidents)

    final_args = dict(
        likely_cause="worn bearing",
        confidence=0.6,
        recommendation="Inspect the bearing assembly.",
        supporting_evidence=[],
        retrieved_incident_ids=[],
        manual_sources=[],
    )

    responses = [
        _response([_tool_call("search_past_incidents", {"query": "test", "top_k": 3})]),
        _response([_tool_call("finish_investigation", final_args)]),
    ]
    scripted = _patch_chat(monkeypatch, responses)

    machine, history = _new_machine_and_history()

    # If the tool exception propagated out of run_investigation instead of
    # being recorded and handled, this call would raise and fail the test.
    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    final_state = states[-1]
    assert scripted.call_count == 2
    assert any(
        error.startswith("tool_error:search_past_incidents") for error in final_state.errors
    )
    assert final_state.final_result == final_args


def test_nudge_recovers_when_model_calls_tool_on_retry(monkeypatch):
    _patch_healthy_services(monkeypatch)

    final_args = dict(
        likely_cause="worn bearing",
        confidence=0.75,
        recommendation="Inspect the bearing assembly.",
        supporting_evidence=[],
        retrieved_incident_ids=[],
        manual_sources=[],
    )

    responses = [
        _response([], content="I think it might be a worn bearing, let me think..."),
        _response([_tool_call("finish_investigation", final_args)]),
    ]
    scripted = ScriptedChat(responses)
    captured_messages: list[list] = []

    def capturing_chat(model, messages, tools, think=False):
        captured_messages.append(list(messages))
        return scripted(model, messages, tools, think=think)

    monkeypatch.setattr(
        harness_module, "get_client", lambda: SimpleNamespace(chat=capturing_chat)
    )

    machine, history = _new_machine_and_history()

    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    final_state = states[-1]
    # The nudge must have produced the model's real finish_investigation
    # result, not the synthesized fallback.
    assert final_state.final_result == final_args
    assert scripted.call_count == 2

    nudge_text = "You didn't call a tool"
    nudge_messages_in_second_call = [
        m
        for m in captured_messages[1]
        if isinstance(m, dict) and m.get("role") == "user" and nudge_text in m.get("content", "")
    ]
    assert len(nudge_messages_in_second_call) == 1
    assert any(
        error.startswith("model_ended_without_tool_call:") for error in final_state.errors
    )
    assert "model_ended_without_finish_investigation" not in final_state.errors


def test_nudge_does_not_help_falls_back_exactly_as_before(monkeypatch):
    _patch_healthy_services(monkeypatch)

    responses = [
        _response([], content="Thinking out loud, no tool call yet."),
        _response([], content="Still no tool call after the nudge."),
    ]
    scripted = _patch_chat(monkeypatch, responses)

    machine, history = _new_machine_and_history()

    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    final_state = states[-1]
    # Exactly one nudge attempt: the original call plus one retry, then
    # terminal handling — a third chat() call would raise inside
    # ScriptedChat, so this also proves no second nudge was attempted.
    assert scripted.call_count == 2
    assert "model_ended_without_finish_investigation" in final_state.errors
    assert (
        sum(
            1
            for error in final_state.errors
            if error.startswith("model_ended_without_tool_call:")
        )
        == 2
    )
    assert final_state.final_result is not None
    assert final_state.final_result["confidence"] == 0.0
    assert final_state.final_result["likely_cause"].startswith("Undetermined")


def test_connectivity_precheck_skips_model_calls_when_ollama_unreachable(monkeypatch):
    monkeypatch.setattr(
        harness_module,
        "llm_health_check",
        lambda: {"reachable": False, "model_available": False, "local_models": []},
    )

    def _get_client_should_not_be_called():
        raise AssertionError("get_client should not have been called")

    def _vectorai_health_check_should_not_be_called():
        raise AssertionError("vectorai_health_check should not have been called")

    monkeypatch.setattr(harness_module, "get_client", _get_client_should_not_be_called)
    monkeypatch.setattr(
        harness_module, "vectorai_health_check", _vectorai_health_check_should_not_be_called
    )

    machine, history = _new_machine_and_history()

    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    assert len(states) == 1
    assert states[0].errors == ["Ollama is not reachable — is it running?"]
    assert states[0].final_result is not None


def test_connectivity_precheck_skips_model_calls_when_actian_unreachable(monkeypatch):
    monkeypatch.setattr(
        harness_module,
        "llm_health_check",
        lambda: {"reachable": True, "model_available": True, "local_models": []},
    )
    monkeypatch.setattr(
        harness_module, "vectorai_health_check", lambda: {"reachable": False}
    )

    def _get_client_should_not_be_called():
        raise AssertionError("get_client should not have been called")

    monkeypatch.setattr(harness_module, "get_client", _get_client_should_not_be_called)

    machine, history = _new_machine_and_history()

    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    assert len(states) == 1
    assert states[0].errors == [
        "Actian VectorAI DB is not reachable — is the container running "
        "(docker compose up -d)?"
    ]
    assert states[0].final_result is not None


def test_mid_investigation_connection_loss_falls_back_gracefully(monkeypatch):
    _patch_healthy_services(monkeypatch)
    manual_log = _patch_search_manual(monkeypatch)

    responses = [
        _response([_tool_call("search_manual", {"query": "bearing", "top_k": 3})]),
        ConnectionError(
            "Failed to connect to Ollama. Please check that Ollama is downloaded, "
            "running and accessible."
        ),
    ]
    scripted = _patch_chat(monkeypatch, responses)

    machine, history = _new_machine_and_history()

    # If the connection error propagated out of run_investigation() instead
    # of being caught and turned into a fallback, this call would raise and
    # fail the test.
    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    final_state = states[-1]
    assert scripted.call_count == 2
    assert any(
        error.startswith("llm_connection_lost_mid_investigation")
        for error in final_state.errors
    )
    assert final_state.final_result is not None
    assert final_state.final_result["confidence"] == 0.0
    assert final_state.final_result["likely_cause"].startswith("Undetermined")
    # The fallback must reuse evidence already gathered before the
    # connection was lost (round 1's search_manual), not just an empty stub.
    assert final_state.final_result["manual_sources"] == ["Bearing Troubleshooting"]
    assert len(manual_log) == 1


def test_malformed_finish_investigation_args_recorded_and_continues(monkeypatch):
    _patch_healthy_services(monkeypatch)

    valid_args = dict(
        likely_cause="worn bearing",
        confidence=0.8,
        recommendation="Inspect the bearing assembly.",
        supporting_evidence=[],
        retrieved_incident_ids=[],
        manual_sources=[],
    )
    missing_confidence_args = {k: v for k, v in valid_args.items() if k != "confidence"}

    responses = [
        _response([_tool_call("finish_investigation", missing_confidence_args)]),
        _response([_tool_call("finish_investigation", valid_args)]),
    ]
    scripted = _patch_chat(monkeypatch, responses)

    machine, history = _new_machine_and_history()

    # A TypeError from the missing required argument must be caught by the
    # existing tool-execution try/except, not propagate and crash the loop.
    states = list(
        harness_module.run_investigation(machine, history, "Investigate bearing fault")
    )

    final_state = states[-1]
    assert scripted.call_count == 2
    assert any(
        error.startswith("tool_error:finish_investigation") for error in final_state.errors
    )
    # The investigation continues past the malformed call and reaches a real
    # finish_investigation once the model retries with valid arguments.
    assert final_state.final_result == valid_args
