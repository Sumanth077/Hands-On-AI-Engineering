import importlib
import json
import sys
import types
import unittest


# The orchestrator only uses llama-cpp for the Llama type annotation, and its
# config module only uses python-dotenv to load local settings. Keep these unit
# tests independent of the model runtime and optional environment setup.
try:
    importlib.import_module("llama_cpp")
except ModuleNotFoundError:
    llama_cpp = types.ModuleType("llama_cpp")
    llama_cpp.Llama = type("Llama", (), {})
    sys.modules["llama_cpp"] = llama_cpp

try:
    importlib.import_module("dotenv")
except ModuleNotFoundError:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv

from customer_query_routing_agent.orchestrator import triage


class FakeLlama:
    def __init__(self, response):
        self.response = response

    def create_chat_completion(self, **_kwargs):
        return {
            "choices": [
                {"message": {"content": json.dumps(self.response)}}
            ]
        }


def _response(**overrides):
    response = {
        "path": "resolve",
        "sentiment_score": 0.0,
        "urgency_score": 0.0,
        "legal_flag": False,
        "repeat_contact": False,
        "reasoning": "Routine policy question.",
        "escalation_reasons": [],
    }
    response.update(overrides)
    return response


class TriageBooleanFieldTests(unittest.TestCase):
    def test_json_boolean_values_are_preserved(self):
        decision = triage(
            "Where is my order?",
            0.9,
            FakeLlama(_response(legal_flag=True, repeat_contact=False)),
        )

        self.assertTrue(decision.legal_flag)
        self.assertFalse(decision.repeat_contact)
        self.assertEqual(decision.reasoning, "Routine policy question.")

    def test_string_false_uses_regex_fallback(self):
        decision = triage(
            "Can I get a refund for my order?",
            0.9,
            FakeLlama(_response(legal_flag="false", repeat_contact="false")),
        )

        self.assertFalse(decision.legal_flag)
        self.assertFalse(decision.repeat_contact)
        self.assertTrue(decision.reasoning.startswith("(Fallback: regex-only"))

    def test_missing_boolean_fields_keep_regex_defaults(self):
        response = _response()
        del response["legal_flag"]
        del response["repeat_contact"]
        decision = triage(
            "I will sue you over this fraud.",
            0.9,
            FakeLlama(response),
        )

        self.assertTrue(decision.legal_flag)
        self.assertFalse(decision.repeat_contact)
        self.assertEqual(decision.reasoning, "Routine policy question.")

    def test_invalid_boolean_preserves_regex_escalation(self):
        decision = triage(
            "I will sue you over this fraud.",
            0.9,
            FakeLlama(_response(legal_flag="false", repeat_contact="false")),
        )

        self.assertEqual(decision.path, "escalate")
        self.assertTrue(decision.legal_flag)
        self.assertTrue(decision.reasoning.startswith("(Fallback: regex-only"))


if __name__ == "__main__":
    unittest.main()
