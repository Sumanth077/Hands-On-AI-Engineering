"""Provider-free checks for browser history result handling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import TestCase
from unittest.mock import patch


def load_agent_module():
    """Load agent.py with provider packages stubbed for an offline unit test."""
    project_dir = Path(__file__).resolve().parents[1]

    dotenv = ModuleType("dotenv")

    def load_dotenv() -> None:
        pass

    dotenv.load_dotenv = load_dotenv

    browser_use = ModuleType("browser_use")
    browser_use.Agent = object
    browser_use.ChatOpenAI = object

    spec = importlib.util.spec_from_file_location(
        "browser_automation_agent_under_test", project_dir / "agent.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load browser_automation_agent/agent.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"dotenv": dotenv, "browser_use": browser_use}):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class FakeHistory:
    def __init__(self, summary: str, done: bool, errors: list[str]):
        self._summary = summary
        self._done = done
        self._errors = errors

    def final_result(self) -> str:
        return self._summary

    def is_done(self) -> bool:
        return self._done

    def errors(self) -> list[str]:
        return self._errors

    def urls(self) -> list[str]:
        return []

    def model_thoughts(self) -> list[str]:
        return []


class ExtractHistoryTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_module = load_agent_module()

    def test_incomplete_partial_result_keeps_last_step_error(self) -> None:
        result = self.agent_module._extract_history(
            FakeHistory(
                "Found two pages, but the search did not finish.",
                done=False,
                errors=["NavigationTimeout: page did not respond"],
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "NavigationTimeout: page did not respond")
        self.assertIn("Found two pages", result.summary)

    def test_completed_result_suppresses_recovered_step_error(self) -> None:
        result = self.agent_module._extract_history(
            FakeHistory(
                "The search completed successfully.",
                done=True,
                errors=["Transient model response error"],
            )
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
