"""Focused tests for long-running agent UI behavior without provider calls."""

import importlib.util
import queue
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def click(self, *args, **kwargs):
        return None


def _load_app_without_optional_dependencies():
    gradio = types.ModuleType("gradio")
    gradio.Blocks = lambda *args, **kwargs: _Context()
    gradio.Markdown = lambda *args, **kwargs: None
    gradio.Row = lambda *args, **kwargs: _Context()
    gradio.Column = lambda *args, **kwargs: _Context()
    gradio.Textbox = lambda *args, **kwargs: _Context()
    gradio.Button = lambda *args, **kwargs: _Context()
    gradio.HTML = lambda *args, **kwargs: _Context()
    gradio.update = lambda **kwargs: kwargs
    gradio.themes = types.SimpleNamespace(Soft=lambda: None)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None

    smolagents = types.ModuleType("smolagents")
    smolagents.CodeAgent = object
    smolagents.DuckDuckGoSearchTool = object
    smolagents.LiteLLMModel = object

    tools = types.ModuleType("tools")
    tools.WikipediaTool = object

    modules = {
        "gradio": gradio,
        "dotenv": dotenv,
        "smolagents": smolagents,
        "tools": tools,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("smolagents_app_under_test", PROJECT_DIR / "app.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
    return app


class RunAgentTimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _load_app_without_optional_dependencies()

    def test_idle_step_queue_keeps_run_active_until_done_event(self):
        app = self.app

        class DelayedQueue:
            def __init__(self):
                self.calls = 0

            def get(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise queue.Empty
                return "done", None

            def put(self, item):
                raise AssertionError("the mocked worker does not enqueue events")

        class UnstartedThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                return None

            def join(self, timeout=None):
                return None

        with patch.object(app.queue, "Queue", DelayedQueue), patch.object(
            app.threading, "Thread", UnstartedThread
        ):
            outputs = list(app.run_agent("summarize a documentation page"))

        self.assertGreaterEqual(len(outputs), 3)
        waiting_trace, _, waiting_spinner = outputs[1]
        self.assertIn("[WAITING]", waiting_trace)
        self.assertTrue(waiting_spinner["visible"])
        self.assertNotIn("[TIMEOUT]", waiting_trace)
        self.assertIn("Agent finished.", outputs[-1][0])
        self.assertFalse(outputs[-1][2]["visible"])


if __name__ == "__main__":
    unittest.main()
