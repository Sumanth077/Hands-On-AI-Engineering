import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]


class FakeSecret:
    def __init__(self, token):
        self.token = token

    @classmethod
    def from_token(cls, token):
        return cls(token)


def make_fake_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    if name in {
        "haystack",
        "haystack.components",
        "haystack_integrations",
        "haystack_integrations.components",
        "haystack_integrations.components.generators",
        "haystack_integrations.tools",
    }:
        module.__path__ = []
    return module


def load_agent_module():
    created_agents = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created_agents.append(self)

    class FakeSearchableToolset:
        def __init__(self, catalog):
            self.catalog = catalog

    class FakeMCPToolset:
        def __init__(self, server_info):
            self.server_info = server_info

    class FakeServerInfo:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_modules = {
        "haystack": make_fake_module("haystack"),
        "haystack.components": make_fake_module("haystack.components"),
        "haystack.components.agents": make_fake_module(
            "haystack.components.agents", Agent=FakeAgent
        ),
        "haystack.tools": make_fake_module(
            "haystack.tools", SearchableToolset=FakeSearchableToolset
        ),
        "haystack.utils": make_fake_module("haystack.utils", Secret=FakeSecret),
        "haystack_integrations": make_fake_module("haystack_integrations"),
        "haystack_integrations.components": make_fake_module(
            "haystack_integrations.components"
        ),
        "haystack_integrations.components.generators": make_fake_module(
            "haystack_integrations.components.generators"
        ),
        "haystack_integrations.components.generators.google_genai": make_fake_module(
            "haystack_integrations.components.generators.google_genai",
            GoogleGenAIChatGenerator=FakeGenerator,
        ),
        "haystack_integrations.tools": make_fake_module("haystack_integrations.tools"),
        "haystack_integrations.tools.mcp": make_fake_module(
            "haystack_integrations.tools.mcp",
            MCPToolset=FakeMCPToolset,
            StreamableHttpServerInfo=FakeServerInfo,
        ),
    }
    module_path = PROJECT_DIR / "agent.py"
    spec = importlib.util.spec_from_file_location(
        "github_intelligence_agent_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
        module.build_agent("session-a-github", "session-a-gemini")
        module.build_agent("session-b-github", "session-b-gemini")

    return created_agents


class AgentCredentialIsolationTests(unittest.TestCase):
    def test_credentials_are_bound_to_each_agent_without_environment_mutation(self):
        with patch.dict(
            os.environ,
            {
                "GITHUB_PAT": "ambient-github",
                "GOOGLE_API_KEY": "ambient-gemini",
            },
        ):
            original_environment = {
                "GITHUB_PAT": os.environ["GITHUB_PAT"],
                "GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"],
            }
            agent_a, agent_b = load_agent_module()

            self.assertEqual(
                agent_a.tools.catalog.server_info.kwargs["token"].token,
                "session-a-github",
            )
            self.assertEqual(
                agent_a.chat_generator.kwargs["api_key"].token,
                "session-a-gemini",
            )
            self.assertEqual(
                agent_b.tools.catalog.server_info.kwargs["token"].token,
                "session-b-github",
            )
            self.assertEqual(
                agent_b.chat_generator.kwargs["api_key"].token,
                "session-b-gemini",
            )
            self.assertEqual(
                {
                    "GITHUB_PAT": os.environ["GITHUB_PAT"],
                    "GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"],
                },
                original_environment,
            )


if __name__ == "__main__":
    unittest.main()
