import sys
import types
import unittest
from unittest.mock import patch


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _identity_tool(function):
    return function


community = _module("langchain_community")
community.__path__ = []
community_tools = _module(
    "langchain_community.tools",
    DuckDuckGoSearchRun=object,
    OpenWeatherMapQueryRun=object,
)
community_utilities = _module(
    "langchain_community.utilities", OpenWeatherMapAPIWrapper=object
)
langchain_core = _module("langchain_core")
langchain_core.__path__ = []
langchain_core_tools = _module("langchain_core.tools", tool=_identity_tool)
dotenv = _module("dotenv", load_dotenv=lambda: None)

with patch.dict(
    sys.modules,
    {
        "dotenv": dotenv,
        "langchain_community": community,
        "langchain_community.tools": community_tools,
        "langchain_community.utilities": community_utilities,
        "langchain_core": langchain_core,
        "langchain_core.tools": langchain_core_tools,
    },
):
    import tools


class CropCalendarTests(unittest.TestCase):
    def test_blank_crop_name_returns_unknown_crop_response(self) -> None:
        response = tools._build_crop_calendar_response("   ")

        self.assertIn("No calendar entry found", response)
        self.assertIn("Available crops:", response)

    def test_known_crop_still_returns_calendar_data(self) -> None:
        response = tools._build_crop_calendar_response("tomato")

        self.assertIn("Crop calendar for Tomato", response)
        self.assertIn("Planting season:", response)


if __name__ == "__main__":
    unittest.main()
