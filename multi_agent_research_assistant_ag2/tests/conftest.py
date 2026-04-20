"""
Stub out UI and optional dependencies so tests run without a Streamlit install.
research_assistant.py calls st.* at module level, so we must intercept before import.
"""
import sys
import os
from types import ModuleType
from unittest.mock import MagicMock


def _stub_module(name: str, **attrs) -> ModuleType:
    m = ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# Streamlit: stub the entire module — research_assistant.py runs st.* at module level
_st = _stub_module(
    "streamlit",
    set_page_config=MagicMock(),
    title=MagicMock(),
    caption=MagicMock(),
    header=MagicMock(),
    text_input=MagicMock(return_value=""),
    text_area=MagicMock(return_value=""),
    button=MagicMock(return_value=False),
    spinner=MagicMock(),
    error=MagicMock(),
    markdown=MagicMock(),
    download_button=MagicMock(),
    sidebar=MagicMock(),
    session_state=MagicMock(report=None),
)
_st.sidebar.__enter__ = MagicMock(return_value=_st.sidebar)
_st.sidebar.__exit__ = MagicMock(return_value=False)

# Ensure parent dir (multi_agent_research_assistant_ag2/) is on sys.path
parent = os.path.dirname(os.path.dirname(__file__))
if parent not in sys.path:
    sys.path.insert(0, parent)
