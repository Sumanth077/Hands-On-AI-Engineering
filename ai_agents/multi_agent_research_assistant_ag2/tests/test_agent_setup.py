import os
import pytest
os.environ.setdefault("OPENAI_API_KEY", "test-key-no-llm-calls")

def test_agents_instantiate():
    """Verify agent setup does not raise — no LLM calls made."""
    from research_assistant import build_llm_config
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

    llm_config = build_llm_config()
    researcher = AssistantAgent(name="researcher", system_message="Test", llm_config=llm_config)
    analyst = AssistantAgent(name="analyst", system_message="Test", llm_config=llm_config)
    writer = AssistantAgent(name="writer", system_message="Test", llm_config=llm_config)
    executor = UserProxyAgent(name="executor", human_input_mode="NEVER", code_execution_config=False)

    gc = GroupChat(agents=[executor, researcher, analyst, writer], messages=[], max_round=12)
    manager = GroupChatManager(groupchat=gc, llm_config=llm_config)
    assert len(gc.agents) == 4
    assert manager.groupchat is gc

def test_llm_config_structure():
    """Validate build_llm_config returns a proper LLMConfig."""
    from research_assistant import build_llm_config
    from autogen import LLMConfig
    cfg = build_llm_config()
    assert isinstance(cfg, LLMConfig)
    assert len(cfg.config_list) >= 1
    entry = cfg.config_list[0]
    assert entry.model
    assert entry.api_key

def test_tool_registration():
    """Register both tools on researcher/executor pair."""
    from tools.research_tools import web_search, fetch_page_content
    from autogen import AssistantAgent, UserProxyAgent, LLMConfig, register_function

    llm = LLMConfig({"model": "gpt-4o-mini", "api_key": "test"})
    researcher = AssistantAgent(name="r", system_message="test", llm_config=llm)
    executor = UserProxyAgent(name="e", human_input_mode="NEVER", code_execution_config=False)
    for fn in (web_search, fetch_page_content):
        register_function(fn, caller=researcher, executor=executor,
                          name=fn.__name__, description=(fn.__doc__ or "").strip().split("\n")[0])
