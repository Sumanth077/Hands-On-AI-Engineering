# research_assistant.py
import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, LLMConfig, register_function
from tools.research_tools import web_search, fetch_page_content

load_dotenv()

# ── AG2 (formerly AutoGen) requires ag2>=0.11 ──────────────────────────────────

def build_llm_config() -> LLMConfig:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env or the sidebar.")
    return LLMConfig(
        {"model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
         "api_key": api_key,
         "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")},
        temperature=0.3,
        cache_seed=None,  # always fetch fresh data
    )


def run_research(topic: str) -> str:
    llm_config = build_llm_config()

    # ── Agents ─────────────────────────────────────────────────────────────────

    researcher = AssistantAgent(
        name="researcher",
        system_message="""You are a research specialist. Your job is to gather
comprehensive information about the given topic using web_search and fetch_page_content.
Perform at least 3 searches and fetch content from 2+ pages.
Summarise all findings clearly. End with: RESEARCH COMPLETE.""",
        llm_config=llm_config,
    )

    analyst = AssistantAgent(
        name="analyst",
        system_message="""You are a senior analyst. Once RESEARCH COMPLETE is signalled,
critically evaluate the research: identify key themes, contradictions, and knowledge gaps.
Produce a structured analysis with bullet points. End with: ANALYSIS COMPLETE.""",
        llm_config=llm_config,
    )

    writer = AssistantAgent(
        name="writer",
        system_message="""You are a professional technical writer. Once ANALYSIS COMPLETE
is signalled, produce a polished markdown report with:
## Executive Summary
## Key Findings
## Detailed Analysis
## Conclusions & Next Steps
End with: REPORT COMPLETE""",
        llm_config=llm_config,
    )

    # Executes tool calls; no LLM of its own
    executor = UserProxyAgent(
        name="executor",
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "workspace", "use_docker": False},
        is_termination_msg=lambda msg: "REPORT COMPLETE" in (msg.get("content") or ""),
        default_auto_reply="",
    )

    # ── Tool registration: caller=LLM agent, executor=UserProxyAgent ──────────
    # This AG2 pattern separates tool description (for the LLM) from execution
    # (sandboxed in the UserProxyAgent). The executor can be swapped for a
    # Docker container without changing any agent reasoning code.

    for fn in (web_search, fetch_page_content):
        register_function(
            fn,
            caller=researcher,
            executor=executor,
            name=fn.__name__,
            description=(fn.__doc__ or "").strip().split("\n")[0],
        )

    # ── GroupChat orchestration ────────────────────────────────────────────────
    groupchat = GroupChat(
        agents=[executor, researcher, analyst, writer],
        messages=[],
        max_round=20,
        speaker_selection_method="auto",  # LLM selects the next speaker
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    executor.initiate_chat(
        manager,
        message=f"Research the following topic thoroughly: {topic}",
        clear_history=True,
    )

    # Extract the final report from writer messages
    report_msgs = [
        m["content"]
        for m in groupchat.messages
        if m.get("name") == "writer" and m.get("content")
    ]
    return report_msgs[-1] if report_msgs else "Report generation did not complete."


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multi-Agent Research Assistant (AG2)", layout="wide")
st.title("Multi-Agent Research Assistant")
st.caption("Powered by AG2 (formerly AutoGen) — multi-agent research orchestration")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model = st.text_input("Model", value=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    if model:
        os.environ["LLM_MODEL"] = model
    topic = st.text_area("Research Topic", placeholder="e.g. 'Latest advances in quantum computing'")
    run_btn = st.button("Start Research", type="primary")

if "report" not in st.session_state:
    st.session_state.report = None

if run_btn:
    if not topic.strip():
        st.error("Please enter a research topic.")
    else:
        with st.spinner("AG2 agents are researching..."):
            try:
                st.session_state.report = run_research(topic)
            except Exception as exc:
                st.error(f"Error: {exc}")

if st.session_state.report:
    st.markdown(st.session_state.report)
    st.download_button(
        "Download Report",
        data=st.session_state.report,
        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )

st.markdown("---")
st.caption("For educational purposes only")
