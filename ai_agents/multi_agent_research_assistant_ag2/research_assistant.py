"""Multi-agent research pipeline using AG2 (AutoGen): researcher, analyst, and writer
agents collaborate under GroupChat, powered by Mistral Small 4, with a Streamlit UI."""
import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, LLMConfig, register_function
from tools.research_tools import web_search, fetch_page_content

load_dotenv()

# ── Patch: normalize Mistral citation chunks before AG2's message parser sees them ──
# Mistral's grounding/web-search feature returns AssistantMessage.content as
# list[TextChunk | ReferenceChunk] instead of a plain str. AG2's ChatCompletionMessage
# expects str | dict | list[dict] | None, so passing Pydantic objects causes validation
# errors. We replace the ChatCompletionMessage name in autogen.oai.mistral's module
# namespace with a factory that flattens the list to a string first.
import autogen.oai.mistral as _mistral_module
from autogen.oai.oai_models import ChatCompletionMessage as _RealCCM
from mistralai.client.models import TextChunk as _TextChunk


def _normalize_mistral_content(content):
    if not isinstance(content, list):
        return content
    return "".join(
        chunk.text if isinstance(chunk, _TextChunk) else
        (chunk.text if hasattr(chunk, "text") else "")
        for chunk in content
    )


def _CCMFactory(**kwargs):
    content = kwargs.get("content")
    if isinstance(content, list):
        kwargs["content"] = _normalize_mistral_content(content)
    return _RealCCM(**kwargs)


_mistral_module.ChatCompletionMessage = _CCMFactory
# ── End patch ──────────────────────────────────────────────────────────────────


def build_llm_config() -> LLMConfig:
    """Builds the AG2 LLMConfig for Mistral Small 4 using AG2's native Mistral client."""
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set. Add it to .env or the sidebar.")
    return LLMConfig(
        {"api_type": "mistral",
         "model": os.getenv("LLM_MODEL", "mistral-small-latest"),
         "api_key": api_key},
        temperature=0.3,
        cache_seed=None,  # always fetch fresh data
    )


def run_research(topic: str) -> str:
    """Runs the researcher, analyst, and writer agents and returns the final report.

    Tool calls are confined to an isolated researcher/executor exchange, separate from
    the analyst/writer GroupChat. Mistral's API rejects message histories where the
    number of function calls and responses don't match, which happens when GroupChat's
    "auto" speaker selection asks the LLM who should speak next while a tool call from
    researcher is still awaiting its response from executor.
    """
    llm_config = build_llm_config()

    # ── Agents ─────────────────────────────────────────────────────────────────

    researcher = AssistantAgent(
        name="researcher",
        system_message="""You are a research specialist. Search for information about the given topic using web_search.
Perform exactly 2 searches, then summarise the findings clearly. End with: RESEARCH COMPLETE.""",
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
        is_termination_msg=lambda msg: any(
            phrase in (msg.get("content") or "") for phrase in ("RESEARCH COMPLETE", "REPORT COMPLETE")
        ),
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

    # ── Phase 1: isolated researcher/executor exchange ─────────────────────────
    # A plain two-agent chat always pairs a tool call with its response in the
    # same turn, so there is never an orphaned tool call in the history.

    executor.initiate_chat(
        researcher,
        message=f"Research the following topic thoroughly: {topic}",
        clear_history=True,
    )
    research_notes = researcher.last_message(executor)["content"]
    # Strip the Phase 1 sentinel before handing off to Phase 2. Without this,
    # executor's is_termination_msg fires on the GroupChat kickoff message
    # (which contains the raw notes) and terminates the chat before analyst
    # or writer ever speak.
    research_notes = research_notes.replace("RESEARCH COMPLETE", "").strip()

    # ── Phase 2: GroupChat orchestration for analyst and writer ────────────────
    # Neither agent calls tools, so the LLM-driven speaker selection below never
    # sees an unanswered tool call in the shared history.

    groupchat = GroupChat(
        agents=[executor, analyst, writer],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",  # analyst then writer, no extra LLM call per round
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    executor.initiate_chat(
        manager,
        message=f"Research on '{topic}' is complete. Analyst, please critically evaluate the findings below:\n\n{research_notes}",
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
st.caption("Powered by AG2 (formerly AutoGen): multi-agent research orchestration")

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
