import time
import streamlit as st
import os
from dotenv import load_dotenv
from mem0 import MemoryClient
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

MEM0_API_KEY = os.getenv("MEM0_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LLM_MODEL = "mistral-small-latest"

st.set_page_config(
    page_title="CartMate — NovaMart Support",
    page_icon="🛒",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to memory…")
def init_memory() -> MemoryClient:
    return MemoryClient(api_key=MEM0_API_KEY)


@st.cache_resource(show_spinner="Loading AI model…")
def init_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=LLM_MODEL,
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.7,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def _unwrap(results) -> list:
    """Accept both list and {'results': [...]} formats from mem0."""
    if isinstance(results, dict) and "results" in results:
        return results["results"]
    return results if isinstance(results, list) else []


def get_all_memories(user_id: str) -> list:
    try:
        return _unwrap(init_memory().get_all(filters={"user_id": user_id}))
    except Exception:
        return []


def search_memories(query: str, user_id: str, limit: int = 5) -> list:
    try:
        return _unwrap(init_memory().search(query, filters={"user_id": user_id}, limit=limit))
    except Exception:
        return []


def add_to_memory(user_msg: str, assistant_msg: str, user_id: str) -> None:
    # Store outcome in session_state so it survives st.rerun() and can be
    # displayed in the sidebar on the next render pass.
    try:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        result = init_memory().add(messages, user_id=user_id)
        time.sleep(3)
        extracted = _unwrap(result)
        st.session_state.mem_extracted_count = len(extracted)
        st.session_state.mem_status = "ok"
        st.session_state.mem_raw_result = result
        st.session_state.mem_error = None
    except Exception as e:
        st.session_state.mem_status = "error"
        st.session_state.mem_extracted_count = 0
        st.session_state.mem_raw_result = None
        st.session_state.mem_error = str(e)


def memory_text(mem) -> str:
    if isinstance(mem, dict):
        return mem.get("memory", mem.get("text", str(mem)))
    return str(mem)


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------
def build_system_prompt(user_name: str, memories: list) -> str:
    if memories:
        bullet_list = "\n".join(f"- {memory_text(m)}" for m in memories)
        memory_block = (
            f"Relevant memories about {user_name} from past interactions:\n"
            f"{bullet_list}\n\n"
            "Use these memories to personalise your response. "
            "Reference past interactions naturally — don't force it."
        )
    else:
        memory_block = (
            f"No specific memories found for this query with {user_name}. "
            "This may be a new topic or a first-time interaction."
        )

    return (
        f"You are CartMate, NovaMart's friendly and knowledgeable e-commerce support agent. "
        f"The customer's name is {user_name}.\n\n"
        f"{memory_block}\n\n"
        f"You specialise in the following support areas:\n"
        f"- Order tracking: status updates, estimated delivery, missing orders\n"
        f"- Returns & refunds: eligibility, how to initiate, timelines, and status\n"
        f"- Delivery issues: late parcels, wrong address, damaged goods, carrier problems\n"
        f"- Product questions: specifications, availability, compatibility, and recommendations\n"
        f"- Account issues: login problems, password resets, profile and address updates\n"
        f"- Payment problems: failed charges, duplicate payments, billing queries, vouchers\n\n"
        f"Guidelines:\n"
        f"- Be warm, concise, and solution-focused.\n"
        f"- Address {user_name} by name occasionally, not every message.\n"
        f"- When a past memory is directly relevant, acknowledge it naturally.\n"
        f"- Always offer a clear next step or resolution path.\n"
        f"- If an issue requires human escalation (e.g. fraud, complex disputes), "
        f"advise {user_name} to contact NovaMart support at support@novamart.com."
    )


def generate_response(user_name: str, user_msg: str, history: list, memories: list) -> str:
    llm = init_llm()
    msgs = [SystemMessage(content=build_system_prompt(user_name, memories))]

    for m in history[-10:]:  # last 10 messages for context window efficiency
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))

    msgs.append(HumanMessage(content=user_msg))
    content = llm.invoke(msgs).content
    if isinstance(content, list):
        return "".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# Guard: API keys
# ---------------------------------------------------------------------------
missing = [k for k, v in {"MEM0_API_KEY": MEM0_API_KEY, "MISTRAL_API_KEY": MISTRAL_API_KEY}.items() if not v]
if missing:
    st.error(f"Missing API keys in .env: {', '.join(missing)}")
    st.code("\n".join(f"{k}=your_{k.lower()}_here" for k in missing), language="bash")
    st.stop()

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "mem_status" not in st.session_state:
    st.session_state.mem_status = None   # "ok" | "error" | None

if "mem_error" not in st.session_state:
    st.session_state.mem_error = None

if "mem_extracted_count" not in st.session_state:
    st.session_state.mem_extracted_count = 0

if "mem_raw_result" not in st.session_state:
    st.session_state.mem_raw_result = None

# ---------------------------------------------------------------------------
# Sidebar — live memory viewer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("CartMate Memory")

    if st.session_state.user_name:
        st.caption(f"Stored memories for **{st.session_state.user_name}**")

        if st.session_state.mem_status == "ok":
            st.success("Memory saved.", icon="✅")
        elif st.session_state.mem_status == "error":
            st.error(f"Memory save failed: {st.session_state.mem_error}", icon="🚨")

        all_mems = get_all_memories(st.session_state.user_id)

        if all_mems:
            for m in all_mems:
                st.markdown(f"- {memory_text(m)}")
        else:
            st.info("No memories stored yet for this user.")

        st.divider()
        if st.button("Clear My Memories", use_container_width=True):
            try:
                init_memory().delete_all(filters={"user_id": st.session_state.user_id})
                st.session_state.messages = []
                st.success("Memories cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Enter your name to get started.")

    st.divider()
    st.caption("Mem0 Cloud · Mistral Small 4 · NovaMart")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("CartMate — NovaMart Support")
st.caption("Your personal shopping assistant, powered by long-term memory")

# ── Step 1: collect user name ────────────────────────────────────────────────
if not st.session_state.user_name:
    st.markdown("### Welcome to NovaMart Support!")
    st.markdown(
        "Hi, I'm CartMate, your personal shopping support agent from NovaMart. "
        "I'll remember your orders, preferences, and past issues so you never have to repeat yourself."
    )

    with st.form("name_form", clear_on_submit=True):
        name_input = st.text_input(
            "Your name",
            placeholder="Type your name and press Enter…",
            label_visibility="collapsed",
        )
        col_a, col_b = st.columns([4, 1])
        with col_b:
            submitted = st.form_submit_button("Start", type="primary", use_container_width=True)

    if submitted and name_input.strip():
        clean_name = name_input.strip().title()
        user_id = clean_name.lower().replace(" ", "_")
        st.session_state.user_name = clean_name
        st.session_state.user_id = user_id

        existing = get_all_memories(user_id)
        if existing:
            greeting = (
                f"Welcome back, {clean_name}! Great to have you back at NovaMart. "
                "I still have your history on file, so we can pick up right where we left off. "
                "What can I help you with today?"
            )
        else:
            greeting = (
                f"Hi {clean_name}! I'm CartMate, your personal shopping support agent from NovaMart. "
                "I'll remember your orders, preferences, and past issues so you never have to repeat yourself. "
                "What can I help you with today — order tracking, a return, a delivery issue, or something else?"
            )

        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.rerun()

# ── Step 2: chat interface ───────────────────────────────────────────────────
else:
    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Handle new message
    if prompt := st.chat_input("Type your message here…"):
        # Show user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve + generate + display
        with st.chat_message("assistant"):
            with st.spinner("Searching memories and generating response…"):
                relevant = search_memories(prompt, st.session_state.user_id)
                response = generate_response(
                    st.session_state.user_name,
                    prompt,
                    st.session_state.messages[:-1],  # history without current user msg
                    relevant,
                )
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save to Mem0 — result stored in session_state so the sidebar can
        # display success/failure after the rerun without losing the status.
        with st.spinner("Saving to memory…"):
            add_to_memory(prompt, response, st.session_state.user_id)

        # Rerun to refresh sidebar memory list
        st.rerun()
