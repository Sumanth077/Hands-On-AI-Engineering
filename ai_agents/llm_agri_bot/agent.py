import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI

from tools import get_tools

load_dotenv()

SYSTEM_PROMPT = """You are an expert agricultural assistant helping farmers make informed decisions.
You have access to weather data, web search for crop diseases and pest control, and a crop calendar.
Always give practical, actionable advice. Be concise and clear.

When answering:
- Use the weather tool for current conditions and forecasts when location is known or can be inferred.
- Use the search tool for recent pest outbreaks, disease symptoms, and treatment options.
- Use the crop calendar for planting and harvest timing questions.
- Combine tool results into clear steps farmers can follow today.
- If a tool fails or data is missing, say so and provide the best general guidance you can."""


def build_llm() -> ChatMistralAI:
    """Create a streaming Mistral chat model from environment config."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set. Add it to your .env file.")

    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        streaming=True,
    )


def build_agent_executor() -> AgentExecutor:
    """Assemble the LangChain tool-calling agent and executor."""
    llm = build_llm()
    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def format_chat_history(messages: list[dict]) -> list[tuple[str, str]]:
    """Convert UI message dicts to LangChain chat history tuples."""
    history: list[tuple[str, str]] = []
    for message in messages[:-1]:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            history.append(("human", content))
        elif role == "assistant":
            history.append(("ai", content))
    return history


def run_agent(
    user_input: str,
    chat_history: list[dict] | None = None,
    callbacks: list | None = None,
) -> str:
    """Invoke the agent and return the full text response."""
    executor = build_agent_executor()
    history = format_chat_history(chat_history or [])

    result = executor.invoke(
        {
            "input": user_input,
            "chat_history": history,
        },
        config={"callbacks": callbacks or []},
    )
    return result["output"]


def stream_agent_response(
    user_input: str,
    chat_history: list[dict] | None = None,
):
    """Stream agent response tokens as they are generated."""
    executor = build_agent_executor()
    history = format_chat_history(chat_history or [])

    for chunk in executor.stream(
        {
            "input": user_input,
            "chat_history": history,
        },
        stream_mode="messages",
    ):
        message_chunk, _metadata = chunk
        if message_chunk.content:
            yield message_chunk.content
