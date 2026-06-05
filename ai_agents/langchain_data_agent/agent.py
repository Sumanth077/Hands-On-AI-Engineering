"""LangGraph SQL agent for the Chinook database (read-only)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

DB_PATH = Path(__file__).parent / "chinook.db"
TOP_K = 10

FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|"
    r"GRANT|REVOKE|ATTACH|DETACH|MERGE|CALL|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)
READ_ONLY_SQL = re.compile(r"^\s*(SELECT|WITH|PRAGMA|EXPLAIN)\b", re.IGNORECASE | re.DOTALL)


def _llm_context(state: MessagesState) -> list[HumanMessage]:
    """Text-only context for Kimi — never replay assistant tool_calls to the API."""
    user_question: str | None = None
    blocks: list[str] = []

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                summary = ", ".join(
                    f"{tc.get('name')}({tc.get('args')})" for tc in msg.tool_calls
                )
                blocks.append(f"Assistant requested tools: {summary}")
            elif msg.content and str(msg.content).strip():
                blocks.append(str(msg.content).strip())
        elif isinstance(msg, ToolMessage):
            name = msg.name or "tool"
            body = msg.content if isinstance(msg.content, str) else str(msg.content)
            blocks.append(f"### {name}\n{body}")

    parts: list[str] = []
    if blocks:
        parts.append("Database context:\n" + "\n\n".join(blocks))
    if user_question:
        parts.append(f"User question: {user_question}")
    return [HumanMessage(content="\n\n".join(parts) if parts else "")]


def _validate_read_only(query: str) -> str:
    cleaned = query.strip().rstrip(";")
    if FORBIDDEN_SQL.search(cleaned):
        raise ValueError("Only read-only SELECT queries are allowed.")
    if not READ_ONLY_SQL.match(cleaned):
        raise ValueError("Only read-only SELECT queries are allowed.")
    return cleaned


class ReadOnlyQuerySQLDatabaseTool(QuerySQLDatabaseTool):
    """SQL query tool that rejects non-SELECT statements."""

    def _run(self, query: str, run_manager=None) -> str:
        safe_query = _validate_read_only(query)
        return self.db.run_no_throw(safe_query, include_columns=True)


def _build_model() -> ChatOpenAI:
    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        raise ValueError("ORQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatOpenAI(
        model="kimi-k2.6",
        openai_api_key=api_key,
        openai_api_base="https://api.orq.ai/v3/router",
        temperature=1,
        # Kimi K2.6 defaults to thinking mode, which breaks multi-turn tool calling
        # unless reasoning_content is preserved on every assistant tool-call message.
        extra_body={"thinking": {"type": "disabled"}},
    )


def _build_tools(db: SQLDatabase, model: ChatOpenAI) -> list:
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()
    for i, tool in enumerate(tools):
        if tool.name == "sql_db_query":
            tools[i] = ReadOnlyQuerySQLDatabaseTool(db=db)
    return tools


def _create_agent():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.resolve()}")
    model = _build_model()
    tools = _build_tools(db, model)

    get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")

    run_query_tool = next(t for t in tools if t.name == "sql_db_query")
    run_query_node = ToolNode([run_query_tool], name="run_query")

    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")

    generate_query_system_prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {TOP_K} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
When you have enough information to answer, respond in clear natural language.
""".strip()

    check_query_system_prompt = f"""
You are a SQL expert with a strong attention to detail.
Double check the {db.dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
Only SELECT (read-only) queries are permitted.
""".strip()

    def list_tables(state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "list_tables_call",
            "type": "tool_call",
        }
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(content=f"Available tables: {tool_message.content}")
        return {"messages": [response]}

    def _require_tool_call(llm_with_tools, messages, tool_name: str, retry_hint: str):
        response = llm_with_tools.invoke(messages)
        if response.tool_calls and any(tc.get("name") == tool_name for tc in response.tool_calls):
            return response
        return llm_with_tools.invoke([*messages, HumanMessage(content=retry_hint)])

    def call_get_schema(state: MessagesState):
        llm_with_tools = model.bind_tools([get_schema_tool])
        messages = [
            {
                "role": "system",
                "content": (
                    "Call sql_db_schema with comma-separated table names relevant "
                    "to the user question."
                ),
            },
            *_llm_context(state),
        ]
        response = _require_tool_call(
            llm_with_tools,
            messages,
            "sql_db_schema",
            "Call sql_db_schema now with the relevant table names.",
        )
        return {"messages": [response]}

    def generate_query(state: MessagesState):
        llm_with_tools = model.bind_tools([run_query_tool])
        messages = [
            {"role": "system", "content": generate_query_system_prompt},
            *_llm_context(state),
        ]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def check_query(state: MessagesState):
        system_message = {"role": "system", "content": check_query_system_prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = model.bind_tools([run_query_tool])
        response = _require_tool_call(
            llm_with_tools,
            [system_message, user_message],
            "sql_db_query",
            "Call sql_db_query with the validated read-only SELECT query.",
        )
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["check_query", END]:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return END
        return "check_query"

    builder = StateGraph(MessagesState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", get_schema_node)
    builder.add_node("generate_query", generate_query)
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", run_query_node)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")

    return builder.compile()


_agent = None


def get_agent(*, reload: bool = False):
    global _agent
    if _agent is None or reload:
        _agent = _create_agent()
    return _agent


def _to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _extract_sql_metadata(messages: list[BaseMessage]) -> tuple[str | None, str | None]:
    last_sql: str | None = None
    last_results: str | None = None
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get("name") == "sql_db_query":
                    last_sql = tool_call.get("args", {}).get("query")
        if isinstance(message, ToolMessage) and message.name == "sql_db_query":
            last_results = message.content
    return last_sql, last_results


def extract_answer(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            text = message.content
            if isinstance(text, str) and text.strip():
                if not text.startswith("Available tables:"):
                    return text.strip()
    return "I could not generate an answer. Please try rephrasing your question."


def run_query(question: str, history: list[dict] | None = None) -> dict:
    """Run the SQL agent and return answer metadata for the UI."""
    agent = get_agent(reload=True)
    prior = _to_langchain_messages(history or [])
    prior.append(HumanMessage(content=question))

    result = agent.invoke({"messages": prior})

    last_sql, last_results = _extract_sql_metadata(result["messages"])
    return {
        "answer": extract_answer(result["messages"]),
        "sql": last_sql,
        "raw_results": last_results,
        "messages": result["messages"],
    }
