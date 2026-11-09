"""
LangGraph ReAct agent for Agentic SQL Search, with three tools:
  • load_skill: loads SQL syntax rules and schema reference
  • get_schema: inspects the live database schema
  • execute_sql: runs a SELECT query and returns results
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from db_tools import create_database_functions

load_dotenv()

DB_PATH = Path(__file__).parent / "data" / "ecommerce.db"
SKILLS_DIR = Path(__file__).parent / "skills"

llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


@tool
def load_skill(skill_name: str) -> str:
    """
    Load a skill document with syntax rules and query patterns.
    Always call this first with skill_name='sql' before writing any query.
    """
    path = SKILLS_DIR / f"{skill_name}.md"
    if not path.exists():
        available = [p.stem for p in SKILLS_DIR.glob("*.md")]
        return f"Skill '{skill_name}' not found. Available: {', '.join(available)}"
    return path.read_text()


SYSTEM_PROMPT = """You are a data analyst agent with access to an e-commerce SQLite database \
containing customers, products, orders, and order items.

Your job is to answer the user's question by querying the database.

Always follow this exact workflow:
1. Call load_skill with skill_name='sql' to review SQL syntax rules and query patterns
2. Call get_schema to confirm available tables and column names
3. Write and run the SQL query with execute_sql
4. Return a clear, concise summary of the results

Never guess column names — always check the schema first.
Only write SELECT queries. Be precise with JOINs and aggregations."""


def run_agent(question: str, db_path: str | Path = DB_PATH) -> dict:
    """
    Run the agent against one database path and return its answer and tool trace.

    {"answer": str, "steps": [{"tool": str, "input": dict, "output": str}]}
    """
    get_schema_fn, execute_sql_fn = create_database_functions(db_path)
    tools = [load_skill, tool(get_schema_fn), tool(execute_sql_fn)]
    graph = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)
    result = graph.invoke({"messages": [("human", question)]})
    messages = result["messages"]

    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, str):
                answer = content.strip()
            elif isinstance(content, list):
                parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                answer = " ".join(parts).strip()
            if answer:
                break

    steps = []
    tool_outputs: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            tool_outputs[msg.tool_call_id] = (
                content if isinstance(content, str) else str(content)
            )

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                steps.append({
                    "tool": tool_call["name"],
                    "input": tool_call["args"],
                    "output": tool_outputs.get(tool_call["id"], ""),
                })

    return {"answer": answer, "steps": steps}
