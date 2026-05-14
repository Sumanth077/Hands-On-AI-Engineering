"""LangChain agent orchestration with Fetch MCP and NVIDIA NIM."""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_ID = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash")
print(f"Using NVIDIA model: {MODEL_ID}")

SYSTEM_PROMPT = """You are a Documentation QnA Agent. You help users understand documentation
using the fetched documentation content provided in the conversation.

When answering questions:
- Prefer the documentation content already provided in the conversation.
- Use the fetch tool only when you need additional pages or sections from the
  same documentation site. Always pass the exact documentation URL given by the
  user — never invent or modify URLs.
- For long pages, use start_index to read additional sections in chunks.
- Base your answers only on the fetched documentation content.
- If the information is not in the documentation, say so clearly.
- Be concise, accurate, and helpful.
"""

TRUNCATION_RE = re.compile(r"start_index of (\d+)")


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to your .env file."
        )
    return ChatOpenAI(
        model=MODEL_ID,
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        temperature=0.2,
    )


def get_mcp_config() -> dict[str, dict[str, Any]]:
    args = ["-m", "mcp_server_fetch"]
    if os.getenv("FETCH_IGNORE_ROBOTS", "").lower() in {"1", "true", "yes"}:
        args.append("--ignore-robots-txt")

    return {
        "fetch": {
            "transport": "stdio",
            "command": sys.executable,
            "args": args,
            "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
        }
    }


async def get_fetch_tool():
    client = MultiServerMCPClient(get_mcp_config())
    tools = await client.get_tools()
    for tool in tools:
        if tool.name == "fetch":
            return tool
    raise RuntimeError("Fetch MCP server did not expose a 'fetch' tool.")


def extract_tool_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(result, dict) and result.get("type") == "text":
        return str(result.get("text", ""))
    return str(result)


def next_start_index(text: str) -> int | None:
    match = TRUNCATION_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


async def fetch_page_content(url: str, max_chunks: int = 20) -> str:
    """Fetch a URL directly through MCP using the exact user-provided URL."""
    fetch_tool = await get_fetch_tool()
    chunks: list[str] = []
    start_index = 0

    for _ in range(max_chunks):
        result = await fetch_tool.ainvoke(
            {
                "url": url,
                "start_index": start_index,
                "max_length": 50000,
            }
        )
        text = extract_tool_text(result)

        if "Failed to fetch" in text and "status code 404" in text:
            raise RuntimeError(
                f"The documentation URL returned HTTP 404: {url}"
            )
        if "Failed to fetch" in text:
            raise RuntimeError(text.strip())

        chunks.append(text)
        next_index = next_start_index(text)
        if next_index is None:
            break
        start_index = next_index

    return "\n\n".join(chunks)


async def build_agent():
    client = MultiServerMCPClient(get_mcp_config())
    tools = await client.get_tools()
    return create_agent(get_llm(), tools, system_prompt=SYSTEM_PROMPT)


def to_langchain_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    return messages


def extract_response(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return str(result)

    last = messages[-1]
    content = getattr(last, "content", last)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return str(content)


async def summarize_documentation(url: str, content: str) -> str:
    llm = get_llm()
    prompt = (
        f"The following content was fetched from {url}.\n\n"
        f"{content[:12000]}\n\n"
        "Provide a brief summary (3-5 sentences) of what this documentation "
        "covers and confirm it was loaded successfully."
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return extract_response({"messages": [response]})


async def fetch_documentation(url: str) -> tuple[str, str]:
    """Fetch documentation via MCP, then summarize with the LLM."""
    content = await fetch_page_content(url)
    summary = await summarize_documentation(url, content)
    return content, summary


async def answer_question(
    url: str,
    question: str,
    history: list[dict[str, str]],
    doc_content: str,
) -> str:
    agent = await build_agent()
    messages = to_langchain_messages(history)
    messages.append(
        HumanMessage(
            content=(
                f"Documentation URL (use this exact URL with the fetch tool if needed): "
                f"{url}\n\n"
                f"Fetched documentation content:\n{doc_content[:12000]}\n\n"
                f"User question: {question}"
            )
        )
    )
    result = await agent.ainvoke({"messages": messages})
    return extract_response(result)
