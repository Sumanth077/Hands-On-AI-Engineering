"""Agent loop: sends the transcript + tool schema to AssemblyAI's LLM
Gateway on qwen3-next-80b-a3b (the model in the gateway's lineup that
supports tool calling / agentic workflows), and executes whichever GitHub
tools the model decides to call, until it returns a final answer.

Docs:
  https://www.assemblyai.com/docs/llm-gateway/quickstart
  https://www.assemblyai.com/docs/llm-gateway/tool-calling
  https://www.assemblyai.com/docs/llm-gateway/agentic-workflows
"""
import json
import os

from openai import OpenAI

from github_tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MODEL = "qwen3-next-80b-a3b"
MAX_TURNS = 8  # safety cap so a confused model can't loop forever

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
with open(_PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


def _client() -> OpenAI:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set. Check your .env file.")
    return OpenAI(base_url="https://llm-gateway.assemblyai.com/v1", api_key=api_key)


def run_agent(spoken_instruction: str) -> dict:
    """Runs the tool-calling loop. Returns a dict with the agent's final
    summary text and the list of tool calls made along the way, each as
    {"name": str, "args": dict, "result": Any}."""
    client = _client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spoken_instruction},
    ]
    tool_call_log = []

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_tokens=2000,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            return {
                "summary": message.content or "(no summary returned)",
                "tool_calls": tool_call_log,
            }

        # Record the assistant's tool-call request in the conversation history.
        messages.append(message.model_dump(exclude_none=True))

        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            print(f"  -> calling tool: {name}({args})")
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result = {"error": f"Unknown tool '{name}'"}
            else:
                try:
                    result = fn(**args)
                except Exception as exc:  # tool errors get fed back to the model, not raised
                    result = {"error": str(exc)}

            tool_call_log.append({"name": name, "args": args, "result": result})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )

    return {
        "summary": "Stopped after reaching the turn limit without a final answer — check the tool calls above.",
        "tool_calls": tool_call_log,
    }
