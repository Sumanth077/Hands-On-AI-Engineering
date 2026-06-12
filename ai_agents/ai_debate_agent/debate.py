"""LangGraph multi-agent debate graph.

Three agents collaborate on a single graph:

* **Debater A** - argues the "for" position (gemini-3.0-flash via Orq.ai).
* **Debater B** - argues the "against" position (mistral-small-latest via Orq.ai).
* **Judge**     - scores every argument and declares a winner (kimi-k2.6 via Orq.ai).

All three agents are routed through the Orq.ai OpenAI-compatible router.

The graph alternates A -> B for a configurable number of rounds, then routes to the
judge. ``run_debate`` is a generator so the UI can stream the debate as it unfolds.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

# Position labels used throughout the transcript and prompts.
FOR = "For"
AGAINST = "Against"
DEBATER_A = "Debater A"
DEBATER_B = "Debater B"


class Argument(TypedDict):
    """A single turn in the debate."""

    speaker: str  # "Debater A" or "Debater B"
    position: str  # "For" or "Against"
    round: int
    content: str


class DebateState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""

    topic: str
    max_rounds: int
    round_num: int
    transcript: List[Argument]
    verdict: dict


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
# All three agents are routed through Orq.ai via the OpenAI-compatible SDK:
#   * Debater A -> google-ai/gemini-3-flash-preview
#   * Debater B -> mistral/mistral-small-latest
#   * Judge     -> moonshotai/kimi-k2.6
DEBATER_A_MODEL = os.getenv("DEBATER_A_MODEL", "google-ai/gemini-3-flash-preview")
DEBATER_B_MODEL = os.getenv("DEBATER_B_MODEL", "mistral/mistral-small-latest")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "moonshotai/kimi-k2.6")

# Some Orq-routed models (e.g. gemini-3-flash-preview, kimi-k2.6) reject temperature != 1.
ORQ_TEMPERATURE = float(os.getenv("ORQ_TEMPERATURE", "1"))


def _orq_client():
    """Shared Orq.ai router client used by every agent."""
    from openai import OpenAI

    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ORQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(
        base_url="https://api.orq.ai/v3/router",
        api_key=api_key,
    )


# --------------------------------------------------------------------------- #
# Prompt helpers
# --------------------------------------------------------------------------- #
def _history_block(transcript: List[Argument]) -> str:
    """Render the debate so far as plain text for the next debater."""
    if not transcript:
        return "(No arguments yet - you are opening the debate.)"
    lines = []
    for arg in transcript:
        lines.append(f"[Round {arg['round']}] {arg['speaker']} ({arg['position']}):\n{arg['content']}")
    return "\n\n".join(lines)


def _debater_system_prompt(position: str, topic: str) -> str:
    """Return the system prompt assigning a debater to its position on the given topic."""
    return (
        f"You are a sharp, persuasive debater arguing the **{position.upper()}** position "
        f'on the topic: "{topic}".\n'
        "Make a focused, well-structured case. Use clear logic, concrete evidence or "
        "examples, and rhetorical persuasion. Directly rebut your opponent's most recent "
        "point when one exists. Stay on your assigned side at all times. Keep each turn to "
        "roughly 120-180 words and do not break character or mention that you are an AI."
    )


def _debater_user_prompt(position: str, round_num: int, transcript: List[Argument]) -> str:
    """Return the user-turn prompt instructing the debater to deliver their argument for this round."""
    return (
        f"Debate so far:\n\n{_history_block(transcript)}\n\n"
        f"It is now round {round_num}. Deliver your {position} argument."
    )


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def _debater_a_node(state: DebateState) -> DebateState:
    """Call Debater A's model and append its For-side argument to the transcript."""
    client = _orq_client()
    round_num = state["round_num"]
    messages = [
        {"role": "system", "content": _debater_system_prompt(FOR, state["topic"])},
        {"role": "user", "content": _debater_user_prompt(FOR, round_num, state["transcript"])},
    ]
    response = client.chat.completions.create(
        model=DEBATER_A_MODEL,
        messages=messages,
        temperature=ORQ_TEMPERATURE,
    )
    argument: Argument = {
        "speaker": DEBATER_A,
        "position": FOR,
        "round": round_num,
        "content": (response.choices[0].message.content or "").strip(),
    }
    return {"transcript": state["transcript"] + [argument]}


def _debater_b_node(state: DebateState) -> DebateState:
    """Call Debater B's model, append its Against-side argument, and increment the round counter."""
    client = _orq_client()
    round_num = state["round_num"]
    messages = [
        {"role": "system", "content": _debater_system_prompt(AGAINST, state["topic"])},
        {"role": "user", "content": _debater_user_prompt(AGAINST, round_num, state["transcript"])},
    ]
    response = client.chat.completions.create(
        model=DEBATER_B_MODEL,
        messages=messages,
        temperature=ORQ_TEMPERATURE,
    )
    argument: Argument = {
        "speaker": DEBATER_B,
        "position": AGAINST,
        "round": round_num,
        "content": (response.choices[0].message.content or "").strip(),
    }
    return {
        "transcript": state["transcript"] + [argument],
        "round_num": round_num + 1,
    }


def _route_after_b(state: DebateState) -> str:
    """Loop back for another round, or hand off to the judge."""
    if state["round_num"] <= state["max_rounds"]:
        return "continue"
    return "judge"


_JUDGE_SYSTEM = (
    "You are an impartial debate judge. Score every argument on three criteria, each from "
    "1 to 10: logic, evidence, and persuasiveness. Be fair and consistent. After scoring, "
    "tally totals for each debater, declare a single winner, and write a concise verdict.\n\n"
    "Respond with ONLY valid JSON in exactly this shape (no markdown, no extra text):\n"
    "{\n"
    '  "arguments": [\n'
    '    {"speaker": "Debater A", "round": 1, "logic": 8, "evidence": 7, '
    '"persuasiveness": 9, "comment": "..."}\n'
    "  ],\n"
    '  "totals": {"Debater A": 0, "Debater B": 0},\n'
    '  "winner": "Debater A" | "Debater B" | "Tie",\n'
    '  "verdict": "A few sentences explaining the decision."\n'
    "}"
)


def _judge_node(state: DebateState) -> DebateState:
    client = _orq_client()
    transcript_text = _history_block(state["transcript"])
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f'Topic: "{state["topic"]}"\n'
                f"Debater A argued FOR; Debater B argued AGAINST.\n\n"
                f"Full transcript:\n\n{transcript_text}\n\n"
                "Score each argument and declare the winner now."
            ),
        },
    ]
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        temperature=ORQ_TEMPERATURE,
    )
    content = response.choices[0].message.content or ""
    return {"verdict": _parse_verdict(content, state["transcript"])}


# --------------------------------------------------------------------------- #
# Verdict parsing
# --------------------------------------------------------------------------- #
def _parse_verdict(raw: str, transcript: List[Argument]) -> dict:
    """Best-effort extraction of the judge's JSON, with graceful fallbacks."""
    parsed: Optional[dict] = None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None

    if not isinstance(parsed, dict):
        return {
            "arguments": [],
            "totals": {DEBATER_A: 0, DEBATER_B: 0},
            "winner": "Tie",
            "verdict": raw.strip() or "The judge did not return a parsable verdict.",
            "raw": raw.strip(),
        }

    # Recompute totals from per-argument scores when possible (more reliable than the LLM's math).
    totals = {DEBATER_A: 0, DEBATER_B: 0}
    for arg in parsed.get("arguments", []):
        speaker = arg.get("speaker")
        if speaker in totals:
            totals[speaker] += (
                int(arg.get("logic", 0))
                + int(arg.get("evidence", 0))
                + int(arg.get("persuasiveness", 0))
            )
    if any(totals.values()):
        parsed["totals"] = totals
        if totals[DEBATER_A] > totals[DEBATER_B]:
            parsed.setdefault("winner", DEBATER_A)
        elif totals[DEBATER_B] > totals[DEBATER_A]:
            parsed.setdefault("winner", DEBATER_B)

    parsed.setdefault("totals", totals)
    parsed.setdefault("winner", "Tie")
    parsed.setdefault("verdict", "")
    parsed.setdefault("arguments", [])
    return parsed


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    """Construct and compile the debate graph."""
    graph = StateGraph(DebateState)
    graph.add_node("debater_a", _debater_a_node)
    graph.add_node("debater_b", _debater_b_node)
    graph.add_node("judge", _judge_node)

    graph.add_edge(START, "debater_a")
    graph.add_edge("debater_a", "debater_b")
    graph.add_conditional_edges(
        "debater_b",
        _route_after_b,
        {"continue": "debater_a", "judge": "judge"},
    )
    graph.add_edge("judge", END)
    return graph.compile()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_debate(topic: str, rounds: int) -> Iterator[DebateState]:
    """Run the debate, yielding the full state after each agent turn.

    Args:
        topic: The debate topic supplied by the user.
        rounds: Number of rounds (1-5). Each round is one A turn + one B turn.

    Yields:
        The accumulated :class:`DebateState` after every node, ending with a
        state that contains the judge's ``verdict``.
    """
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Please enter a debate topic.")
    rounds = max(1, min(5, int(rounds)))

    app = build_graph()
    initial: DebateState = {
        "topic": topic,
        "max_rounds": rounds,
        "round_num": 1,
        "transcript": [],
        "verdict": {},
    }
    # stream_mode="values" emits the full state snapshot after each node runs.
    for state in app.stream(initial, stream_mode="values"):
        yield state
