"""LangGraph intake flow with tightly bounded legal-documentation scope."""

from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph

from legal_intake.llm import structured_response
from legal_intake.models import IntakeDecision, IntakeFacts, IntakeGraphState


SYSTEM_PROMPT = """You are a legal intake documentation assistant.
Your job is to collect a clear factual account for review by a qualified human professional.
You are not a lawyer. Do not give legal advice, evaluate a case, predict an outcome, interpret the law, recommend a strategy, or research legal authorities.

Use only details stated by the user. Preserve uncertainty as uncertainty. Do not invent names, dates, claims, evidence, or missing facts.
Ask one short, neutral, high-value follow-up question at a time. Do not ask for passwords, bank details, national identification numbers, or unnecessary sensitive data.

Return JSON only in this shape:
{
  "facts": {
    "matter_type": "string or null",
    "narrative": "string",
    "people_or_organisations": ["string"],
    "locations": ["string"],
    "dates_and_timeline": ["string"],
    "evidence_mentioned": ["string"],
    "impact_or_loss": ["string"],
    "missing_information": ["string"]
  },
  "next_question": "string or null",
  "status": "collecting or ready_for_review",
  "reason": "short explanation of why this is the next action"
}

Mark ready_for_review only when the narrative, people involved, approximate timeline, evidence status, and impact are each either supplied or explicitly unknown."""


def _render_history(history: list[dict[str, str]]) -> str:
    """Keep the model context limited to the intake transcript and current facts."""

    return "\n".join(f"{item['role'].upper()}: {item['content']}" for item in history[-12:])


def _evaluate(state: IntakeGraphState) -> IntakeGraphState:
    prompt = (
        "CURRENT FACT RECORD:\n"
        f"{json.dumps(state.get('facts', {}), ensure_ascii=False)}\n\n"
        "INTAKE CONVERSATION:\n"
        f"{_render_history(state['history'])}"
    )
    decision = IntakeDecision.model_validate(structured_response(SYSTEM_PROMPT, prompt))
    return {"facts": decision.facts.model_dump(), "decision": decision.model_dump()}


def build_intake_graph():
    """Build the bounded graph used after each user turn."""

    graph = StateGraph(IntakeGraphState)
    graph.add_node("evaluate_intake", _evaluate)
    graph.add_edge(START, "evaluate_intake")
    graph.add_edge("evaluate_intake", END)
    return graph.compile()


def process_turn(
    history: list[dict[str, str]], facts: IntakeFacts
) -> IntakeDecision:
    """Evaluate the complete intake transcript and return the next decision."""

    graph = build_intake_graph()
    result = graph.invoke(
        {
            "history": history,
            "facts": facts.model_dump(),
            "decision": {},
        }
    )
    return IntakeDecision.model_validate(result["decision"])


def draft_summary(facts: IntakeFacts) -> str:
    """Create a factual Markdown draft without calling the model again."""

    def bullets(values: list[str]) -> str:
        return "\n".join(f"- {value}" for value in values) if values else "- Not provided"

    matter = facts.matter_type or "Not specified"
    narrative = facts.narrative or "Not provided"
    return f"""# Intake summary

> This document records information supplied during an intake. It is not legal advice, a legal opinion, or a case assessment.

## Matter type

{matter}

## Narrative

{narrative}

## People or organisations mentioned

{bullets(facts.people_or_organisations)}

## Locations mentioned

{bullets(facts.locations)}

## Dates and timeline

{bullets(facts.dates_and_timeline)}

## Evidence mentioned

{bullets(facts.evidence_mentioned)}

## Impact or loss mentioned

{bullets(facts.impact_or_loss)}

## Information still to confirm

{bullets(facts.missing_information)}
"""
