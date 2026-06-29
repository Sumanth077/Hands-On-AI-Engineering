"""
Orchestrator Agent: decides whether an incoming query should be resolved
automatically or escalated to a human agent.

This sits between the router and the resolver. Every query passes through here
after VectorAI DB retrieval and before any response-generation LLM call.

The orchestrator is itself an LLM agent. It receives:
  - The raw customer query
  - The routing confidence score from VectorAI DB (how well the KB matched)
  - Pre-detected signals from a fast regex pass (sentiment, urgency, legal, repeat)

It reasons over all of those signals and outputs a structured JSON decision:
  - path               : "resolve" or "escalate"
  - sentiment_score    : 0.0-1.0 (agent's assessment of negative emotional tone)
  - urgency_score      : 0.0-1.0 (agent's assessment of time pressure)
  - legal_flag         : true/false
  - repeat_contact     : true/false
  - reasoning          : 1-2 sentence explanation of the decision
  - escalation_reasons : list of specific reasons (empty if resolving)

The regex patterns run first as a fast signal-detection pass. Their output is
passed to the agent as context, not as a decision. The agent has the final say.

If the LLM output cannot be parsed as valid JSON, the orchestrator falls back
to a deterministic regex-only decision so the pipeline never hard-fails.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from llama_cpp import Llama

from customer_query_routing_agent.config import (
    ESCALATION_SENTIMENT_THRESHOLD,
    ESCALATION_URGENCY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Signal patterns (used as pre-scan context for the agent, not as the decision)
# Each entry is (regex, weight) for scored dimensions, plain regex for flags.
# ---------------------------------------------------------------------------

_SENTIMENT_PATTERNS: list[tuple[str, float]] = [
    (r"\bunacceptable\b", 0.55),
    (r"\boutrageous\b", 0.65),
    (r"\bdisgraceful\b", 0.65),
    (r"\bfurious\b", 0.75),
    (r"\blivid\b", 0.75),
    (r"\bangry\b", 0.55),
    (r"\bterrible\b", 0.45),
    (r"\bawful\b", 0.45),
    (r"\bhorrible\b", 0.45),
    (r"\bdisgusting\b", 0.65),
    (r"\bappalling\b", 0.65),
    (r"\bfrustrated\b", 0.35),
    (r"\bfrustrating\b", 0.35),
    (r"\bfed up\b", 0.50),
    (r"\bsick of\b", 0.50),
    (r"\bnever again\b", 0.55),
    (r"\bworse.{0,15}ever\b", 0.55),
    (r"\bextremely (disappointed|upset|unhappy)\b", 0.55),
    (r"\bvery (upset|angry|frustrated)\b", 0.45),
    (r"\bdemand\b", 0.35),
    (r"\bcompletely unacceptable\b", 0.75),
    (r"\bhow dare\b", 0.65),
    (r"\bdisappointed\b", 0.30),
    (r"\blet down\b", 0.35),
    (r"\bhardly (believe|acceptable)\b", 0.40),
]

_URGENCY_PATTERNS: list[tuple[str, float]] = [
    (r"\burgent\b", 0.75),
    (r"\basap\b", 0.65),
    (r"\bas soon as possible\b", 0.55),
    (r"\bimmediately\b", 0.65),
    (r"\bright now\b", 0.55),
    (r"\bemergency\b", 0.85),
    (r"\bcritical\b", 0.65),
    (r"\btime.?sensitive\b", 0.65),
    (r"\bdeadline\b", 0.55),
    (r"\bstill (hasn.t|haven.t|not)\b", 0.45),
    (r"\bweeks? (ago|later|on)\b", 0.45),
    (r"\b(two|three|four|2|3|4)\s?(weeks?|months?)\b", 0.55),
    (r"\bmonths? ago\b", 0.55),
    (r"\bno response\b", 0.50),
    (r"\bignored\b", 0.55),
    (r"\bwaiting (for|since)\b", 0.35),
    (r"\bhave not (heard|received)\b", 0.45),
    (r"\bnot (arrived|received|delivered)\b", 0.35),
    (r"\boverdue\b", 0.55),
    (r"\blong.?overdue\b", 0.65),
    (r"\bdays? and (no|nothing)\b", 0.45),
]

_LEGAL_PATTERNS: list[str] = [
    r"\blawy?ers?\b",
    r"\blegal action\b",
    r"\b(take|taking) you to court\b",
    r"\bsue\b",
    r"\blawsuit\b",
    r"\bchargeback\b",
    r"\bfraud\b",
    r"\bscam\b",
    r"\bstolen\b",
    r"\bunauthori[sz]ed\b",
    r"\bconsumer rights\b",
    r"\btrading standards\b",
    r"\bombudsman\b",
    r"\bsmall claims\b",
    r"\breport (you|this|the company)\b",
    r"\bcriminal\b",
    r"\bpolice\b",
    r"\bregulator\b",
    r"\bwatch.?dog\b",
]

_REPEAT_CONTACT_PATTERNS: list[str] = [
    r"\bthird time\b",
    r"\b(2nd|second|3rd|third|fourth|4th) time (contact|call|email|reach)\b",
    r"\bagain and again\b",
    r"\bmultiple times\b",
    r"\bkeep (calling|contacting|emailing|reaching)\b",
    r"\bnobody (helps?|responds?|answer)\b",
    r"\bevery time i (call|contact|email)\b",
    r"\bspoke to (someone|an agent|support) (already|before|last)\b",
]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorDecision:
    """Result of the triage step produced by the orchestrator agent."""
    path: str                        # "resolve" | "escalate"
    sentiment_score: float           # 0.0 (neutral) to 1.0 (very negative)
    urgency_score: float             # 0.0 (calm) to 1.0 (highly urgent)
    legal_flag: bool                 # True if any legal/fraud language detected
    repeat_contact: bool             # True if customer mentions repeated attempts
    triggered_signals: list[str]     # human-readable list of matched pre-scan signals
    escalation_reasons: list[str]    # why escalation was triggered (empty if resolve)
    reasoning: str = ""              # agent's 1-2 sentence explanation of its decision


# ---------------------------------------------------------------------------
# Internal: fast regex pre-scan (produces context for the agent)
# ---------------------------------------------------------------------------

def _match_weighted(
    text: str,
    patterns: list[tuple[str, float]],
) -> tuple[float, list[str]]:
    total = 0.0
    matched: list[str] = []
    for pattern, weight in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            label = re.sub(r"\\b|[()\\?+.*^$]", "", pattern).strip()
            matched.append(label)
            total += weight
    return min(1.0, total), matched


def _match_flags(text: str, patterns: list[str]) -> tuple[bool, list[str]]:
    matched: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            label = re.sub(r"\\b|[()\\?+.*^$]", "", pattern).strip()
            matched.append(label)
    return bool(matched), matched


def _regex_prescan(query: str) -> dict:
    """Run the full regex pass and return a structured summary dict."""
    sentiment_score, sentiment_signals = _match_weighted(query, _SENTIMENT_PATTERNS)
    urgency_score, urgency_signals = _match_weighted(query, _URGENCY_PATTERNS)
    legal_flag, legal_signals = _match_flags(query, _LEGAL_PATTERNS)
    repeat_flag, repeat_signals = _match_flags(query, _REPEAT_CONTACT_PATTERNS)
    return {
        "sentiment_score": sentiment_score,
        "urgency_score": urgency_score,
        "legal_flag": legal_flag,
        "repeat_contact": repeat_flag,
        "all_signals": sentiment_signals + urgency_signals + legal_signals + repeat_signals,
    }


# ---------------------------------------------------------------------------
# Internal: fallback decision from regex alone (used if LLM parse fails)
# ---------------------------------------------------------------------------

def _regex_fallback(pre: dict, routing_confidence: float) -> OrchestratorDecision:
    """Deterministic regex-only decision used when LLM output cannot be parsed."""
    reasons: list[str] = []

    if pre["legal_flag"]:
        reasons.append("Legal or fraud language detected")

    if pre["sentiment_score"] >= ESCALATION_SENTIMENT_THRESHOLD:
        reasons.append(
            f"High negative sentiment (score {pre['sentiment_score']:.2f} >= "
            f"threshold {ESCALATION_SENTIMENT_THRESHOLD})"
        )

    if pre["urgency_score"] >= ESCALATION_URGENCY_THRESHOLD:
        reasons.append(
            f"High urgency (score {pre['urgency_score']:.2f} >= "
            f"threshold {ESCALATION_URGENCY_THRESHOLD})"
        )

    if pre["repeat_contact"]:
        reasons.append("Customer indicates this is a repeated contact attempt")

    if (
        routing_confidence < LOW_CONFIDENCE_THRESHOLD
        and (pre["sentiment_score"] > 0.2 or pre["urgency_score"] > 0.2)
    ):
        reasons.append(
            f"Low knowledge base match ({routing_confidence:.2f}) combined with "
            "negative or urgent tone"
        )

    return OrchestratorDecision(
        path="escalate" if reasons else "resolve",
        sentiment_score=pre["sentiment_score"],
        urgency_score=pre["urgency_score"],
        legal_flag=pre["legal_flag"],
        repeat_contact=pre["repeat_contact"],
        triggered_signals=pre["all_signals"],
        escalation_reasons=reasons,
        reasoning="(Fallback: regex-only decision -- LLM output could not be parsed.)",
    )


# ---------------------------------------------------------------------------
# Orchestrator agent system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a customer support triage agent. Your job is to read an incoming \
customer query and decide whether it should be handled automatically by the \
support system or escalated to a human agent.

You will be given:
1. The customer query
2. The knowledge base confidence score (how well the internal KB matched, 0.0-1.0)
3. Pre-detected signals from a fast pattern scan

Escalate if ANY of the following are true:
- Legal threats, fraud, chargebacks, regulatory complaints, or police involvement
- Strong anger, outrage, or clear distress (not mild disappointment)
- Time-critical issue that has already gone unresolved for too long
- Customer has explicitly contacted support multiple times without resolution
- Knowledge base confidence is low AND the customer sounds negative or urgent

If none of the above apply, resolve automatically.

Output ONLY a valid JSON object -- no extra text before or after. Schema:

{
  "path": "resolve",
  "sentiment_score": 0.0,
  "urgency_score": 0.0,
  "legal_flag": false,
  "repeat_contact": false,
  "reasoning": "One or two sentences explaining the routing decision.",
  "escalation_reasons": []
}

Field rules:
- "path": exactly "resolve" or "escalate"
- "sentiment_score": float 0.0 (calm) to 1.0 (extremely angry)
- "urgency_score": float 0.0 (no time pressure) to 1.0 (crisis-level urgency)
- "legal_flag": true only if the customer mentions legal action, fraud, police, etc.
- "repeat_contact": true only if the customer explicitly says they have contacted support before
- "reasoning": plain English, one or two sentences, explains why you chose this path
- "escalation_reasons": list of specific reasons; empty list [] if path is "resolve"
"""


def _build_agent_message(
    query: str,
    routing_confidence: float,
    pre: dict,
) -> str:
    signals_text = (
        ", ".join(pre["all_signals"]) if pre["all_signals"] else "none detected"
    )
    conf_label = (
        "high" if routing_confidence >= 0.80
        else "medium" if routing_confidence >= 0.45
        else "low"
    )
    return (
        f"Customer query:\n{query}\n\n"
        f"Knowledge base confidence: {routing_confidence:.2f} ({conf_label})\n\n"
        f"Pre-scan signals: {signals_text}\n"
        f"Pre-scan sentiment score: {pre['sentiment_score']:.2f}\n"
        f"Pre-scan urgency score: {pre['urgency_score']:.2f}\n"
        f"Legal language detected: {pre['legal_flag']}\n"
        f"Repeat contact detected: {pre['repeat_contact']}\n\n"
        "Output your JSON decision now."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def triage(query: str, routing_confidence: float, llm: Llama) -> OrchestratorDecision:
    """
    Analyse an incoming customer query and decide whether to resolve it
    automatically or escalate it to a human agent.

    The orchestrator is an LLM agent (Ministral 3B, same instance as the
    resolver). It receives the query, the VectorAI DB routing confidence,
    and pre-detected regex signals, then reasons over all of them to produce
    a structured JSON routing decision.

    If the LLM output cannot be parsed, falls back to a deterministic
    regex-only decision so the pipeline never hard-fails.

    Parameters
    ----------
    query : str
        The raw customer message.
    routing_confidence : float
        Best cosine similarity score from the VectorAI DB search (0-1).
    llm : Llama
        The already-loaded llama-cpp Llama instance (shared with resolver).

    Returns
    -------
    OrchestratorDecision
        Contains the path ("resolve" | "escalate"), scores, flags,
        triggered signals, escalation reasons, and the agent's reasoning.
    """
    pre = _regex_prescan(query)
    message = _build_agent_message(query, routing_confidence, pre)

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            max_tokens=400,
            temperature=0.1,
            repeat_penalty=1.0,
            response_format={"type": "json_object"},
        )
        raw = result["choices"][0]["message"]["content"].strip()
        data = json.loads(raw)
    except Exception:
        # LLM call failed or output was not valid JSON -- use regex fallback
        return _regex_fallback(pre, routing_confidence)

    # Validate path field -- if invalid, fall back
    path = data.get("path", "")
    if path not in ("resolve", "escalate"):
        return _regex_fallback(pre, routing_confidence)

    try:
        sentiment_score = float(data.get("sentiment_score", pre["sentiment_score"]))
        urgency_score = float(data.get("urgency_score", pre["urgency_score"]))
        legal_flag = bool(data.get("legal_flag", pre["legal_flag"]))
        repeat_contact = bool(data.get("repeat_contact", pre["repeat_contact"]))
        reasoning = str(data.get("reasoning", ""))
        escalation_reasons = [str(r) for r in data.get("escalation_reasons", [])]
    except (TypeError, ValueError):
        return _regex_fallback(pre, routing_confidence)

    # If the agent chose escalate but gave no reasons, derive from scores
    if path == "escalate" and not escalation_reasons:
        if legal_flag:
            escalation_reasons.append("Legal or fraud language detected")
        if sentiment_score >= ESCALATION_SENTIMENT_THRESHOLD:
            escalation_reasons.append(
                f"High negative sentiment (score {sentiment_score:.2f})"
            )
        if urgency_score >= ESCALATION_URGENCY_THRESHOLD:
            escalation_reasons.append(
                f"High urgency (score {urgency_score:.2f})"
            )
        if not escalation_reasons:
            escalation_reasons.append("Escalated by orchestrator agent judgment")

    return OrchestratorDecision(
        path=path,
        sentiment_score=max(0.0, min(1.0, sentiment_score)),
        urgency_score=max(0.0, min(1.0, urgency_score)),
        legal_flag=legal_flag,
        repeat_contact=repeat_contact,
        triggered_signals=pre["all_signals"],
        escalation_reasons=escalation_reasons,
        reasoning=reasoning,
    )
