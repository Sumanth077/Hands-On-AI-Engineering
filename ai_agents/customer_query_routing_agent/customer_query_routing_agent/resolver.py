"""
Response generation using Ministral 3B via llama-cpp-python.

Two paths:

  resolve()   -- standard RAG-grounded customer support response.
                 Receives context from all four VectorAI DB sources and
                 generates a direct, policy-grounded answer.

  escalate()  -- structured escalation message.
                 The LLM writes only a brief empathetic acknowledgment of the
                 specific situation (2-3 sentences). Everything else -- the
                 human handoff confirmation, case reference, response time,
                 and contact channel -- is assembled from a guaranteed template
                 the LLM never touches. This ensures the customer always
                 receives a clear, unambiguous handoff message regardless of
                 what the LLM produces.
"""

from __future__ import annotations

import time

from llama_cpp import Llama

from customer_query_routing_agent.config import LLM_FILENAME, LLM_REPO_ID
from customer_query_routing_agent.orchestrator import OrchestratorDecision
from customer_query_routing_agent.router import RoutingResult

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_RESOLUTION_SYSTEM_PROMPT = """\
You are a professional customer support agent. Your job is to resolve customer queries \
clearly and helpfully.

Guidelines:
- Ground your response in the context provided. Do not invent policies, procedures, or figures.
- Be concise, warm, and direct.
- If context from multiple sources is relevant, synthesise it into a single coherent answer.
- If the context includes a resolved ticket thread with a similar past issue, you may reference \
the resolution approach but adapt it to the current customer's situation.
- If the context does not fully address the issue, acknowledge it and describe the correct next step.
- Never fabricate order numbers, dates, dollar amounts, or account details.
"""

_EMPATHY_SYSTEM_PROMPT = """\
You are a customer support agent writing the opening of an escalation message.

Write 2-3 sentences that:
1. Acknowledge specifically what the customer described -- their frustration, the problem,
   or how long they have been waiting. Reference the actual situation, not generic language.
2. Apologise sincerely without using hollow phrases like "I apologise for any inconvenience."

Do NOT:
- Attempt to resolve the issue.
- Mention case numbers, response times, or next steps -- those will be added separately.
- Write more than 3 sentences.
- Use hollow corporate language.

Output only the acknowledgment sentences. Nothing else.
"""


# ---------------------------------------------------------------------------
# Context formatting (shared between both paths)
# ---------------------------------------------------------------------------

def _format_context_block(doc: dict) -> str:
    label = doc["source_label"]
    dept = doc["department"]

    if doc["source"] == "tickets":
        header = f"[{label} | {doc['ticket_id']} | {dept}]"
        return f"{header}\nIssue: {doc['question']}\nThread:\n{doc['answer']}"

    if doc["source"] == "docs":
        title = doc.get("title", "Policy Document")
        header = f"[{label} | {title} | {dept}]"
        return f"{header}\n{doc['answer']}"

    # FAQ or memory
    header = f"[{label} | {dept}]"
    return f"{header}\nQ: {doc['question']}\nA: {doc['answer']}"


def _build_resolution_message(query: str, routing: RoutingResult) -> str:
    lines = [
        f"Department: {routing.department}",
        f"Customer query: {query}",
        "",
        "--- Context retrieved from knowledge base ---",
    ]
    if routing.context_docs:
        for doc in routing.context_docs[:5]:
            lines.append("")
            lines.append(_format_context_block(doc))
    else:
        lines.append("(No matching context found.)")
    lines += [
        "",
        "--- End of context ---",
        "",
        "Write a helpful, accurate response to the customer query using the context above.",
    ]
    return "\n".join(lines)


def _build_empathy_message(query: str, decision: OrchestratorDecision) -> str:
    """
    Narrow prompt for the LLM: produce only a short empathetic acknowledgment.
    No context documents -- the LLM does not need them and should not see them here.
    """
    reasons_text = (
        "; ".join(decision.escalation_reasons)
        if decision.escalation_reasons
        else "customer requires escalation"
    )
    return (
        f"Customer query:\n{query}\n\n"
        f"Escalation reasons: {reasons_text}\n\n"
        "Write 2-3 sentences acknowledging this customer's specific situation."
    )


def _case_reference() -> str:
    """Generate a deterministic case reference from the current Unix timestamp."""
    return f"ESC-{int(time.time()) % 1_000_000:06d}"


def _escalation_template(
    empathy: str,
    department: str,
    case_ref: str,
) -> str:
    """
    Assemble the final escalation message. The empathy paragraph comes from the LLM.
    All structural facts are hardcoded here and never left to LLM discretion.
    """
    return (
        f"{empathy}\n\n"
        f"I have forwarded your case to a member of our {department} team. "
        f"A human agent will review your case and contact you directly within 2-4 business hours "
        f"on the email address registered to your account.\n\n"
        f"Your case reference is **{case_ref}**. Please keep this number -- you can quote it "
        f"in any follow-up so our team can pick up exactly where we left off.\n\n"
        f"If you need to reach us before then, reply to this message or call us on 1-800-555-0100."
    )


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------

class Resolver:
    def __init__(self) -> None:
        print(f"Loading language model: {LLM_REPO_ID}/{LLM_FILENAME}")
        self.llm = Llama.from_pretrained(
            repo_id=LLM_REPO_ID,
            filename=LLM_FILENAME,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        print("Language model ready.")

    def resolve(self, query: str, routing: RoutingResult) -> str:
        """
        Generate a grounded customer support response.
        Called when the orchestrator routes to the resolution path.
        """
        user_message = _build_resolution_message(query, routing)
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _RESOLUTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=600,
            temperature=0.2,
            repeat_penalty=1.1,
        )
        return result["choices"][0]["message"]["content"].strip()

    def escalate(
        self,
        query: str,
        routing: RoutingResult,
        decision: OrchestratorDecision,
    ) -> str:
        """
        Produce a structured escalation message.

        The LLM writes only a brief empathetic acknowledgment of the specific
        situation (2-3 sentences). The human handoff confirmation, case reference,
        response time, and contact channel are assembled by _escalation_template()
        and are never left to LLM discretion.

        This guarantees the customer always receives a clear, explicit handoff
        message regardless of what the LLM produces.
        """
        user_message = _build_empathy_message(query, decision)
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _EMPATHY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=120,
            temperature=0.3,
            repeat_penalty=1.1,
        )
        empathy = result["choices"][0]["message"]["content"].strip()
        case_ref = _case_reference()
        return _escalation_template(empathy, routing.department, case_ref)
