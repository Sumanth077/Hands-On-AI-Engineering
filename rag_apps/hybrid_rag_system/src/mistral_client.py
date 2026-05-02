from mistralai.client import Mistral

_SYSTEM = (
    "You are an expert research assistant. Answer questions using the provided context "
    "which comes from two sources: a knowledge graph (entities and relationships) and "
    "relevant document chunks. Synthesize both sources to give a comprehensive, accurate answer. "
    "If the context is insufficient, say so clearly and state what information is missing."
)

_USER_TEMPLATE = """\
CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, well-structured answer based on the context above."""


def generate_answer(client: Mistral, question: str, fused_context: str) -> str:
    resp = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_TEMPLATE.format(
                context=fused_context,
                question=question,
            )},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content
