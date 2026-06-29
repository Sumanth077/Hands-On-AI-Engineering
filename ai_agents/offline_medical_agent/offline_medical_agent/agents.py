"""
Two-agent pipeline.

Retrieval Agent  - semantic search over local Qdrant to find the single
                   most relevant clinical protocol.
Protocol Agent   - feeds the retrieved protocol as grounded context to the
                   local LLM and generates a concise, actionable response.
                   Strictly forbidden from hallucinating outside the protocol.
"""

from __future__ import annotations

from llama_cpp import Llama

from offline_medical_agent.retriever import ProtocolRetriever

MODEL_REPO_ID = "lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF"
MODEL_FILENAME = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"

PROTOCOL_AGENT_SYSTEM = """You are a clinical decision support assistant deployed at a remote, \
offline medical facility.

STRICT RULES - read carefully before responding:
1. Base your answer EXCLUSIVELY on the protocol text provided in the user message.
2. Do NOT add drug dosages, procedures, or recommendations not present in the protocol.
3. If the protocol does not fully address the situation, say so explicitly and quote the \
relevant section instead of guessing.
4. Be concise and actionable. Clinicians need clear guidance, not lengthy disclaimers.
5. Structure your response as: Assessment, Immediate Actions, and Follow-up (when applicable).
"""


class OfflineMedicalAgent:
    def __init__(
        self,
        db_path: str = "./qdrant_data",
        protocols_dir: str = "./protocols",
        repo_id: str = MODEL_REPO_ID,
        filename: str = MODEL_FILENAME,
    ):
        self.retriever = ProtocolRetriever(db_path=db_path)
        if not self.retriever.is_populated():
            print("Ingesting protocols...")
            n = self.retriever.ingest(protocols_dir)
            print(f"Loaded {n} protocol(s) into local database.")

        print("Loading language model, this may take a moment on first run...")
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        print("Model ready.")

    # ------------------------------------------------------------------
    # Retrieval Agent
    # ------------------------------------------------------------------

    def retrieval_agent(self, query: str) -> dict | None:
        """Return the single most relevant protocol or None if DB is empty."""
        results = self.retriever.retrieve(query, top_k=1)
        if not results:
            return None
        hit = results[0]
        return {
            "title": hit.payload["title"],
            "content": hit.payload["content"],
            "source": hit.payload["source"],
            "score": round(hit.score, 3),
        }

    # ------------------------------------------------------------------
    # Protocol Agent
    # ------------------------------------------------------------------

    def protocol_agent(self, query: str, protocol: dict) -> str:
        """Generate a grounded response from the retrieved protocol."""
        user_message = (
            f"Patient situation: {query}\n\n"
            f"Retrieved protocol: {protocol['title']}\n"
            f"{'=' * 60}\n"
            f"{protocol['content']}\n"
            f"{'=' * 60}\n\n"
            "Based strictly on the protocol above, provide a concise clinical response."
        )
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": PROTOCOL_AGENT_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            max_tokens=600,
            temperature=0.1,
            repeat_penalty=1.1,
        )
        return result["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict:
        """
        Run both agents and return a result dict with keys:
          response, protocol_title, protocol_content, source, score
        """
        if not query.strip():
            return {
                "response": "Please describe the patient situation.",
                "protocol_title": "",
                "protocol_content": "",
                "source": "",
                "score": 0.0,
            }

        protocol = self.retrieval_agent(query)
        if protocol is None:
            return {
                "response": "No protocols loaded. Use the 'Load Protocols' button first.",
                "protocol_title": "",
                "protocol_content": "",
                "source": "",
                "score": 0.0,
            }

        response = self.protocol_agent(query, protocol)
        return {
            "response": response,
            "protocol_title": protocol["title"],
            "protocol_content": protocol["content"],
            "source": protocol["source"],
            "score": protocol["score"],
        }
