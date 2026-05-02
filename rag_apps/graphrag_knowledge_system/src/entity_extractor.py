import json
from src.llm_client import LLMClient

EXTRACTION_PROMPT = """You are an expert information extractor implementing the GraphRAG methodology.

Analyze the text and extract ALL significant entities and relationships.

Return ONLY a valid JSON object with this exact structure:
{{
  "entities": [
    {{
      "name": "Exact entity name, properly capitalized",
      "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT|TECHNOLOGY|PRODUCT|OTHER",
      "description": "Detailed description based on what the text says about this entity"
    }}
  ],
  "relationships": [
    {{
      "source": "Source entity name (must match an extracted entity)",
      "target": "Target entity name (must match an extracted entity)",
      "description": "Precise description of the relationship between source and target",
      "weight": 0.8
    }}
  ]
}}

Extraction rules:
- Extract 3–12 meaningful, specific entities; skip generic words
- Every relationship source and target MUST be an entity you extracted
- weight is 0.0–1.0 reflecting relationship strength or importance
- Return only valid JSON — no markdown fences, no extra text

Text to analyze:
{text}"""


class EntityExtractor:
    def __init__(self):
        self.llm = LLMClient()

    def extract(self, text: str, chunk_id: str) -> dict:
        prompt = EXTRACTION_PROMPT.format(text=text[:3000])

        try:
            raw = self.llm.complete(prompt, json_mode=True)
            data = json.loads(raw)
        except Exception:
            return {"entities": [], "relationships": []}

        entities = []
        for e in data.get("entities", []):
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "OTHER").strip()
            if name:
                entities.append(
                    {
                        "name": name,
                        "type": etype,
                        "description": (e.get("description") or "").strip(),
                        "chunk_id": chunk_id,
                    }
                )

        entity_names = {e["name"] for e in entities}
        relationships = []
        for r in data.get("relationships", []):
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            if src and tgt and src in entity_names and tgt in entity_names:
                try:
                    weight = float(r.get("weight", 0.5))
                except (TypeError, ValueError):
                    weight = 0.5
                relationships.append(
                    {
                        "source": src,
                        "target": tgt,
                        "description": (r.get("description") or "").strip(),
                        "weight": max(0.0, min(1.0, weight)),
                    }
                )

        return {"entities": entities, "relationships": relationships}
