"""
Receipt extractor using Gemma 4 E2B vision via llama-cpp-python.

Downloads the model and mmproj on first run (from HuggingFace cache),
then runs fully offline on subsequent runs.
"""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download

MODEL_REPO = "unsloth/gemma-4-E2B-it-GGUF"
MODEL_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_REPO = "ggml-org/gemma-4-E2B-it-GGUF"
MMPROJ_FILE = "mmproj-gemma-4-E2B-it-Q8_0.gguf"

EXTRACTION_PROMPT = """\
Look at this receipt or invoice image and extract all available information.
Return ONLY a valid JSON object with exactly these fields (use null for any field not found):

{
  "vendor": "store or restaurant name",
  "date": "YYYY-MM-DD",
  "line_items": [
    {"name": "item name", "quantity": 1, "price": 0.00}
  ],
  "subtotal": 0.00,
  "tax": 0.00,
  "total": 0.00,
  "category": "one of: Food & Dining, Groceries, Transport, Shopping, Utilities, Healthcare, Entertainment, Other"
}

Return only the JSON object, no explanation, no markdown code fences.\
"""


class ReceiptExtractor:
    def __init__(self):
        print("Downloading model files if needed (first run only)...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        mmproj_path = hf_hub_download(repo_id=MMPROJ_REPO, filename=MMPROJ_FILE)

        print("Loading vision model...")
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
        self.llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
            logits_all=True,
        )
        print("Model ready.")

    def extract(self, image: Image.Image) -> dict:
        """
        Run vision extraction on a preprocessed PIL image.
        Returns a structured dict with receipt fields.
        """
        # Encode image to base64 JPEG
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"

        result = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.1,
        )

        raw = result["choices"][0]["message"]["content"].strip()
        return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    """
    Extract JSON from the model response robustly.
    Handles cases where the model wraps the output in markdown fences.
    """
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object anywhere in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Return an empty record so the UI can still show and let the user fill in manually
    return {
        "vendor": None,
        "date": None,
        "line_items": [],
        "subtotal": None,
        "tax": None,
        "total": None,
        "category": "Other",
    }
