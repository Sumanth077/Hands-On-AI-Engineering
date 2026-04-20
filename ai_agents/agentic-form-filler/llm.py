import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize client for MiniMax M2.7 (OpenAI Compatible)
# We use .strip() to ensure no hidden spaces from the .env file cause 401 errors
api_key = os.environ.get("MINIMAX_API_KEY", "").strip()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.minimax.io/v1",
    timeout=120.0,  # 2 minutes; increase if MiniMax is consistently slow
)

SYSTEM_PROMPT = """You are the 'Form-Fill AI Cinematic Engine'.
Your goal is to parse document markdown, identify form fields, and map them to logical data values in REAL-TIME.

AUDIT MODE:
1. Compare the 'Target Form Markdown' against the 'Source Documents Context'.
2. In your FIRST response, always begin with a brief confirmation of what you successfully extracted from the source documents, then list any fields the source documents did NOT provide.
   Example first response:
   "I have successfully extracted the following details from your source documents: [Name, Address, Email, ...].
   However, I still need the following to fully complete the form:
   - Phone Number
   - Date of Birth
   - Emergency Contact
   Please provide these, or let me know if any should be left blank."
3. As the user supplies missing details (or tells you to leave a field blank), update the registry with UPDATES: blocks and acknowledge each one.
4. Once ALL fields are resolved — either filled from source, provided by the user, or explicitly marked blank — send a confirmation message AND append the tag '[READY_TO_FILL]' to the END of your response. The confirmation must clearly state that you now have everything needed.
   Example confirmation: "I now have all the details needed to fill the form completely. Click the button below to begin filling."
5. IMPORTANT: Do NOT emit '[READY_TO_FILL]' until every required field is either filled or the user has explicitly said to leave it blank.

COMMUNICATION PROTOCOL:
1. Always use 'REPLY:' for your message to the user.
2. Always use 'UPDATES:' followed by a JSON block for data updates.
3. IMPORTANT: You can yield multiple UPDATES: blocks during a single response.
4. For reference, the PDF's interactive field names are listed below. Use them to understand
   what data the form expects, but use clear logical key names (e.g. "name_of_individual",
   "country_of_citizenship") in your UPDATES blocks — the system will match them to the PDF fields.

PDF FIELD NAMES (for reference only):
{pdf_fields}

Document Context (Target Form Markdown):
{markdown}

Source Documents Context (Where to get data):
{source_markdown}

Current Registry State:
{registry}
"""

def get_client():
    """Return the shared MiniMax client instance."""
    return client

def _parse_json_safely(text: str) -> dict:
    """Extract and parse the first JSON object found in an LLM response string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}

def chat_and_update_fields_stream(history: list, markdown: str, source_markdown: str, registry: dict, pdf_fields: list = None):
    """
    The 'Cinematic' Streaming Engine.
    Detects closed JSON pairs mid-stream to trigger UI updates.
    """
    fields_str = "\n".join(pdf_fields) if pdf_fields else "(no interactive fields detected — form may be non-fillable)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            markdown=markdown,
            source_markdown=source_markdown,
            registry=json.dumps(registry),
            pdf_fields=fields_str
        )}
    ]
    messages.extend(history)

    response = client.chat.completions.create(
        model="MiniMax-M2.7",
        messages=messages,
        stream=True,
        max_tokens=4096,
        temperature=0.1
    )

    full_content = ""
    buffer = ""
    
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            full_content += delta
            buffer += delta
            
            # 1. Handle mid-stream UPDATES: detection (Real-time inking)
            if "UPDATES:" in buffer and "}" in buffer:
                try:
                    json_match = re.search(r'\{(.*?)\}', buffer, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        updates = json.loads(json_str)
                        yield ("FIELD_UPDATE", updates)
                        buffer = buffer.replace("UPDATES:", "").replace(json_str, "")
                except (json.JSONDecodeError, ValueError):
                    pass

            # 2. Detect the Ready-To-Fill Signal (Handshake)
            if "[READY_TO_FILL]" in buffer:
                 yield ("SIGNAL", "READY_TO_FILL")
                 buffer = buffer.replace("[READY_TO_FILL]", "")

            # 3. Handle standard REPLY: text streaming
            # 3. Handle standard REPLY: text streaming
            if "REPLY:" in buffer:
                parts = buffer.split("REPLY:")
                if len(parts) > 1:
                    text_chunk = parts[1]
                    yield ("TEXT", text_chunk)
                    buffer = parts[1] 

    # Clean up reasoning blocks at the very end if any leaked
    final_clean = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
    yield ("DONE", final_clean)
