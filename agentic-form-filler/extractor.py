import os
from pathlib import Path
from landingai_ade import LandingAIADE

def extract_form_data(file_path: str) -> str:
    """
    Extracts purely semantic text, form fields, and structure from a document 
    using the Landing AI Agentic Document Extraction (ADE) SDK.
    """
    api_key = os.environ.get("VISION_AGENT_API_KEY", "").strip()
    if not api_key:
        raise ValueError("VISION_AGENT_API_KEY is not set.")
    
    client = LandingAIADE(apikey=api_key)
    response = client.parse(document=Path(file_path), model="dpt-2-latest")
    
    if hasattr(response, 'markdown') and response.markdown:
        return response.markdown
    elif hasattr(response, 'text') and response.text:
        return response.text
    else:
        return str(response)
