import base64
import io

from mistralai.client import Mistral
from PIL import Image

from schemas import Prescription


SYSTEM_PROMPT = """You are a medical prescription digitizer. Your task is to carefully read and extract
structured information from prescription images, including handwritten ones.

Guidelines:
- Extract ALL medications listed, even if handwriting is difficult
- Decode common medical abbreviations: QD/OD=once daily, BID=twice daily, TID=three times daily,
  QID=four times daily, PRN=as needed, PO=by mouth, SIG=directions, Rx=prescription
- For illegible sections, add a descriptive entry to illegible_fields (e.g. "medication 2 dosage unclear")
- Do NOT guess drug names if truly illegible — mark them as illegible instead
- Extract dosage units precisely: mg, mcg, ml, units, etc.
- If a field is absent from the prescription, leave it as null
"""


def image_to_base64(image_file) -> tuple[str, str]:
    """Convert uploaded file to base64 string. Returns (base64_data, media_type)."""
    image_bytes = image_file.read()
    image_file.seek(0)

    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format or "JPEG"
    media_type_map = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp",
    }
    media_type = media_type_map.get(fmt, "image/jpeg")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return b64, media_type


def extract_prescription(image_file, api_key: str) -> Prescription:
    """Send image to Mistral Large 3 and extract structured prescription data."""
    b64_data, media_type = image_to_base64(image_file)

    client = Mistral(api_key=api_key)

    result = client.chat.parse(
        response_format=Prescription,
        model="mistral-large-latest",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all structured information from this medical prescription image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{media_type};base64,{b64_data}",
                    },
                ],
            },
        ],
    )

    return result.choices[0].message.parsed
