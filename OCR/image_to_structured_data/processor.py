import base64
import instructor
<<<<<<< HEAD
from mistralai import Mistral
=======
from mistralai.client import Mistral
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
from PIL import Image
import io

def process_and_encode_image(image_file, max_size=(2048, 2048)):
    """Resizes image to fit API limits and converts to base64."""
    img = Image.open(image_file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_structured_data(image_file, schema_model, api_key: str):
<<<<<<< HEAD
=======
    """Send the image to Mistral Large 3 and return data validated against the given schema."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    client = instructor.from_mistral(Mistral(api_key=api_key))
    base64_image = process_and_encode_image(image_file)

    return client.chat.completions.create(
        model="mistral-large-latest",
        response_model=schema_model,
        max_retries=1,  # Set retries to 1 to avoid hitting rate limits on errors
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all items found in this image into the requested structure."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ],
            }
        ],
    )