import base64
import instructor
from groq import Groq
from PIL import Image
import io

def process_and_encode_image(image_file, max_size=(2048, 2048)):
    """
    Resizes the image if it's too large and converts to base64.
    2048x2048 is a 'sweet spot' for OCR: high enough to read small text,
    but well within the pixel limits of Groq/Llama.
    """
    # Open the image using PIL
    img = Image.open(image_file)
    
    # Convert RGBA to RGB (in case of PNGs with transparency)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Resize image while maintaining aspect ratio if it exceeds max_size
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Save to a bytes buffer
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90) # JPEG at 90 quality is efficient
    
    # Encode to base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_structured_data(image_file, schema_model, api_key: str):
    # Initialize the Groq client wrapped with Instructor
    client = instructor.from_groq(
        Groq(api_key=api_key), 
        mode=instructor.Mode.JSON 
    )
    
    # Process and encode the image
    base64_image = process_and_encode_image(image_file)

    return client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        response_model=schema_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Extract all relevant information from this image into the requested structured JSON format. Ensure high accuracy for numbers and dates."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ],
        temperature=0.1,
    )