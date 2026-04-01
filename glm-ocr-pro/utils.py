import base64


def image_bytes_to_base64(file_bytes: bytes) -> str:
    """Encodes image bytes to base64 string for Ollama consumption."""
    return base64.b64encode(file_bytes).decode("utf-8")
