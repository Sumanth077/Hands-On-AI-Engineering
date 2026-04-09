import fitz  # PyMuPDF
import os
import base64
import json

def fill_pdf_with_mapping(input_pdf_path, mapping, output_pdf_path=None):
    """
    High-speed In-Memory Stamping Engine.
    Uses fuzzy matching and coordinate caching for 50ms frame renders.
    """
    try:
        if not os.path.exists(input_pdf_path):
            return None
            
        doc = fitz.open(input_pdf_path)
        
        # Normalize mapping keys for better matching
        norm_mapping = {str(k).lower().replace("_", " ").replace(" ", ""): v for k, v in mapping.items()}

        # Stamping Loop
        for page in doc:
            widgets = page.widgets()
            if widgets:
                for widget in widgets:
                    field_name = str(widget.field_name).lower().replace("_", " ").replace(" ", "")
                    
                    # Fuzzy match the logical registry key to the PDF field ID
                    match_found = False
                    if field_name in norm_mapping:
                        widget.field_value = str(norm_mapping[field_name])
                        match_found = True
                    else:
                        for norm_key, val in norm_mapping.items():
                            if norm_key in field_name or field_name in norm_key:
                                widget.field_value = str(val)
                                match_found = True
                                break
                    
                    if match_found:
                        widget.update()

        # If we just want the bytes for the UI
        if not output_pdf_path:
            pdf_bytes = doc.write()
            doc.close()
            return pdf_bytes
            
        # Or if we want to save an actual file
        doc.save(output_pdf_path)
        doc.close()
        return True
    except Exception as e:
        print(f"pdf_filler error: {e}")
        return None

def get_pdf_base64(pdf_bytes):
    """Converts raw PDF bytes to a base64 string for the Streamlit iframe."""
    return base64.b64encode(pdf_bytes).decode('utf-8')

def get_pdf_field_names(pdf_path):
    """Returns all interactive widget field names present in the PDF."""
    try:
        doc = fitz.open(pdf_path)
        fields = []
        for page in doc:
            for widget in (page.widgets() or []):
                if widget.field_name:
                    fields.append(widget.field_name)
        doc.close()
        return list(dict.fromkeys(fields))  # deduplicate, preserve order
    except Exception as e:
        print(f"get_pdf_field_names error: {e}")
        return []

def build_semantic_field_map(pdf_path: str) -> dict:
    """
    For each widget in the PDF, extract surrounding text to give a semantic label.
    Returns {widget_field_name: descriptive_text_context}.
    """
    try:
        doc = fitz.open(pdf_path)
        field_labels = {}
        for page in doc:
            spans = []
            for b in page.get_text("dict")["blocks"]:
                if b["type"] == 0:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            t = span["text"].strip()
                            if t:
                                spans.append((span["bbox"], t))

            for widget in (page.widgets() or []):
                wr = widget.rect
                # Tight column-above search: same x-column as widget, 22px above
                tight_above = fitz.Rect(wr.x0 - 5, wr.y0 - 22, wr.x1 + 5, wr.y0 - 1)
                matches = [s[1] for s in spans if fitz.Rect(s[0]).intersects(tight_above)]
                field_labels[widget.field_name] = " ".join(matches)

        doc.close()
        return field_labels
    except Exception as e:
        print(f"build_semantic_field_map error: {e}")
        return {}


def create_field_mapping(logical_keys: list, pdf_path: str) -> dict:
    """
    One LLM call that maps the registry's logical key names to the PDF's exact internal
    widget field names.  Returns {logical_name: pdf_internal_name}.
    Uses surrounding-text context extracted from the PDF so the LLM can match
    opaque names like 'f_1[0]' to logical names like 'name_of_individual'.
    Called once, right before live inking starts.
    """
    if not logical_keys or not pdf_path:
        return {}
    field_context = build_semantic_field_map(pdf_path)
    if not field_context:
        return {}

    from llm import get_client, _parse_json_safely
    client = get_client()

    # Build a human-readable list: widget_name => nearby text label
    context_lines = "\n".join(
        f'  "{name}": "{label}"'
        for name, label in field_context.items()
    )
    sys_prompt = (
        "You are a PDF field mapping specialist. "
        "You will receive:\n"
        "1. Logical Field Names — semantic keys extracted from source documents (e.g. 'name_of_individual').\n"
        "2. PDF Widget Fields — the actual internal field names from the PDF, each with nearby label text as context.\n\n"
        "Create a 1-to-1 JSON mapping: KEY = logical name, VALUE = matching PDF widget name.\n"
        "Use the label context to figure out which widget corresponds to each logical key.\n"
        "Only include pairs you are confident about. Return {} if nothing matches.\n"
        "Return ONLY a valid JSON object, no markdown, no explanation."
    )
    user_prompt = (
        f"Logical Field Names:\n{json.dumps(logical_keys, indent=2)}\n\n"
        f"PDF Widget Fields (name => nearby label text):\n{{\n{context_lines}\n}}"
    )
    response = client.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    return _parse_json_safely(response.choices[0].message.content)


def fill_pdf_with_exact_mapping(input_pdf_path: str, data: dict, mapping: dict):
    """
    Fill PDF widgets using a pre-calculated logical→internal mapping.
    data:    {logical_name: value}
    mapping: {logical_name: pdf_internal_widget_name}
    Returns PDF bytes.
    """
    try:
        doc = fitz.open(input_pdf_path)
        internal_to_logical = {v: k for k, v in mapping.items()}
        for page in doc:
            for widget in (page.widgets() or []):
                internal_name = widget.field_name
                logical_key = internal_to_logical.get(internal_name)
                if logical_key and logical_key in data and data[logical_key] is not None:
                    widget.field_value = str(data[logical_key])
                    widget.update()
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes
    except Exception as e:
        print(f"fill_pdf_with_exact_mapping error: {e}")
        return None


def render_pdf_as_image(pdf_bytes, zoom=1.5):
    """Renders all pages of a PDF (bytes) to a list of PNG image bytes."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))
        doc.close()
        return images  # list of PNG bytes, one per page
    except Exception as e:
        print(f"render_pdf_as_image error: {e}")
        return []
