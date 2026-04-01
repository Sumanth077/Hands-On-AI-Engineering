import fitz
import os
import json
from llm import get_client, _parse_json_safely

def get_pdf_field_names(pdf_path: str) -> list:
    """Extract all interactive field names from a PDF."""
    doc = fitz.open(pdf_path)
    fields = []
    for page in doc:
        for widget in page.widgets():
            if widget.field_name:
                fields.append(widget.field_name)
    doc.close()
    return list(set(fields))

def render_pdf_to_image(pdf_bytes) -> bytes:
    """Renders the first page of a PDF to a high-resolution PNG."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    # Zoom for high-res (2x)
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

def create_field_mapping(logical_keys: list, pdf_field_names: list) -> dict:
    """Use LLM once to create a translation map between logical names and PDF internal names."""
    client = get_client()
    sys_prompt = (
        "You are a PDF mapping specialist. You will be given a list of 'Logical Field Names' (from an extraction) "
        "and a list of 'Internal PDF Field Names'.\n"
        "Your job is to create a 1-to-1 mapping JSON where the KEY is the Logical Name and the VALUE is the Internal PDF Name.\n"
        "Only map fields that clearly correspond. Return an empty JSON {} if no matches are found."
    )
    user_prompt = f"Logical Names: {logical_keys}\nPDF Names: {pdf_field_names}"
    
    response = client.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1
    )
    return _parse_json_safely(response.choices[0].message.content)

def fill_pdf_with_mapping(input_pdf_path: str, data: dict, mapping: dict) -> bytes:
    """Fast, non-LLM stamping using a pre-calculated map. Returns PDF bytes."""
    doc = fitz.open(input_pdf_path)
    
    # Invert mapping for easier lookup: Logical -> Internal
    # Data is {Logical: Value}
    
    fields_filled = 0
    for page in doc:
        for widget in page.widgets():
            internal_name = widget.field_name
            # Find which logical key maps to this internal name
            logical_key = next((k for k, v in mapping.items() if v == internal_name), None)
            
            if logical_key and logical_key in data:
                val = data[logical_key]
                if val is not None:
                    widget.field_value = str(val)
                    widget.update()
                    fields_filled += 1
                    
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def fill_pdf_form(input_pdf_path: str, data: dict, output_pdf_path: str):
    """Legacy wrapper for compatibility - combines mapping and filling."""
    pdf_fields = get_pdf_field_names(input_pdf_path)
    mapping = create_field_mapping(list(data.keys()), pdf_fields)
    pdf_bytes = fill_pdf_with_mapping(input_pdf_path, data, mapping)
    with open(output_pdf_path, "wb") as f:
        f.write(pdf_bytes)
