"""
Image-to-Structured-Data Extractor Streamlit app: upload an image and extract
validated, structured JSON from it using Mistral Large 3 and Instructor.
"""

import streamlit as st
from processor import extract_structured_data
from schemas import ProductCollection, InvoiceCollection 
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Mistral Vision Extractor", layout="wide")

st.title("📸 Image-to-Structured-Data")
st.write("Using **Mistral Large 3** for high-fidelity visual OCR and structured extraction.")

api_key = os.getenv("MISTRAL_API_KEY", "")

with st.sidebar:
    if not api_key:
        st.warning("MISTRAL_API_KEY is not set in your .env file.")
    schema_choice = st.selectbox("Select Extraction Schema", ["Product", "Invoice"])

    # These names now match the updated import above
    schema_map = {
        "Product": ProductCollection,
        "Invoice": InvoiceCollection
    }

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and api_key:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Source Image")
    
    with col2:
        if st.button("Extract Data"):
            with st.spinner("Mistral is analyzing the image..."):
                try:
                    uploaded_file.seek(0)
                    result = extract_structured_data(uploaded_file, schema_map[schema_choice], api_key)
                    st.success("Extracted Successfully!")
                    st.json(result.model_dump())
                except Exception as e:
                    st.error(f"Error: {e}")