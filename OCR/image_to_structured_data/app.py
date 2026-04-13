import streamlit as st
from processor import extract_structured_data
from schemas import StructuredProduct, InvoiceData
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="OSS Vision Extractor", layout="wide")

st.title("📸Image-to-Structured-Data")
st.write("Using **Llama 4 Scout** on Groq for sub-second OCR extraction.")

with st.sidebar:
    # Use GROQ_API_KEY
    api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    schema_choice = st.selectbox("Select Extraction Schema", ["Product", "Invoice"])
    
    schema_map = {
        "Product": StructuredProduct,
        "Invoice": InvoiceData
    }

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and api_key:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Source Image")
    
    with col2:
        if st.button("Extract Data"):
            with st.spinner("Groq is thinking..."):
                try:
                    uploaded_file.seek(0)
                    result = extract_structured_data(uploaded_file, schema_map[schema_choice], api_key)
                    st.success("Extracted Successfully!")
                    st.json(result.model_dump())
                except Exception as e:
                    st.error(f"Error: {e}")