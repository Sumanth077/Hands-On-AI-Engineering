import streamlit as st
import os

# Only importing local mode now
from ocr_local import check_ollama_status, extract_markdown_local_stream, warmup_local_model

# Design System Injection
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _looks_like_html_table(text: str) -> bool:
    lowered = text.lower()
    return "<table" in lowered and ("<tr" in lowered or "<td" in lowered)

st.set_page_config(
    page_title="GLM-OCR Pro",
    page_icon="🧾",
    layout="wide",
)

# Header Section
st.title("GLM-OCR Pro")
st.markdown("##### Modern Intelligence for Structured Document Extraction")
st.divider()

if "ocr_output" not in st.session_state:
    st.session_state["ocr_output"] = ""

# Layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    with st.container(border=True):
        st.subheader("⚙️ Settings")
        
        ok, status = check_ollama_status()
        status_color = "green" if ok else "red"
        st.markdown(f"**Ollama Status:** <span style='color:{status_color}'>{'●' if ok else '○'} {status}</span>", unsafe_allow_html=True)
        
        if st.button("Warm Up Model", use_container_width=True, type="primary"):
            with st.spinner("Loading GLM-OCR weights..."):
                warmup_local_model()
                st.toast("Model warmed up on GPU!")
        
        st.divider()
        
        uploaded_file = st.file_uploader(
            "Drop your document here",
            type=["png", "jpg", "jpeg", "pdf"],
            help="Supported: PNG, JPG, JPEG, PDF",
        )
        
        with st.expander("Advanced Tuning", expanded=True):
            speed_profile = st.selectbox(
                "Performance Profile",
                ["Balanced", "Fast", "Ultra Fast"],
                index=2,
                help="Profiles adjust resolution and grayscale for optimal VRAM usage.",
            )
            
            if speed_profile == "Balanced":
                default_side, default_render, default_gray = 1300, 1.8, False
            elif speed_profile == "Fast":
                default_side, default_render, default_gray = 1000, 1.5, True
            else: # Ultra Fast
                default_side, default_render, default_gray = 800, 1.2, True

            max_pdf_pages = st.slider("Max PDF Pages", 1, 20, 3)
            use_grayscale = st.checkbox("Enable Grayscale (Recommended)", value=default_gray)
            max_side = st.number_input("Max Side (px)", 400, 2400, default_side, step=100)
            pdf_render_scale = st.number_input("PDF Rendering Scale", 1.0, 3.0, float(default_render), step=0.1)

    if st.button("🚀 Extract Knowledge", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Please provide an image or PDF to start.")
        else:
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type or "application/octet-stream"
            
            st.session_state["ocr_output"] = ""
            # Placeholder for streaming output
            output_placeholder = st.empty()
            full_text = ""
            
            try:
                for chunk in extract_markdown_local_stream(
                    file_bytes=file_bytes,
                    mime_type=mime_type,
                    max_pdf_pages=max_pdf_pages,
                    optimize_images=True,
                    max_side=max_side,
                    grayscale=use_grayscale,
                    pdf_render_scale=pdf_render_scale,
                ):
                    full_text += chunk
                    # Prettify rendering during stream
                    output_placeholder.markdown(full_text)
                
                st.session_state["ocr_output"] = full_text
                st.balloons()
            except Exception as exc:
                st.error(f"Extraction Pipeline Error: {exc}")

with col2:
    if st.session_state["ocr_output"]:
        with st.container(border=True):
            st.subheader("🧾 Extracted Results")
            
            # Action Row
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label="💾 Download .md",
                    data=st.session_state["ocr_output"],
                    file_name="extraction.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with btn_col2:
                # Built-in Copy strategy via st.code
                st.info("💡 Copy button is built into the viewer below.")

            st.divider()
            
            # The Viewer
            if _looks_like_html_table(st.session_state["ocr_output"]):
                st.markdown(st.session_state["ocr_output"], unsafe_allow_html=True)
            else:
                st.markdown(st.session_state["ocr_output"])
            
            st.divider()
            st.caption("Raw Markdown Source (Safe Copying)")
            st.code(st.session_state["ocr_output"], language="markdown")
    else:
        st.info("Upload a document and initiate extraction to see structured data here.")
