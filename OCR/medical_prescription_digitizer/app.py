import os
import io

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from extractor import extract_prescription
from validator import validate_prescription_drugs
from schemas import Prescription

load_dotenv()

st.set_page_config(
    page_title="Medical Prescription Digitizer",
    page_icon="💊",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .valid-drug   { background:#d4edda; border-left:4px solid #28a745;
                    padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .invalid-drug { background:#f8d7da; border-left:4px solid #dc3545;
                    padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .field-row    { display:flex; gap:12px; flex-wrap:wrap; margin-top:4px; }
    .field-chip   { background:#e9ecef; border-radius:12px;
                    padding:2px 10px; font-size:0.85rem; color:#495057; }
    .illegible-box{ background:#fff3cd; border-left:4px solid #ffc107;
                    padding:10px 14px; border-radius:6px; margin-top:8px; }
    .section-card { background:#f8f9fa; border-radius:10px;
                    padding:18px 22px; margin-bottom:16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💊 Medical Prescription Digitizer")
st.caption("Upload a prescription image — handwritten or printed — to extract and validate its contents.")
st.divider()


# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Mistral API Key",
        value=os.getenv("MISTRAL_API_KEY", ""),
        type="password",
        help="Your Mistral API key. Store it in .env as MISTRAL_API_KEY to avoid re-entering.",
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a prescription image\n"
        "2. Mistral Large 3 reads and interprets it\n"
        "3. Drug names are validated via [RxNorm](https://rxnav.nlm.nih.gov/)\n"
        "4. Results are displayed with validation status"
    )
    st.markdown("---")
    st.markdown("**Supported formats:** JPG, PNG, WEBP")


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear photo of a handwritten or printed prescription.",
)

if uploaded_file:
    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        uploaded_file.seek(0)

    with col_info:
        st.subheader("Extraction & Validation")

        if not api_key:
            st.error("Please enter your Mistral API key in the sidebar to proceed.")
            st.stop()

        if st.button("🔍 Digitize Prescription", type="primary", use_container_width=True):
            prescription: Prescription | None = None

            with st.spinner("Reading prescription with Mistral Large 3…"):
                try:
                    prescription = extract_prescription(uploaded_file, api_key)
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    st.stop()

            with st.spinner("Validating drug names against RxNorm…"):
                try:
                    prescription = validate_prescription_drugs(prescription)
                except Exception as e:
                    st.warning(f"Validation step encountered an error: {e}")

            # ── Display results ───────────────────────────────────────────────
            st.success("Prescription processed successfully!")

            # Patient / Doctor / Date
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Patient", prescription.patient_name or "—")
            with meta_cols[1]:
                st.metric("Doctor", prescription.doctor_name or "—")
            with meta_cols[2]:
                st.metric("Date", prescription.date or "—")
            st.markdown("</div>", unsafe_allow_html=True)

            # Medications
            st.markdown("#### Medications")

            if not prescription.medications:
                st.warning("No medications could be extracted from this prescription.")
            else:
                validated_count = sum(1 for m in prescription.medications if m.is_validated)
                total = len(prescription.medications)
                st.caption(f"{validated_count}/{total} drug names validated against RxNorm")

                for med in prescription.medications:
                    css_class = "valid-drug" if med.is_validated else "invalid-drug"
                    status_icon = "✅" if med.is_validated else "⚠️"

                    chips = ""
                    if med.dosage:
                        chips += f'<span class="field-chip">💊 {med.dosage}</span>'
                    if med.frequency:
                        chips += f'<span class="field-chip">🕐 {med.frequency}</span>'
                    if med.duration:
                        chips += f'<span class="field-chip">📅 {med.duration}</span>'

                    note_html = ""
                    if med.validation_note:
                        note_color = "#155724" if med.is_validated else "#721c24"
                        note_html = f'<div style="font-size:0.8rem;color:{note_color};margin-top:4px">{med.validation_note}</div>'

                    st.markdown(
                        f"""
                        <div class="{css_class}">
                            <strong>{status_icon} {med.drug_name}</strong>
                            <div class="field-row">{chips}</div>
                            {note_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Notes
            if prescription.notes:
                st.markdown("#### Additional Notes")
                st.info(prescription.notes)

            # Illegible fields
            if prescription.illegible_fields:
                st.markdown(
                    '<div class="illegible-box">'
                    "<strong>⚠️ Illegible / Uncertain Fields</strong><br>"
                    + "<br>".join(f"• {f}" for f in prescription.illegible_fields)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            # Raw JSON expander
            with st.expander("🗂️ View Raw Extracted JSON"):
                st.json(prescription.model_dump())

else:
    # Empty state
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;color:#6c757d;">
            <div style="font-size:4rem">📋</div>
            <h3>No prescription uploaded yet</h3>
            <p>Upload a JPG, PNG, or WEBP image of a prescription above to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
