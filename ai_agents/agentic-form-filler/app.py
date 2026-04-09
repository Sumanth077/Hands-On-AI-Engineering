import streamlit as st

st.set_page_config(page_title="Form-Fill AI", layout="wide")

import os
from dotenv import load_dotenv

load_dotenv()

import re
import json
import time
import base64
import uuid
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from extractor import extract_form_data
from llm import chat_and_update_fields_stream
from pdf_filler import (
    fill_pdf_with_mapping, fill_pdf_with_exact_mapping,
    get_pdf_base64, get_pdf_field_names, render_pdf_as_image, create_field_mapping
)

# ─────────────────────────────────────────────
# SESSION PERSISTENCE HELPERS
# ─────────────────────────────────────────────
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def _sanitize_sid(sid: str) -> str:
    """Strip any path separators or dots to prevent directory traversal."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', sid)

def save_session():
    sid = _sanitize_sid(st.session_state.get("session_id", ""))
    if not sid:
        return
    pdf_b64 = None
    if st.session_state.get("pdf_render"):
        pdf_b64 = base64.b64encode(st.session_state.pdf_render).decode("utf-8")
    data = {
        "session_id": sid,
        "form_name": st.session_state.get("target_filename", ""),
        "timestamp": datetime.datetime.now().isoformat(),
        "step": st.session_state.get("step", 1),
        "registry": st.session_state.get("registry", {}),
        "chat_history": st.session_state.get("chat_history", []),
        "target_pdf_path": st.session_state.get("target_pdf_path"),
        "target_filename": st.session_state.get("target_filename", ""),
        "markdown_context": st.session_state.get("markdown_context", ""),
        "source_context": st.session_state.get("source_context", ""),
        "pdf_field_names": st.session_state.get("pdf_field_names", []),
        "field_mapping": st.session_state.get("field_mapping", {}),
        "ready_to_fill": st.session_state.get("ready_to_fill", False),
        "pdf_render_b64": pdf_b64,
    }
    with open(os.path.join(SESSION_DIR, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_session(sid: str):
    path = os.path.join(SESSION_DIR, f"{sid}.json")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    st.session_state.session_id      = sid
    st.session_state.step            = data.get("step", 1)
    st.session_state.registry        = data.get("registry", {})
    st.session_state.chat_history    = data.get("chat_history", [])
    st.session_state.target_pdf_path = data.get("target_pdf_path")
    st.session_state.target_filename = data.get("target_filename", "")
    st.session_state.markdown_context = data.get("markdown_context", "")
    st.session_state.source_context  = data.get("source_context", "")
    st.session_state.pdf_field_names = data.get("pdf_field_names", [])
    st.session_state.field_mapping   = data.get("field_mapping", {})
    st.session_state.ready_to_fill   = data.get("ready_to_fill", False)
    b64 = data.get("pdf_render_b64")
    st.session_state.pdf_render      = base64.b64decode(b64) if b64 else None

def get_session_history():
    history = []
    for fname in os.listdir(SESSION_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(SESSION_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not d.get("form_name") and not d.get("registry"):
                continue
            history.append({
                "id":        d.get("session_id", fname.replace(".json", "")),
                "form_name": d.get("form_name") or d.get("target_filename") or "Untitled",
                "step":      d.get("step", 1),
                "timestamp": d.get("timestamp", ""),
                "mtime":     os.path.getmtime(fpath),
            })
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load session {fname}: {e}")
            continue
    return sorted(history, key=lambda x: x["mtime"], reverse=True)

def delete_session(sid: str):
    path = os.path.join(SESSION_DIR, f"{sid}.json")
    if os.path.exists(path):
        os.remove(path)

# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    qp = st.query_params
    if "session" in qp:
        load_session(qp["session"])
    else:
        new_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = new_id
        st.query_params["session"] = new_id

defaults = {
    "step": 1,
    "registry": {},
    "chat_history": [],
    "target_pdf_path": None,
    "target_filename": "",
    "pdf_render": None,
    "markdown_context": "",
    "source_context": "",
    "source_file_paths": [],
    "ready_to_fill": False,
    "pdf_field_names": [],
    "field_mapping": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# PAGE CONFIG & STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stChatFloatingInputContainer { bottom: 20px; }
    .session-card { padding: 6px 0; border-bottom: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("Form-Fill AI")
steps = ["Upload", "Analysis", "Extraction", "Audit", "Conversation", "Ready", "Live-Fill", "Verification", "Export"]
st.write(f"**Status:** {steps[st.session_state.step-1]} (Step {st.session_state.step}/9)")
st.progress(st.session_state.step / 9)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("1. Target PDF Form")
    uploaded_file = st.file_uploader("Upload the form to fill", type=["pdf"])

    if uploaded_file:
        os.makedirs("temp_forms", exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', uploaded_file.name)
        temp_path = f"temp_forms/{safe_name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.target_pdf_path = temp_path
        st.session_state.target_filename = uploaded_file.name
        st.success(f"Loaded: {uploaded_file.name}")
        if st.session_state.step == 1:
            st.session_state.step = 2

    st.header("2. Source Documents")
    source_files = st.file_uploader(
        "Upload CV, ID, or any source documents",
        type=["pdf", "png", "jpg", "jpeg", "docx"],
        accept_multiple_files=True,
    )
    if source_files:
        temp_source_paths = []
        os.makedirs("temp_sources", exist_ok=True)
        for sf in source_files:
            safe_sname = re.sub(r'[^a-zA-Z0-9_.-]', '_', sf.name)
            spath = f"temp_sources/{safe_sname}"
            with open(spath, "wb") as f:
                f.write(sf.getbuffer())
            temp_source_paths.append(spath)
        st.session_state.source_file_paths = temp_source_paths
        st.info(f"Attached {len(source_files)} source doc(s).")

    st.divider()

    extract_disabled = st.session_state.step < 2 or not source_files
    if st.button("Extract & Analyze", disabled=extract_disabled, use_container_width=True):
        all_paths = [st.session_state.target_pdf_path] + st.session_state.source_file_paths
        total = len(all_paths)
        progress_bar = st.progress(0, text="Starting extraction (parallel)...")

        results = {}
        with ThreadPoolExecutor(max_workers=min(total, 4)) as executor:
            futures = {executor.submit(extract_form_data, p): p for p in all_paths}
            completed = 0
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    results[path] = f"[Extraction failed: {e}]"
                completed += 1
                progress_bar.progress(
                    completed / total,
                    text=f"Extracted {completed}/{total}: {Path(path).name}",
                )
        progress_bar.empty()

        st.session_state.markdown_context = results.get(st.session_state.target_pdf_path, "")
        combined_source_md = ""
        for spath in st.session_state.source_file_paths:
            combined_source_md += f"\n--- SOURCE DOC: {Path(spath).name} ---\n"
            combined_source_md += results.get(spath, "")
        st.session_state.source_context = combined_source_md
        st.session_state.pdf_field_names = get_pdf_field_names(st.session_state.target_pdf_path)
        st.session_state.step = 3

        initial_prompt = (
            f"I have uploaded a target form and {len(source_files)} source document(s). "
            "Please extract all details from the source documents and pre-fill as many form fields as possible. "
            "List what you found and what is still missing."
        )
        st.session_state.chat_history.append({"role": "user", "content": initial_prompt})
        save_session()
        st.rerun()

    # Fill button
    st.divider()
    if st.session_state.ready_to_fill and st.session_state.step < 7:
        st.success("All details confirmed — ready to fill!")
        if st.button("Fill the Form Now!", use_container_width=True, type="primary"):
            if st.session_state.target_pdf_path and st.session_state.registry:
                with st.spinner("Mapping fields to PDF..."):
                    st.session_state.field_mapping = create_field_mapping(
                        list(st.session_state.registry.keys()),
                        st.session_state.target_pdf_path,
                    )
            st.session_state.step = 7
            save_session()
            st.rerun()
    elif 3 <= st.session_state.step < 7 and not st.session_state.ready_to_fill:
        st.info("Chat with the agent to provide missing details. The fill button appears once everything is confirmed.")

    if st.session_state.step >= 7:
        st.info("Filling in progress..." if st.session_state.step == 7 else "Verification mode active.")

    st.divider()
    col_new, col_reset = st.columns(2)
    with col_new:
        if st.button("New Session", use_container_width=True):
            save_session()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.query_params.clear()
            st.rerun()
    with col_reset:
        if st.button("Reset", use_container_width=True):
            sid = st.session_state.get("session_id")
            if sid:
                delete_session(sid)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.query_params.clear()
            st.rerun()

    # ── PREVIOUS SESSIONS PANEL ──────────────────────
    st.divider()
    st.subheader("Previous Sessions")
    history = get_session_history()
    current_sid = st.session_state.get("session_id", "")

    if not history:
        st.caption("No saved sessions yet.")
    else:
        for s in history[:10]:   # show last 10
            is_current = s["id"] == current_sid
            label = s["form_name"] or "Untitled"
            step_label = steps[min(s["step"] - 1, 8)]
            ts = s["timestamp"][:16].replace("T", " ") if s["timestamp"] else ""

            if is_current:
                st.markdown(f"**→ {label}** *(current)*  \n`{step_label}` · {ts}")
            else:
                col_load, col_del = st.columns([3, 1])
                with col_load:
                    if st.button(f"{label}", key=f"load_{s['id']}", use_container_width=True):
                        save_session()  # save current before switching
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        load_session(s["id"])
                        st.query_params["session"] = s["id"]
                        st.rerun()
                with col_del:
                    if st.button("✕", key=f"del_{s['id']}", use_container_width=True):
                        delete_session(s["id"])
                        st.rerun()
                st.caption(f"`{step_label}` · {ts}")


# ─────────────────────────────────────────────
# MAIN WORKSPACE
# ─────────────────────────────────────────────

if st.session_state.step == 7:
    st.header("Live Filling")
    progress_bar = st.progress(0, text="Preparing...")
    status_text = st.empty()
    live_img_slot = st.empty()

    registry_items = list(st.session_state.registry.items())
    total_items = len(registry_items)

    if total_items == 0:
        st.warning("No field data collected yet. The agent may not have finished auditing.")
        st.session_state.step = 8
        st.rerun()

    use_exact = bool(st.session_state.field_mapping)
    batch_size = 3

    for i in range(0, total_items, batch_size):
        batch = registry_items[i: i + batch_size]
        field_labels = ", ".join(k for k, _ in batch)
        progress_val = min((i + batch_size) / total_items, 1.0)
        progress_bar.progress(progress_val, text=f"Filling: {field_labels}")
        status_text.markdown(f"**Inking** `{field_labels}`")

        partial_data = dict(registry_items[: i + batch_size])
        if use_exact:
            stamped_bytes = fill_pdf_with_exact_mapping(
                st.session_state.target_pdf_path, partial_data, st.session_state.field_mapping
            )
        else:
            stamped_bytes = fill_pdf_with_mapping(st.session_state.target_pdf_path, partial_data)

        if stamped_bytes:
            st.session_state.pdf_render = stamped_bytes
            images = render_pdf_as_image(stamped_bytes)
            if images:
                live_img_slot.image(images, use_container_width=True)
        time.sleep(0.6)

    # Final full fill
    if use_exact:
        final_bytes = fill_pdf_with_exact_mapping(
            st.session_state.target_pdf_path,
            st.session_state.registry,
            st.session_state.field_mapping,
        )
    else:
        final_bytes = fill_pdf_with_mapping(st.session_state.target_pdf_path, st.session_state.registry)
    if final_bytes:
        st.session_state.pdf_render = final_bytes

    progress_bar.progress(1.0, text="Done!")
    status_text.markdown("**All fields filled.**")
    time.sleep(0.8)
    st.session_state.step = 8
    save_session()
    st.rerun()


elif st.session_state.step == 8:
    st.header("Verification")
    col_orig, col_fill = st.columns(2, gap="large")

    with col_orig:
        st.subheader("Original (Blank)")
        if st.session_state.target_pdf_path and os.path.exists(st.session_state.target_pdf_path):
            with open(st.session_state.target_pdf_path, "rb") as f:
                orig_bytes = f.read()
            orig_images = render_pdf_as_image(orig_bytes)
            if orig_images:
                st.image(orig_images, use_container_width=True)

    with col_fill:
        st.subheader("Filled")
        if st.session_state.pdf_render:
            filled_images = render_pdf_as_image(st.session_state.pdf_render)
            if filled_images:
                st.image(filled_images, use_container_width=True)
        else:
            st.warning("No filled PDF available.")

    st.divider()
    dl_col, next_col = st.columns(2)
    with dl_col:
        if st.session_state.pdf_render:
            st.download_button(
                label="Download Filled PDF",
                data=st.session_state.pdf_render,
                file_name=f"filled_{st.session_state.target_filename or 'form.pdf'}",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
    with next_col:
        if st.button("Done!", use_container_width=True):
            st.session_state.step = 9
            save_session()
            st.rerun()


elif st.session_state.step == 9:
    st.success("All done! Your filled form is ready.")
    if st.session_state.pdf_render:
        st.download_button(
            label="Download Filled PDF",
            data=st.session_state.pdf_render,
            file_name=f"filled_{st.session_state.target_filename or 'form.pdf'}",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    if st.button("Start a new form"):
        save_session()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.query_params.clear()
        st.rerun()


else:
    # Steps 1–6: Upload + Chat workspace
    col_preview, col_agent = st.columns([0.55, 0.45], gap="large")

    with col_preview:
        st.subheader("Document Preview")
        if st.session_state.target_pdf_path and os.path.exists(st.session_state.target_pdf_path):
            display_bytes = st.session_state.pdf_render
            if not display_bytes:
                with open(st.session_state.target_pdf_path, "rb") as f:
                    display_bytes = f.read()
            if display_bytes:
                preview_images = render_pdf_as_image(display_bytes)
                if preview_images:
                    st.image(preview_images, use_container_width=True)
        else:
            st.info("Upload a PDF form to see a preview here.")

    with col_agent:
        st.subheader("Agent")
        chat_container = st.container(height=560)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user" and msg["content"].startswith("I have uploaded a target form"):
                continue
            with chat_container.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with chat_container.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                stream = chat_and_update_fields_stream(
                    st.session_state.chat_history,
                    st.session_state.markdown_context,
                    st.session_state.source_context,
                    st.session_state.registry,
                    st.session_state.pdf_field_names,
                )

                try:
                    for kind, content in stream:
                        if kind == "TEXT":
                            full_response += content
                            placeholder.markdown(full_response + "▌")
                        elif kind == "FIELD_UPDATE":
                            st.session_state.registry.update(content)
                        elif kind == "SIGNAL" and content == "READY_TO_FILL":
                            st.session_state.ready_to_fill = True
                        elif kind == "DONE":
                            cleaned = content.replace("[READY_TO_FILL]", "").strip()
                            placeholder.markdown(cleaned)
                            st.session_state.chat_history.append({"role": "assistant", "content": cleaned})
                            if st.session_state.step == 3:
                                st.session_state.step = 4
                            save_session()
                            st.rerun()
                except Exception as e:
                    st.error(f"Stream error: {e}")
                    save_session()

        if prompt := st.chat_input(
            "Provide missing details or say 'leave blank' for any field...",
            disabled=st.session_state.step < 3,
        ):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            if st.session_state.step == 4:
                st.session_state.step = 5
            st.rerun()
