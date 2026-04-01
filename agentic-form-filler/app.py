import streamlit as st
import os
import tempfile
import json
import uuid
import base64
import datetime
from dotenv import load_dotenv

from extractor import extract_form_data
from llm import extract_form_and_prefill, validate_field_input, chat_and_update_fields_stream
from pdf_filler import get_pdf_field_names, create_field_mapping, fill_pdf_with_mapping, render_pdf_to_image

load_dotenv()
st.set_page_config(page_title="Form-Fill AI: Workspace", page_icon="📝", layout="wide")

SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def save_session():
    if "session_id" in st.session_state:
        filepath = os.path.join(SESSION_DIR, f"{st.session_state.session_id}.json")
        dump = {
            "form_state": st.session_state.get("form_state", {}),
            "chat_history": st.session_state.get("chat_history", []),
            "target_pdf_path": st.session_state.get("target_pdf_path"),
            "target_filename": st.session_state.get("target_filename"),
            "live_fill_complete": st.session_state.get("live_fill_complete", False),
            "pdf_bytes_b64": base64.b64encode(st.session_state.get("pdf_bytes")).decode('utf-8') if st.session_state.get("pdf_bytes") else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "form_name": st.session_state.get("target_filename")
        }
        with open(filepath, "w") as f:
            json.dump(dump, f)

def load_session(sid: str):
    filepath = os.path.join(SESSION_DIR, f"{sid}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state.form_state = data.get("form_state", {})
            st.session_state.chat_history = data.get("chat_history", [])
            st.session_state.target_pdf_path = data.get("target_pdf_path")
            st.session_state.target_filename = data.get("target_filename")
            st.session_state.live_fill_complete = data.get("live_fill_complete", False)
            pb64 = data.get("pdf_bytes_b64")
            if pb64:
                st.session_state.pdf_bytes = base64.b64decode(pb64)

def get_session_history():
    """Scans the sessions folder and returns sorted metadata."""
    history = []
    if not os.path.exists(SESSION_DIR):
        return []
    for f in os.listdir(SESSION_DIR):
        if f.endswith(".json"):
            sid = f.replace(".json", "")
            try:
                with open(os.path.join(SESSION_DIR, f), "r") as json_f:
                    data = json.load(json_f)
                    # We only show sessions that have at least some data
                    if data.get("target_filename") or data.get("form_state"):
                        m_time = os.path.getmtime(os.path.join(SESSION_DIR, f))
                        history.append({
                            "id": sid,
                            "name": data.get("target_filename", sid),
                            "time": data.get("timestamp"),
                            "mtime": m_time
                        })
            except Exception:
                continue
    # Sort by mtime descending
    return sorted(history, key=lambda x: x["mtime"], reverse=True)

# State Initialization
if "session_id" not in st.session_state:
    qp = st.query_params
    if "session" in qp:
        st.session_state.session_id = qp["session"]
        load_session(qp["session"])
    else:
        new_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = new_id
        st.query_params["session"] = new_id

if "form_state" not in st.session_state:
    st.session_state.form_state = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "target_pdf_path" not in st.session_state:
    st.session_state.target_pdf_path = None
if "target_markdown" not in st.session_state:
    st.session_state.target_markdown = ""
if "source_markdown" not in st.session_state:
    st.session_state.source_markdown = ""
if "ready_to_generate" not in st.session_state:
    st.session_state.ready_to_generate = False
if "field_mapping" not in st.session_state:
    st.session_state.field_mapping = {}
if "is_live_filling" not in st.session_state:
    st.session_state.is_live_filling = False
if "current_filling_idx" not in st.session_state:
    st.session_state.current_filling_idx = -1
if "live_fill_complete" not in st.session_state:
    st.session_state.live_fill_complete = False
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

def render_pdf_iframe(pdf_source, height=700):
    if pdf_source is None:
        return st.warning("PDF data is still being processed or is currently unavailable.")
        
    if isinstance(pdf_source, bytes):
        b64 = base64.b64encode(pdf_source).decode('utf-8')
    else:
        with open(pdf_source, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
    pdf_html = f'<iframe src="data:application/pdf;base64,{b64}#toolbar=0" width="100%" height="{height}" type="application/pdf" style="border:1px solid #D1D5DB; border-radius:8px;"></iframe>'
    st.markdown(pdf_html, unsafe_allow_html=True)

# 🎩 Activity Tracker
st.title("📝 Form-Fill AI: Workspace")

if not st.session_state.form_state:
    st.info("👋 Welcome! Start by uploading your documents in the sidebar.")
else:
    # 🏎️ Stepper Logic
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.write("✅ 1. Analysis")
    with s2:
        st.write("🔵 2. Conversation" if not st.session_state.get("live_fill_complete") else "✅ 2. Conversation")
    with s3:
        st.write("🔵 3. Auto-Filling" if st.session_state.get("is_live_filling") else ("✅ 3. Auto-Filling" if st.session_state.get("live_fill_complete") else "⚪ 3. Auto-Filling"))
    with s4:
        st.write("🏁 4. Ready" if st.session_state.get("live_fill_complete") else "⚪ 4. Final Review")
    st.divider()

# Sidebar Config
with st.sidebar:
    st.header("1. Upload Documents")
    target_form = st.file_uploader("Target Blank Form (PDF/Img)", type=["pdf", "png", "jpg", "jpeg"], key="target")
    source_doc = st.file_uploader("Source Data Context (CV, ID)", type=["pdf", "png", "jpg", "jpeg"], key="source")
    
    if st.button("🚀 Extract & Pre-Fill", use_container_width=True):
        st.session_state.pop("pdf_bytes", None)
        if not target_form:
            st.error("Target Form is required!")
        elif not os.environ.get("VISION_AGENT_API_KEY") or not os.environ.get("MINIMAX_API_KEY"):
            st.error("Ensure API keys are provided in your .env file.")
        else:
            with st.spinner("Extracting Target Layout (Landing AI)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t1:
                    t1.write(target_form.getvalue())
                    target_path = t1.name
                st.session_state.target_markdown = extract_form_data(target_path)
                st.session_state.target_pdf_path = target_path # Persist the file for final PDF rendering
                st.session_state.target_filename = target_form.name
            
            src_md = ""
            if source_doc:
                with st.spinner("Extracting Source Document (Landing AI)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t2:
                        t2.write(source_doc.getvalue())
                        src_path = t2.name
                    src_md = extract_form_data(src_path)
                    st.session_state.source_markdown = src_md
                    os.unlink(src_path)
                    
            with st.spinner("Minimax M2.7 is Mapping Fields..."):
                extracted_dict = extract_form_and_prefill(st.session_state.target_markdown, src_md)
                
                # Build rich state dict
                st.session_state.form_state = {}
                for k, v in extracted_dict.items():
                    if k == "READY_TO_GENERATE":
                        st.session_state.ready_to_generate = bool(v)
                        continue
                    if v and str(v).strip():
                        st.session_state.form_state[k] = {"value": str(v), "source": "AUTO", "status": "FILLED"}
                    else:
                        st.session_state.form_state[k] = {"value": "", "source": "MISSING", "status": "MISSING"}
                
                # Pre-calculate internal PDF field mapping for fast live stamping
                with st.spinner("Preparing Live Stamping Engine..."):
                    pdf_names = get_pdf_field_names(st.session_state.target_pdf_path)
                    st.session_state.field_mapping = create_field_mapping(list(st.session_state.form_state.keys()), pdf_names)

                # Proactive AI Chat Initialization
                st.session_state.chat_history = []
                missing_keys = [k for k, v in st.session_state.form_state.items() if v["status"] == "MISSING"]
                if missing_keys:
                    msg = "I successfully extracted the context! However, I still need your inputs for the following missing fields:\n"
                    for m in missing_keys:
                        msg += f"- **{m}**\n"
                    msg += "\nPlease provide them here so I can seamlessly complete the document."
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
                else:
                    msg = "Excellent! I was able to successfully extract and map every single field from your source document. I am generating your final PDF now!"
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
                
                # Save the explicitly initialized session state to disk!
                save_session()

    # 📚 Session History Section
    st.divider()
    st.subheader("📚 Recent Forms")
    history = get_session_history()
    if not history:
        st.caption("No previous forms found.")
    else:
        for item in history[:8]: # Show last 8 sessions
            t_str = ""
            if item["time"]:
                try:
                    dt = datetime.datetime.fromisoformat(item["time"])
                    t_str = dt.strftime("%b %d, %H:%M")
                except:
                    t_str = ""
            else:
                # Fallback to file mtime
                dt = datetime.datetime.fromtimestamp(item["mtime"])
                t_str = dt.strftime("%b %d, %H:%M")
            
            label = f"📄 {item['name'][:20]}... ({t_str})" if len(item['name']) > 20 else f"📄 {item['name']} ({t_str})"
            if st.button(label, key=f"hist_{item['id']}", use_container_width=True):
                # FORCE RELOAD MORPHING
                st.session_state.session_id = item["id"]
                st.query_params["session"] = item["id"]
                load_session(item["id"])
                st.rerun()

# Main Content Area
# Step 3 & 4: Data Collection Phase
if st.session_state.get("form_state") and not st.session_state.get("is_live_filling") and not st.session_state.get("live_fill_complete"):
    main_col, side_col = st.columns([1.5, 1])
    
    with side_col:
        st.subheader("🤖 AI Agent")
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        with st.expander("📊 Logical Data Registry", expanded=True):
            registry_df = []
            for k, v in st.session_state.form_state.items():
                registry_df.append({
                    "Field": k,
                    "Value": v["value"] if v["status"] != "SKIPPED" else "⏩ SKIPPED",
                    "Status": "✅ READY" if v["status"] != "MISSING" else "❌ MISSING"
                })
            st.table(registry_df)

        missing_keys = [k for k, v in st.session_state.form_state.items() if v["status"] == "MISSING"]
        is_complete = len(missing_keys) == 0
        
        if is_complete or st.session_state.get("ready_to_generate"):
             st.success("🎯 I've gathered all necessary information!")
             if st.button("🚀 Step 6: Start Auto-Filling!", type="primary", use_container_width=True):
                 st.session_state["is_live_filling"] = True
                 st.rerun()

        if prompt := st.chat_input("Ex: My SSN is 111-22-3333"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with chat_container:
                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    stream_gen, result_ref = chat_and_update_fields_stream(st.session_state.chat_history, st.session_state.get("form_state"))
                    
                    # Dynamic Interlaced Loop
                    message_placeholder = st.empty()
                    full_reply = ""
                    
                    for signal in stream_gen:
                        stype = signal[0]
                        if stype == "TEXT":
                            full_reply += signal[1]
                            message_placeholder.markdown(full_reply + "▌")
                        elif stype == "FIELD_UPDATE":
                            fid, val = signal[1], signal[2]
                            fs = st.session_state.get("form_state")
                            if fid in fs:
                                if val == "[[BLANK]]":
                                    fs[fid]["value"] = ""
                                    fs[fid]["status"] = "SKIPPED"
                                else:
                                    fs[fid]["value"] = str(val)
                                    fs[fid]["status"] = "FILLED"
                    
                    message_placeholder.markdown(full_reply)
                    updates = result_ref.get("updates_json", {})
                    if "READY_TO_GENERATE" in updates:
                         st.session_state["ready_to_generate"] = bool(updates["READY_TO_GENERATE"])

            st.session_state.chat_history.append({"role": "assistant", "content": full_reply})
            save_session()
            st.rerun()

    with main_col:
        st.subheader("📄 Original Document")
        render_pdf_iframe(st.session_state.get("target_pdf_path"), height=750)

# Step 7: Live Action Painting (Flicker-Free Mode)
elif st.session_state.get("is_live_filling"):
    import time
    st.markdown("<h2 style='text-align: center;'>⚡ AI Agent: Writing to PDF...</h2>", unsafe_allow_html=True)
    
    # Progress & Log
    all_fields = [f for f in st.session_state.form_state.items() if f[1]["status"] != "SKIPPED"]
    total = len(all_fields)
    progress_bar = st.progress(0)
    log_slot = st.empty()
    img_slot = st.empty()
    
    current_data = {}
    for i, (fid, fmeta) in enumerate(all_fields):
        log_slot.markdown(f"🎨 **Painting field:** `{fid}`... ({i+1}/{total})")
        current_data[fid] = fmeta["value"]
        
        # 1. Fill logical doc
        filled_bytes = fill_pdf_with_mapping(st.session_state.target_pdf_path, current_data, st.session_state.field_mapping)
        
        # 2. Convert to Image for FLICKER-FREE UI (Step 7)
        img_bytes = render_pdf_to_image(filled_bytes)
        img_slot.image(img_bytes, use_column_width=True)
        
        progress_bar.progress((i + 1) / total)
        time.sleep(0.4) 
    
    # Finalize
    st.session_state.pdf_bytes = filled_bytes
    st.session_state.live_fill_complete = True
    st.session_state.is_live_filling = False
    save_session()
    st.rerun()

# Step 8 & 9: Verification & Export Mode
elif st.session_state.get("live_fill_complete"):
    st.success("✅ Form Successfully Filled! Please perform your final review below.")
    
    # Dual Feed (Verification Table View)
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.subheader("📄 Original Blank")
        render_pdf_iframe(st.session_state.get("target_pdf_path"), height=800)
    with v_col2:
        st.subheader("✨ Filled Final")
        render_pdf_iframe(st.session_state.get("pdf_bytes"), height=800)
        
    st.divider()
    
    # Export Toolbar
    e1, e2, e3 = st.columns([1, 1, 1])
    with e1:
        json_export = json.dumps({k: v["value"] for k, v in st.session_state.form_state.items()}, indent=4)
        st.download_button("📂 Export JSON Data", data=json_export, file_name="form_data.json", use_container_width=True)
    with e2:
        if st.session_state.get("pdf_bytes"):
            base_name = os.path.splitext(st.session_state.get("target_filename", "form"))[0]
            st.download_button("📄 Download Final PDF", data=st.session_state.get("pdf_bytes"), file_name=f"{base_name}_filled.pdf", type="primary", use_container_width=True)
    with e3:
        if st.button("🔄 Start New Form", use_container_width=True):
            st.query_params.clear()
            st.session_state.clear()
            st.rerun()
else:
    st.info("👈 Please upload your Target Form and Context (optional) in the sidebar and click Extract to begin.")
