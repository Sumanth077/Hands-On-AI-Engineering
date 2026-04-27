import streamlit as st
import os
import re
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 500       # words per chunk
CHUNK_OVERLAP = 50     # word overlap between chunks
TOP_K = 5              # chunks retrieved per question
WHISPER_MODEL_SIZE = "base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral-small-latest"


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    patterns = [
        r"[?&]v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def collection_name_for(video_id: str) -> str:
    # ChromaDB: 3-63 chars, [a-zA-Z0-9_-], no leading/trailing - or _
    # Wrapping with alpha chars guarantees valid boundaries
    name = f"yt{video_id}yt"
    return name[:63]


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def yt_link(video_id: str, seconds: float) -> str:
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    import whisper
    return whisper.load_model(WHISPER_MODEL_SIZE)


@st.cache_resource(show_spinner=False)
def get_embedding_fn():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


@st.cache_resource(show_spinner=False)
def get_mistral_llm():
    from langchain_mistralai import ChatMistralAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None
    return ChatMistralAI(model=MISTRAL_MODEL, api_key=api_key)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def collection_exists(client, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def download_audio(url: str, out_dir: str) -> str:
    import yt_dlp
    out_path = os.path.join(out_dir, "audio")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out_path + ".mp3"


def transcribe_audio(audio_path: str) -> list[dict]:
    model = load_whisper_model()
    result = model.transcribe(audio_path, verbose=False)
    return result["segments"]


def chunk_segments(segments: list[dict]) -> list[dict]:
    """Flatten whisper segments into overlapping ~500-word chunks with timestamps."""
    word_entries = []
    for seg in segments:
        for word in seg["text"].strip().split():
            word_entries.append({"word": word, "start": seg["start"]})

    chunks = []
    i = 0
    while i < len(word_entries):
        batch = word_entries[i : i + CHUNK_SIZE]
        chunks.append({
            "text": " ".join(e["word"] for e in batch),
            "start": batch[0]["start"],
            "index": len(chunks),
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest_video(url: str, video_id: str, status_widget) -> object:
    """Full pipeline: download → transcribe → chunk → embed → store."""
    chroma = get_chroma_client()
    emb_fn = get_embedding_fn()
    cname = collection_name_for(video_id)

    with tempfile.TemporaryDirectory() as tmp:
        status_widget.write("⬇️ Downloading audio…")
        audio_path = download_audio(url, tmp)

        status_widget.write("🎙️ Transcribing with Whisper (this can take a few minutes for long videos)…")
        segments = transcribe_audio(audio_path)

    status_widget.write(f"✂️ Splitting into chunks…")
    chunks = chunk_segments(segments)

    status_widget.write(f"🗄️ Generating embeddings and storing in ChromaDB…")
    col = chroma.get_or_create_collection(cname, embedding_function=emb_fn)
    col.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"start": c["start"], "video_id": video_id} for c in chunks],
        ids=[f"{cname}_{c['index']}" for c in chunks],
    )
    return col


def load_collection(video_id: str) -> object:
    chroma = get_chroma_client()
    emb_fn = get_embedding_fn()
    return chroma.get_collection(collection_name_for(video_id), embedding_function=emb_fn)


def retrieve_chunks(collection, question: str) -> list[dict]:
    res = collection.query(query_texts=[question], n_results=TOP_K)
    return [
        {"text": doc, "start": meta["start"], "video_id": meta["video_id"]}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


def build_context_block(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        ts = fmt_ts(h["start"])
        link = yt_link(h["video_id"], h["start"])
        parts.append(f"[Timestamp {ts}]({link})\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, hits: list[dict], history: list[dict]) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm = get_mistral_llm()
    if llm is None:
        return "⚠️ `MISTRAL_API_KEY` is not set. Please add it to your `.env` file."

    context = build_context_block(hits)
    system = (
        "You are a knowledgeable assistant that answers questions strictly based on provided YouTube video transcripts.\n"
        "You are given the most relevant transcript chunks, each preceded by a clickable timestamp.\n"
        "When your answer references specific content, include the timestamp as a markdown link, e.g. "
        "[1:23](https://youtu.be/VIDEO_ID?t=83).\n"
        "If the transcript chunks do not contain enough information to answer, say so honestly.\n"
        "Do not fabricate information.\n\n"
        f"Relevant transcript segments:\n\n{context}"
    )

    messages = [SystemMessage(content=system)]
    for msg in history[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    resp = llm.invoke(messages)
    return resp.content


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube Transcript RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state init
for key, default in {
    "messages": [],
    "history": [],
    "video_id": None,
    "collection": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 YouTube RAG")
    st.caption("Ask questions about any YouTube video.")
    st.divider()

    url_input = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    process_btn = st.button(
        "▶  Process Video",
        use_container_width=True,
        type="primary",
        disabled=not url_input.strip(),
    )

    if st.session_state.video_id:
        vid_id = st.session_state.video_id
        st.success(f"✓ Loaded: `{vid_id}`")
        st.markdown(f"[Open on YouTube ↗](https://youtu.be/{vid_id})")
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.video_id = None
            st.session_state.collection = None
            st.rerun()

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. Paste a YouTube URL and click **Process Video**\n"
            "2. Audio is downloaded with **yt-dlp**\n"
            "3. Transcribed locally with **OpenAI Whisper**\n"
            "4. Chunks stored as embeddings in **ChromaDB**\n"
            "5. Your question retrieves the top 5 relevant chunks\n"
            "6. **Mistral Small 4** answers with timestamp references\n\n"
            "Re-pasting the same URL skips transcription — loaded from cache instantly."
        )

    if not os.getenv("MISTRAL_API_KEY"):
        st.warning("⚠️ `MISTRAL_API_KEY` not found in `.env`")

# ── Video processing ──────────────────────────────────────────────────────────
if process_btn and url_input.strip():
    vid = extract_video_id(url_input.strip())
    if not vid:
        st.sidebar.error("Could not find a video ID in that URL.")
    elif vid == st.session_state.video_id:
        st.sidebar.info("This video is already loaded and ready.")
    else:
        chroma = get_chroma_client()
        cname = collection_name_for(vid)

        if collection_exists(chroma, cname):
            with st.spinner("Loading existing transcript from ChromaDB…"):
                st.session_state.collection = load_collection(vid)
            st.session_state.video_id = vid
            st.session_state.messages = []
            st.session_state.history = []
            st.toast("Loaded from cache!", icon="✅")
            st.rerun()
        else:
            with st.status("Processing video…", expanded=True) as status:
                try:
                    col = ingest_video(url_input.strip(), vid, status)
                    st.session_state.collection = col
                    st.session_state.video_id = vid
                    st.session_state.messages = []
                    st.session_state.history = []
                    status.update(label="✅ Video ready! Start asking questions.", state="complete")
                except Exception as exc:
                    status.update(label=f"Failed: {exc}", state="error")
                    st.error(
                        f"**Processing error:** {exc}\n\n"
                        "Make sure `ffmpeg` is installed and the video is publicly accessible."
                    )

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("YouTube Transcript RAG")
st.markdown(
    "Semantic search over video transcripts — powered by **Whisper**, **ChromaDB** & **Mistral Small 4**."
)

if st.session_state.collection is None:
    st.info(
        "👈 Paste a YouTube URL in the sidebar and click **Process Video** to get started.\n\n"
        "Transcription happens locally — no OpenAI API key needed."
    )
else:
    vid_id = st.session_state.video_id
    st.markdown(
        f"**Chatting about:** [youtube.com/watch?v={vid_id}](https://www.youtube.com/watch?v={vid_id})"
    )
    st.divider()

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Accept new question
    if question := st.chat_input("Ask something about the video…"):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve & generate
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript and generating answer…"):
                hits = retrieve_chunks(st.session_state.collection, question)
                answer = generate_answer(question, hits, st.session_state.history)
            st.markdown(answer)

            # Show source chunks in expander
            with st.expander("📄 Source transcript chunks", expanded=False):
                for i, h in enumerate(hits, 1):
                    ts = fmt_ts(h["start"])
                    link = yt_link(h["video_id"], h["start"])
                    st.markdown(f"**Chunk {i}** — [{ts}]({link})")
                    st.caption(h["text"])
                    if i < len(hits):
                        st.divider()

        # Persist to session state
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})
