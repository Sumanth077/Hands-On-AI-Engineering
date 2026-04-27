"""
Music Explorer — Gemini-powered audio analysis
Ask anything about any audio file or YouTube video.
Run: python app.py
"""

import os
import re
import tempfile
import shutil

import gradio as gr
import yt_dlp
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── YouTube helpers ────────────────────────────────────────────────────────────

_yt_cache: dict[str, tuple[str, str]] = {}  # url → (audio_path, title)


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&=%?]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([^&=%?]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([^&=%?]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _youtube_embed(url: str) -> str:
    vid = _extract_video_id(url)
    if not vid:
        return ""
    return (
        f'<div style="position:relative;width:100%;padding-bottom:56.25%;'
        f'border-radius:12px;overflow:hidden;">'
        f'<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" '
        f'src="https://www.youtube.com/embed/{vid}" frameborder="0" allowfullscreen></iframe>'
        f"</div>"
    )


def download_youtube(url: str, force: bool = False) -> tuple[str | None, str]:
    url = url.strip()
    if not _extract_video_id(url):
        return None, "❌ Invalid YouTube URL."

    if not force and url in _yt_cache:
        path, title = _yt_cache[url]
        if os.path.exists(path):
            return path, f"✅ Loaded: {title[:60]}"

    if force and url in _yt_cache:
        old_path, _ = _yt_cache.pop(url)
        try:
            shutil.rmtree(os.path.dirname(old_path), ignore_errors=True)
        except Exception:
            pass

    tmp = tempfile.mkdtemp()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp, "%(title)s.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
        ],
        "noplaylist": True,
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Unknown")
            ydl.download([url])
        for f in os.listdir(tmp):
            if f.endswith(".mp3"):
                path = os.path.join(tmp, f)
                _yt_cache[url] = (path, title)
                return path, f"✅ Downloaded: {title[:60]}"
        return None, "❌ Audio extraction failed."
    except Exception as e:
        return None, f"❌ {e}"


# ── Gemini helpers ─────────────────────────────────────────────────────────────

def _upload_to_gemini(audio_path: str):
    """Upload a local audio file to the Gemini Files API and return the file object."""
    return client.files.upload(file=audio_path)


def _extract_text(content) -> str:
    """Gradio may store message content as a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _build_contents(gemini_file, history: list[dict], new_prompt: str) -> list:
    """
    Build the contents list for a multi-turn conversation.
    history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    The audio file is attached to the first user turn only.
    """
    contents = []
    has_prior_turns = any(m["role"] == "user" for m in history)

    for msg in history:
        role = msg["role"]
        text = _extract_text(msg["content"])
        if role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=text)])
            )
        else:
            contents.append(
                types.Content(role="model", parts=[types.Part.from_text(text=text)])
            )

    # New user turn — attach audio on the very first message in the conversation
    if not has_prior_turns:
        audio_part = types.Part.from_uri(
            file_uri=gemini_file.uri,
            mime_type=gemini_file.mime_type,
        )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=new_prompt), audio_part],
            )
        )
    else:
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=new_prompt)])
        )
    return contents


# ── Core inference ─────────────────────────────────────────────────────────────

def analyze(
    audio_file,
    youtube_url: str,
    prompt: str,
    history: list[dict],
    gemini_file_name: str,
):
    """
    Main inference function. Uploads audio on first turn, reuses on subsequent turns.
    Returns updated chat history and the Gemini file name for reuse.
    """
    if not prompt.strip():
        yield history, gemini_file_name
        return

    # Resolve audio source
    audio_path = None
    if audio_file:
        audio_path = audio_file
    elif youtube_url.strip():
        audio_path, msg = download_youtube(youtube_url.strip())
        if not audio_path:
            history = history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": msg},
            ]
            yield history, gemini_file_name
            return

    if not audio_path and not gemini_file_name:
        history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "❌ Please upload an audio file or provide a YouTube URL first."},
        ]
        yield history, gemini_file_name
        return

    # Upload to Gemini Files API on the first message or if audio changed
    gemini_file = None
    if audio_path and not gemini_file_name:
        try:
            gemini_file = _upload_to_gemini(audio_path)
            gemini_file_name = gemini_file.name
        except Exception as e:
            history = history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"❌ Failed to upload audio: {e}"},
            ]
            yield history, gemini_file_name
            return
    elif gemini_file_name:
        try:
            gemini_file = client.files.get(name=gemini_file_name)
        except Exception:
            # File expired — re-upload
            if audio_path:
                gemini_file = _upload_to_gemini(audio_path)
                gemini_file_name = gemini_file.name

    if not gemini_file:
        history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "❌ No audio available. Please reload your audio source."},
        ]
        yield history, gemini_file_name
        return

    # Build conversation contents
    contents = _build_contents(gemini_file, history, prompt)

    # Stream response
    history = history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]
    try:
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL_ID, contents=contents
        ):
            if chunk.text:
                response_text += chunk.text
                history[-1]["content"] = response_text
                yield history, gemini_file_name
    except Exception as e:
        history[-1]["content"] = f"❌ Error: {e}"
        yield history, gemini_file_name


# ── Gradio UI ──────────────────────────────────────────────────────────────────

QUICK_PROMPTS = [
    "Describe this audio in full detail — genre, tempo, key, instruments, mood, and production style.",
    "Transcribe the lyrics or spoken words with timestamps.",
    "Detect and describe the emotional tone. Does it shift throughout?",
    "List every instrument you can identify and when each enters.",
    "What is the structure of this track? Label each section (intro, verse, chorus, etc.) with timestamps.",
]

CSS = """
:root { --primary: #0d9488; --primary-light: #14b8a6; }
body, .gradio-container { font-family: ui-sans-serif, system-ui, sans-serif; }
.gradio-container { max-width: 1200px !important; margin-inline: auto; padding-bottom: 60px; }
.header { text-align: center; padding: 32px 16px 8px; }
.header h1 { font-size: clamp(2rem, 4vw, 2.8rem); font-weight: 700;
    background: linear-gradient(120deg, #0d9488, #7c3aed); -webkit-background-clip: text;
    background-clip: text; color: transparent; margin: 0 0 8px; }
.header p { font-size: 1.05rem; opacity: 0.75; margin: 0; }
.card { border: 1px solid var(--border-color-primary); border-radius: 20px;
    padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.quick-btn button { border-radius: 20px !important; font-size: 0.82rem !important;
    padding: 6px 14px !important; }
.send-btn { background: linear-gradient(120deg, #0d9488, #7c3aed) !important;
    border-radius: 12px !important; color: #fff !important;
    font-weight: 600 !important; }
.footer { text-align: center; opacity: 0.55; font-size: 0.85rem; margin-top: 24px; }
"""

with gr.Blocks(title="Music Explorer") as demo:

    gemini_file_state = gr.State("")

    # ── Header ──
    gr.HTML("""
    <div class="header">
        <h1>🎧 Music Explorer</h1>
        <p>Upload any audio file or paste a YouTube URL — then ask anything about it.</p>
        <p style="font-size:0.9rem;opacity:0.6;margin-top:6px;">
            Powered by <strong>Gemini 3 Flash</strong> · Transcription · Lyrics · Genre · Emotion · Timestamps
        </p>
    </div>
    """)

    with gr.Row(equal_height=False, elem_classes="card"):

        # ── Left column: audio input ──
        with gr.Column(scale=2):
            gr.Markdown("### 🎵 Audio Source")
            gr.Markdown("Upload a file **or** paste a YouTube URL below.")

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload Audio File",
            )

            gr.Markdown("**Or load from YouTube:**")
            yt_url = gr.Textbox(
                placeholder="https://www.youtube.com/watch?v=...",
                label="YouTube URL",
                show_label=False,
            )
            with gr.Row():
                yt_load_btn = gr.Button("Load YouTube Audio", size="sm", variant="secondary")
                yt_clear_btn = gr.Button("Clear", size="sm")

            yt_status = gr.Textbox(label="Status", interactive=False, visible=False)
            yt_embed = gr.HTML(visible=False)

            gr.Markdown("---")
            gr.Markdown("### ⚡ Quick Analysis")
            gr.Markdown("Click any button to pre-fill the prompt:")

            quick_btns = []
            labels = ["Full Description", "Transcription", "Emotion Analysis", "Instrument List", "Track Structure"]
            for label, prompt_text in zip(labels, QUICK_PROMPTS):
                btn = gr.Button(label, size="sm", elem_classes="quick-btn")
                quick_btns.append((btn, prompt_text))

        # ── Right column: chat ──
        with gr.Column(scale=3):
            gr.Markdown("### 💬 Ask Anything")
            chatbot = gr.Chatbot(
                label="",
                height=480,
            )
            with gr.Row():
                prompt_input = gr.Textbox(
                    placeholder="e.g. What genre is this? Transcribe the lyrics. When does the chorus start?",
                    label="",
                    scale=5,
                    lines=2,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary", elem_classes="send-btn")

            clear_btn = gr.Button("🗑 Clear conversation", size="sm", variant="secondary")

    gr.HTML('<div class="footer">Music Explorer &nbsp;|&nbsp; Powered by Gemini 3 Flash &nbsp;|&nbsp; Audio via Gemini Files API &nbsp;|&nbsp; YouTube via yt-dlp</div>')

    # ── Event wiring ──────────────────────────────────────────────────────────

    def load_yt(url):
        if not url.strip():
            return None, gr.update(value="❌ Enter a URL first.", visible=True), gr.update(visible=False)
        path, msg = download_youtube(url.strip(), force=True)
        embed = _youtube_embed(url.strip())
        return (
            path if path else None,
            gr.update(value=msg, visible=True),
            gr.update(value=embed, visible=bool(embed)),
        )

    yt_load_btn.click(
        fn=load_yt,
        inputs=[yt_url],
        outputs=[audio_input, yt_status, yt_embed],
    )

    def clear_yt():
        return "", None, gr.update(value="", visible=False), gr.update(visible=False)

    yt_clear_btn.click(fn=clear_yt, outputs=[yt_url, audio_input, yt_status, yt_embed])

    # Quick prompt buttons
    for btn, prompt_text in quick_btns:
        btn.click(fn=lambda t=prompt_text: t, outputs=prompt_input)

    # Send message
    send_btn.click(
        fn=analyze,
        inputs=[audio_input, yt_url, prompt_input, chatbot, gemini_file_state],
        outputs=[chatbot, gemini_file_state],
    ).then(fn=lambda: "", outputs=prompt_input)

    prompt_input.submit(
        fn=analyze,
        inputs=[audio_input, yt_url, prompt_input, chatbot, gemini_file_state],
        outputs=[chatbot, gemini_file_state],
    ).then(fn=lambda: "", outputs=prompt_input)

    # Changing audio source resets the Gemini file reference
    audio_input.change(fn=lambda: "", outputs=gemini_file_state)
    yt_url.change(fn=lambda: "", outputs=gemini_file_state)

    # Clear conversation
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, gemini_file_state])




if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="teal"))
