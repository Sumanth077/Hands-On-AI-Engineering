"""
LaTeX Formula OCR
==================
Streamlit app that extracts mathematical formulas from images using GLM-OCR
via Ollama and renders them visually with KaTeX.

SETUP INSTRUCTIONS:
  1. Install Ollama: https://ollama.ai/download
  2. Pull the OCR model:
         ollama pull glm-ocr
  3. Ensure Ollama is running (it auto-starts on most systems after install,
     or start it manually):
         ollama serve
  4. Install Python dependencies:
         pip install -r requirements.txt
  5. Launch the app:
         streamlit run app.py

NOTES:
  - Ollama must be reachable at http://localhost:11434
  - PDF support requires PyMuPDF (included in requirements.txt)
  - KaTeX is loaded from jsDelivr CDN for in-browser formula rendering
  - No API keys or external OCR services are needed — fully local
"""

import streamlit as st
import requests
import base64
import re
import json
from html import escape as html_escape
from io import BytesIO

from PIL import Image

# ── Optional PDF support via PyMuPDF ─────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL     = "http://localhost:11434/api/tags"
MODEL_NAME          = "glm-ocr"
MAX_IMG_DIM         = 1920          # px — downscale larger images before sending
KATEX_CDN           = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

# ── Prompt sent to GLM-OCR ────────────────────────────────────────────────────
EXTRACTION_PROMPT = """\
You are a mathematical formula OCR engine.
Carefully examine the image and extract EVERY mathematical expression or equation visible.
Output ONLY the LaTeX source for each formula, each one wrapped in $$ ... $$ delimiters.
Place each formula on its own line.
Do not include any explanation, prose, or extra text — only the $$ ... $$ blocks.\
"""


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Image helpers
# ╚══════════════════════════════════════════════════════════════════════════════

def resize_if_needed(image_bytes: bytes) -> bytes:
    """Downscale an image so its longest side is at most MAX_IMG_DIM pixels."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMG_DIM:
        ratio = MAX_IMG_DIM / max(w, h)
        img   = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf   = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return image_bytes


def to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def pdf_first_page_to_png(pdf_bytes: bytes) -> bytes:
    """Render the first page of a PDF to a high-resolution PNG byte string."""
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB)
    return pix.tobytes("png")


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Ollama query
# ╚══════════════════════════════════════════════════════════════════════════════

def query_glm_ocr(image_b64: str) -> str:
    """
    POST the image (base64-encoded) to Ollama's native /api/generate endpoint
    using the glm-ocr model.  Returns the raw response string.
    """
    payload = {
        "model":  MODEL_NAME,
        "prompt": EXTRACTION_PROMPT,
        "images": [image_b64],
        "stream": False,
    }
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")


def check_ollama_status() -> tuple[bool, bool]:
    """
    Returns (ollama_reachable, glm_ocr_installed).
    Quick, non-blocking probe used in the sidebar.
    """
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if not r.ok:
            return False, False
        models = [m.get("name", "") for m in r.json().get("models", [])]
        has_model = any("glm-ocr" in m for m in models)
        return True, has_model
    except Exception:
        return False, False


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  LaTeX extraction
# ╚══════════════════════════════════════════════════════════════════════════════

def extract_formulas(text: str) -> list:
    """
    Pull LaTeX formulas out of the model's response.
    Tries delimiter styles in order of preference; falls back to raw lines.
    Returns a list of formula strings (without surrounding delimiters).
    """
    formulas = []

    # 1. $$...$$  — primary format we asked for
    formulas = [m.strip() for m in re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL) if m.strip()]
    if formulas:
        return formulas

    # 2. \[...\]  — display math, alternative LaTeX style
    formulas = [m.strip() for m in re.findall(r'\\\[(.*?)\\\]', text, re.DOTALL) if m.strip()]
    if formulas:
        return formulas

    # 3. Named environments: equation, align, gather, multline, eqnarray
    env_pattern = (
        r'(\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}'
        r'.*?'
        r'\\end\{(?:equation|align|gather|multline|eqnarray)\*?\})'
    )
    formulas = [m.strip() for m in re.findall(env_pattern, text, re.DOTALL) if m.strip()]
    if formulas:
        return formulas

    # 4. $...$  — inline math (avoid matching $$)
    formulas = [
        m.strip()
        for m in re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', text, re.DOTALL)
        if m.strip()
    ]
    if formulas:
        return formulas

    # 5. \(...\)  — inline math, alternative style
    formulas = [m.strip() for m in re.findall(r'\\\((.*?)\\\)', text, re.DOTALL) if m.strip()]
    if formulas:
        return formulas

    # 6. Fallback — treat every non-empty line as a possible formula
    return [line.strip() for line in text.splitlines() if line.strip()]


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  KaTeX HTML builder
# ╚══════════════════════════════════════════════════════════════════════════════

def build_katex_html(formulas: list) -> str:
    """
    Build a self-contained HTML page that:
      • Renders every formula with KaTeX (display mode)
      • Shows the raw LaTeX source in a styled code block
      • Provides a per-formula copy button backed by the Clipboard API
    All formulas are rendered in a single iframe to share one CDN load.
    """
    formulas_js = json.dumps(formulas)   # safe JSON array injected into JS

    # Per-formula card HTML (KaTeX rendering done in JS; only display code here)
    cards_html_parts = []
    for i, formula in enumerate(formulas):
        safe_formula = html_escape(formula)      # HTML-safe for <pre> display
        cards_html_parts.append(f"""
    <div class="card">
      <div class="card-header">
        <span class="formula-num">Formula {i + 1}</span>
      </div>
      <div class="card-body">
        <div class="render-col">
          <div class="col-label">Rendered</div>
          <div class="katex-target" id="f{i}"></div>
        </div>
        <div class="code-col">
          <div class="col-label">LaTeX source</div>
          <pre class="latex-pre"><code>{safe_formula}</code></pre>
          <button class="copy-btn" id="btn{i}" onclick="copyFormula({i})">
            &#x2398;&nbsp;Copy LaTeX
          </button>
        </div>
      </div>
    </div>""")

    cards_html = "\n".join(cards_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet"
      href="{KATEX_CDN}/katex.min.css"
      crossorigin="anonymous">
<script defer
        src="{KATEX_CDN}/katex.min.js"
        crossorigin="anonymous"
        onload="renderFormulas()"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: system-ui, -apple-system, sans-serif;
    padding: 12px;
    background: transparent;
    color: #1a1a2e;
  }}

  .card {{
    border: 1px solid #d0d7de;
    border-radius: 8px;
    margin-bottom: 20px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}

  .card-header {{
    background: #f6f8fa;
    padding: 7px 14px;
    border-bottom: 1px solid #d0d7de;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .formula-num {{
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #57606a;
  }}

  .card-body {{
    display: flex;
  }}

  .render-col {{
    flex: 1;
    padding: 20px;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #d0d7de;
    min-width: 0;
    background: #fff;
  }}

  .code-col {{
    flex: 1;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
    background: #0d1117;
  }}

  .col-label {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 10px;
  }}

  .render-col .col-label {{ color: #888; }}
  .code-col   .col-label {{ color: #6e7681; }}

  .katex-target {{
    overflow-x: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 48px;
    padding: 4px 0;
  }}

  .katex-display {{ margin: 0 !important; }}

  .katex-error {{
    color: #e03131;
    font-family: monospace;
    font-size: 12px;
    padding: 6px;
  }}

  .latex-pre {{
    background: #161b22;
    color: #e6edf3;
    padding: 12px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-family: "Cascadia Code", "Fira Code", "JetBrains Mono", monospace;
    white-space: pre-wrap;
    word-break: break-word;
    flex: 1;
    overflow: auto;
    border: 1px solid #30363d;
  }}

  .copy-btn {{
    align-self: flex-start;
    background: #238636;
    color: #fff;
    border: 1px solid #2ea043;
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background .15s, border-color .15s;
    white-space: nowrap;
  }}

  .copy-btn:hover {{ background: #2ea043; border-color: #3fb950; }}
  .copy-btn.ok    {{ background: #1a7f37; border-color: #2ea043; }}
</style>
</head>
<body>
{cards_html}

<script>
const FORMULAS = {formulas_js};

/* Render every formula with KaTeX once the library has loaded. */
function renderFormulas() {{
  FORMULAS.forEach(function(latex, i) {{
    const el = document.getElementById('f' + i);
    if (!el) return;
    try {{
      katex.render(latex, el, {{
        displayMode:  true,
        throwOnError: false,
        errorColor:   '#e03131'
      }});
    }} catch (err) {{
      el.innerHTML =
        '<span class="katex-error">&#9888; Render error: ' +
        err.message.replace(/</g, '&lt;') + '</span>';
    }}
  }});
}}

/* Copy a formula's LaTeX source to the clipboard. */
function copyFormula(i) {{
  const latex = FORMULAS[i];
  const btn   = document.getElementById('btn' + i);

  function markCopied() {{
    btn.textContent = '\\u2713 Copied!';
    btn.classList.add('ok');
    setTimeout(function() {{
      btn.innerHTML = '&#x2398;&nbsp;Copy LaTeX';
      btn.classList.remove('ok');
    }}, 2000);
  }}

  if (navigator.clipboard && window.isSecureContext) {{
    navigator.clipboard.writeText(latex).then(markCopied).catch(fallbackCopy);
  }} else {{
    fallbackCopy();
  }}

  function fallbackCopy() {{
    const ta = document.createElement('textarea');
    ta.value = latex;
    Object.assign(ta.style, {{ position: 'fixed', top: 0, left: 0, opacity: '0' }});
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {{ document.execCommand('copy'); }} catch (_) {{}}
    document.body.removeChild(ta);
    markCopied();
  }}
}}
</script>
</body>
</html>"""


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Streamlit UI
# ╚══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="LaTeX Formula OCR",
        page_icon="∑",
        layout="wide",
    )

    # ── Sidebar — live Ollama status ──────────────────────────────────────────
    with st.sidebar:
        st.header("Connection")
        ollama_ok, model_ok = check_ollama_status()

        if ollama_ok and model_ok:
            st.success("Ollama running  ✓\nglm-ocr ready  ✓")
        elif ollama_ok and not model_ok:
            st.warning(
                "Ollama is running but **glm-ocr** is not installed.  \n"
                "Run:  `ollama pull glm-ocr`"
            )
        else:
            st.error(
                "Ollama not reachable at `localhost:11434`.  \n"
                "Start it with:  `ollama serve`"
            )

        st.divider()
        st.markdown("**Quick start**")
        st.code(
            "ollama pull glm-ocr\n"
            "ollama serve\n"
            "streamlit run app.py",
            language="bash",
        )
        st.divider()
        st.markdown(
            "**Model:** `glm-ocr` (Ollama)  \n"
            "**Renderer:** KaTeX 0.16.11  \n"
            "**PDF support:** " + ("enabled ✓" if PDF_SUPPORT else "disabled ✗")
        )

    # ── Page header ───────────────────────────────────────────────────────────
    st.title("∑  LaTeX Formula OCR")
    st.caption(
        "Upload an image or PDF page — GLM-OCR (via Ollama) extracts the math "
        "formulas and KaTeX renders them right in the browser."
    )

    # ── File uploader ─────────────────────────────────────────────────────────
    accepted_types = ["png", "jpg", "jpeg"]
    if PDF_SUPPORT:
        accepted_types.append("pdf")

    uploaded = st.file_uploader(
        label="Choose an image" + (" or single-page PDF" if PDF_SUPPORT else ""),
        type=accepted_types,
        help=(
            "Supported: PNG, JPG/JPEG"
            + (", PDF (first page only)" if PDF_SUPPORT else "")
            + ". Make sure mathematical formulas are clearly visible."
        ),
    )

    if not PDF_SUPPORT:
        st.caption(
            "PDF support is disabled. Install PyMuPDF to enable it:  "
            "`pip install pymupdf`"
        )

    if uploaded is None:
        st.info(
            "Upload an image or PDF page that contains mathematical formulas. "
            "Then click **Extract Formulas** to run GLM-OCR."
        )
        return

    # ── Decode file bytes ─────────────────────────────────────────────────────
    raw_bytes = uploaded.read()

    if uploaded.name.lower().endswith(".pdf"):
        with st.spinner("Converting PDF page to image…"):
            try:
                image_bytes = pdf_first_page_to_png(raw_bytes)
            except Exception as exc:
                st.error(f"PDF conversion failed: {exc}")
                return
    else:
        image_bytes = raw_bytes

    # ── Image preview ─────────────────────────────────────────────────────────
    with st.expander("Preview", expanded=True):
        st.image(image_bytes, use_column_width=True)

    # ── Extract button ────────────────────────────────────────────────────────
    if not st.button("Extract Formulas", type="primary", use_container_width=True):
        return

    # ── OCR ───────────────────────────────────────────────────────────────────
    with st.spinner("GLM-OCR is analysing the image — this may take 10–30 s on first run…"):
        try:
            processed  = resize_if_needed(image_bytes)
            b64        = to_base64(processed)
            raw_text   = query_glm_ocr(b64)
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach Ollama at `http://localhost:11434`.  \n"
                "Make sure Ollama is running (`ollama serve`) and "
                "glm-ocr is installed (`ollama pull glm-ocr`)."
            )
            return
        except requests.exceptions.Timeout:
            st.error(
                "The request timed out after 300 s.  \n"
                "The model may still be loading. Wait a moment and try again."
            )
            return
        except requests.exceptions.HTTPError as exc:
            st.error(f"Ollama returned an HTTP error: {exc}")
            return
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            return

    # ── Raw response (collapsed) ──────────────────────────────────────────────
    with st.expander("Raw model response", expanded=False):
        st.text(raw_text.strip() if raw_text.strip() else "(empty)")

    if not raw_text.strip():
        st.warning(
            "GLM-OCR returned an empty response.  \n"
            "Try a clearer image where formulas are large and high-contrast."
        )
        return

    # ── Parse formulas ────────────────────────────────────────────────────────
    formulas = extract_formulas(raw_text)

    if not formulas:
        st.warning(
            "No mathematical formulas were detected in this image.  \n"
            "Tips: use a cropped, high-resolution image where formulas are clearly visible."
        )
        return

    # ── Results header ────────────────────────────────────────────────────────
    count = len(formulas)
    st.success(f"Extracted **{count}** formula{'s' if count != 1 else ''}.")
    st.divider()

    # ── KaTeX rendering ───────────────────────────────────────────────────────
    # Each card is ~220 px tall; add top/bottom padding.
    iframe_height = max(300, 30 + count * 220)

    html_content = build_katex_html(formulas)
    st.components.v1.html(html_content, height=iframe_height, scrolling=True)


if __name__ == "__main__":
    main()
