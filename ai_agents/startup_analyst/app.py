"""
<<<<<<< HEAD
Startup Analyst — Gradio UI
=======
Startup Analyst - Gradio UI
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
Elite startup due-diligence powered by MiniMax M2.5 via OpenRouter.

Usage:
    uv run python app.py
"""

from dotenv import load_dotenv

load_dotenv()

<<<<<<< HEAD
=======
import os
import re
import tempfile

>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
import gradio as gr
from agent import analyst

# ── Styling ────────────────────────────────────────────────────────────────────

CSS = """
#title     { text-align: center; }
#subtitle  { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }
#report    { min-height: 560px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1.2rem; }
"""

EXAMPLES = [
    ["Perform a comprehensive startup intelligence analysis on xAI (https://x.ai)"],
    ["Perform a comprehensive startup intelligence analysis on Perplexity AI (https://perplexity.ai)"],
    ["Perform a comprehensive startup intelligence analysis on Cursor (https://cursor.com)"],
    ["Perform a comprehensive startup intelligence analysis on ElevenLabs (https://elevenlabs.io)"],
    ["Perform a comprehensive startup intelligence analysis on Mistral AI (https://mistral.ai)"],
]

PLACEHOLDER = "_Your due-diligence report will appear here once you submit a company._"


<<<<<<< HEAD
# ── Analysis Function ──────────────────────────────────────────────────────────

def run_analysis(prompt: str):
    prompt = prompt.strip()
    if not prompt:
        yield PLACEHOLDER, ""
        return

    yield "_Analysing company — scraping website, crawling pages, gathering data..._", "Running..."
=======
# ── Helpers ────────────────────────────────────────────────────────────────────

def _report_filename(prompt: str) -> str:
    """Derive a sanitized .md filename from the company name in the prompt."""
    match = re.search(r'\bon\s+(.+?)(?:\s*\(https?://)', prompt, re.IGNORECASE)
    if match:
        name = re.sub(r'[^\w\s]', '', match.group(1)).strip()
        name = re.sub(r'\s+', '_', name).lower()
        return f"{name}_analysis.md"
    return "startup_analysis_report.md"


def _write_report_file(prompt: str, report: str) -> str:
    """Write the report markdown to a temp file and return its path."""
    path = os.path.join(tempfile.gettempdir(), _report_filename(prompt))
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


# ── Analysis Function ──────────────────────────────────────────────────────────

def run_analysis(prompt: str):
    """Stream a startup intelligence report for the given prompt, yielding (report_markdown, status, download_update) tuples."""
    prompt = prompt.strip()
    if not prompt:
        yield PLACEHOLDER, "", gr.update(visible=False)
        return

    yield "_Analysing company: scraping website, crawling pages, gathering data..._", "Running...", gr.update(visible=False)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

    full_output = ""
    try:
        for chunk in analyst.run(prompt, stream=True):
            if chunk.content:
                full_output += chunk.content
                # Only display once the structured report begins (first # heading)
                heading_idx = full_output.find("#")
                if heading_idx != -1:
<<<<<<< HEAD
                    yield full_output[heading_idx:], "Running..."
    except Exception as exc:
        heading_idx = full_output.find("#")
        report = full_output[heading_idx:] if heading_idx != -1 else full_output
        yield (report or PLACEHOLDER) + f"\n\n---\n\n**Error:** {exc}", f"Error: {exc}"
=======
                    yield full_output[heading_idx:], "Running...", gr.update(visible=False)
    except Exception as exc:
        heading_idx = full_output.find("#")
        report = full_output[heading_idx:] if heading_idx != -1 else full_output
        yield (report or PLACEHOLDER) + f"\n\n---\n\n**Error:** {exc}", f"Error: {exc}", gr.update(visible=False)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
        return

    if full_output:
        heading_idx = full_output.find("#")
<<<<<<< HEAD
        yield full_output[heading_idx:] if heading_idx != -1 else full_output, "Done."
    else:
        yield PLACEHOLDER, "No response received."


def clear_all():
    return "", PLACEHOLDER, ""
=======
        report = full_output[heading_idx:] if heading_idx != -1 else full_output
        yield report, "Done.", gr.update(value=_write_report_file(prompt, report), visible=True)
    else:
        yield PLACEHOLDER, "No response received.", gr.update(visible=False)


def clear_all():
    """Reset the prompt input, report output, status box, and download button to their default empty state."""
    return "", PLACEHOLDER, "", gr.update(value=None, visible=False)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Startup Analyst") as demo:

    gr.Markdown("# 📊 Startup Analyst", elem_id="title")
    gr.Markdown(
        "Elite startup due-diligence powered by **MiniMax M2.5** via OpenRouter. "
<<<<<<< HEAD
        "Enter a company name and URL to get a comprehensive investment-grade analysis — "
=======
        "Enter a company name and URL to get a comprehensive investment-grade analysis "
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
        "covering market position, financials, team, risks, and strategic recommendations.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=4):
            prompt_input = gr.Textbox(
                placeholder='e.g. "Perform a comprehensive startup intelligence analysis on Mistral AI (https://mistral.ai)"',
                label="Company / Prompt",
                lines=2,
            )
        with gr.Column(scale=1, min_width=140):
            submit_btn = gr.Button("📊 Analyse", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", size="lg")

    gr.Examples(
        examples=EXAMPLES,
        inputs=prompt_input,
        label="Example Companies",
    )

    with gr.Row():
<<<<<<< HEAD
        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1, scale=1)

    report_output = gr.Markdown(value=PLACEHOLDER, elem_id="report")

    submit_btn.click(fn=run_analysis, inputs=prompt_input, outputs=[report_output, status_box])
    prompt_input.submit(fn=run_analysis, inputs=prompt_input, outputs=[report_output, status_box])
    clear_btn.click(fn=clear_all, outputs=[prompt_input, report_output, status_box])
=======
        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1, scale=3)
        download_btn = gr.DownloadButton(label="⬇ Download Report (.md)", visible=False, scale=1)

    report_output = gr.Markdown(value=PLACEHOLDER, elem_id="report")

    submit_btn.click(fn=run_analysis, inputs=prompt_input, outputs=[report_output, status_box, download_btn])
    prompt_input.submit(fn=run_analysis, inputs=prompt_input, outputs=[report_output, status_box, download_btn])
    clear_btn.click(fn=clear_all, outputs=[prompt_input, report_output, status_box, download_btn])
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="violet"))
