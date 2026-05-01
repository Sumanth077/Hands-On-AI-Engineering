"""
Startup Analyst — Gradio UI
Elite startup due-diligence powered by MiniMax M2.5 via OpenRouter.

Usage:
    uv run python app.py
"""

from dotenv import load_dotenv

load_dotenv()

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


# ── Analysis Function ──────────────────────────────────────────────────────────

def run_analysis(prompt: str):
    prompt = prompt.strip()
    if not prompt:
        yield PLACEHOLDER, ""
        return

    yield "_Analysing company — scraping website, crawling pages, gathering data..._", "Running..."

    full_output = ""
    try:
        for chunk in analyst.run(prompt, stream=True):
            if chunk.content:
                full_output += chunk.content
                # Only display once the structured report begins (first # heading)
                heading_idx = full_output.find("#")
                if heading_idx != -1:
                    yield full_output[heading_idx:], "Running..."
    except Exception as exc:
        heading_idx = full_output.find("#")
        report = full_output[heading_idx:] if heading_idx != -1 else full_output
        yield (report or PLACEHOLDER) + f"\n\n---\n\n**Error:** {exc}", f"Error: {exc}"
        return

    if full_output:
        heading_idx = full_output.find("#")
        yield full_output[heading_idx:] if heading_idx != -1 else full_output, "Done."
    else:
        yield PLACEHOLDER, "No response received."


def clear_all():
    return "", PLACEHOLDER, ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Startup Analyst") as demo:

    gr.Markdown("# 📊 Startup Analyst", elem_id="title")
    gr.Markdown(
        "Elite startup due-diligence powered by **MiniMax M2.5** via OpenRouter. "
        "Enter a company name and URL to get a comprehensive investment-grade analysis — "
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
        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1, scale=1)

    report_output = gr.Markdown(value=PLACEHOLDER, elem_id="report")

    submit_btn.click(fn=run_analysis, inputs=prompt_input, outputs=[report_output, status_box])
    prompt_input.submit(fn=run_analysis, inputs=prompt_input, outputs=[report_output, status_box])
    clear_btn.click(fn=clear_all, outputs=[prompt_input, report_output, status_box])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="violet"))
