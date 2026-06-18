"""Gradio UI for the Competitive Intelligence Agent, which generates AI-powered sales battlecards using a CrewAI multi-agent pipeline."""
import re
import gradio as gr
import io
from contextlib import redirect_stdout
from agents_logic import get_research_crew


def strip_ansi(text: str) -> str:
    """Remove ANSI color and formatting escape sequences from a string."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def clean_log(raw: str) -> str:
    """Remove blank lines and decorative separator lines from captured CrewAI stdout."""
    kept = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and re.search(r'[A-Za-z0-9]', stripped):
            kept.append(stripped)
    return '\n'.join(kept)


def run_analysis(my_company, competitor, pain_point, goal):
    """Run the three-agent CrewAI pipeline and return the reasoning log and final battlecard."""
    # Progress logging
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            crew = get_research_crew(my_company, competitor, pain_point, goal)
            result = crew.kickoff(inputs={
                "my_company": my_company,
                "competitor": competitor,
                "pain_point": pain_point,
                "goal": goal
            })
            final_output = result.raw
        except Exception as e:
            final_output = f"Error: {str(e)}"

    return clean_log(strip_ansi(f.getvalue())), final_output

with gr.Blocks(title="Strategic Intel System") as demo:
    gr.Markdown("# ⚔️ Competitive Intelligence Engine")
    gr.Markdown("Fill in the details below to generate a strategic sales battlecard.")
    
    with gr.Row():
        with gr.Column():
            my_co = gr.Textbox(label="Your Company Name")
            comp_co = gr.Textbox(label="Competitor Name")
            pain = gr.Textbox(label="Main Pain Point (What do you lose deals over?)")
            goal = gr.Textbox(label="Strategic Goal (e.g. Win enterprise deals)")
            submit_btn = gr.Button("Generate Battlecard", variant="primary")
            
        with gr.Column():
            with gr.Accordion("Agent Reasoning Process", open=False):
                logs = gr.Textbox(label="Agent Reasoning Process", lines=10)
            output = gr.Markdown(label="Final Battlecard")

    submit_btn.click(
        fn=run_analysis, 
        inputs=[my_co, comp_co, pain, goal], 
        outputs=[logs, output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)