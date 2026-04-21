import gradio as gr
import io
from contextlib import redirect_stdout
from agents_logic import get_research_crew

def run_analysis(my_company, competitor, pain_point, goal):
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
            
    return f.getvalue(), final_output

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
            logs = gr.Textbox(label="Agent Reasoning Process", lines=10)
            output = gr.Markdown(label="Final Battlecard")

    submit_btn.click(
        fn=run_analysis, 
        inputs=[my_co, comp_co, pain, goal], 
        outputs=[logs, output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)