"""
Offline Medical Agent - Gradio UI.
All inference and retrieval happen locally. No internet required after first setup.
"""

import gradio as gr

from offline_medical_agent.agents import OfflineMedicalAgent
from offline_medical_agent.retriever import ProtocolRetriever

PROTOCOLS_DIR = "./protocols"
DB_PATH = "./qdrant_data"

# ---------------------------------------------------------------------------
# Initialise agent (loads model + auto-ingests protocols on first run)
# ---------------------------------------------------------------------------

print("Initialising Offline Medical Agent...")
agent = OfflineMedicalAgent(db_path=DB_PATH, protocols_dir=PROTOCOLS_DIR)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def query_agent(situation: str):
    if not situation.strip():
        return "", "", "", ""

    result = agent.run(situation)

    response = result["response"]
    protocol_info = (
        f"**{result['protocol_title']}**  "
        f"(source: `{result['source']}` | relevance score: `{result['score']}`)\n\n"
        f"{result['protocol_content']}"
        if result["protocol_title"]
        else "No protocol retrieved."
    )
    return response, protocol_info


def reload_protocols():
    retriever = ProtocolRetriever(db_path=DB_PATH)
    n = retriever.ingest(PROTOCOLS_DIR)
    return f"Loaded {n} protocol(s) from `{PROTOCOLS_DIR}`."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Offline Medical Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Offline Medical Agent")
    gr.Markdown(
        "> **Clinical decision support for resource-limited, offline environments.**  \n"
        "> Responses are grounded strictly in the retrieved protocol. "
        "Always apply professional clinical judgment. This tool does not replace it."
    )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            situation_input = gr.Textbox(
                label="Patient Situation",
                placeholder="e.g. 8-year-old child with fever of 39.2°C, drinking well, no danger signs...",
                lines=5,
            )
            submit_btn = gr.Button("Get Clinical Guidance", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Protocol Library")
            reload_btn = gr.Button("Re-load Protocols", variant="secondary")
            reload_status = gr.Markdown("")

    gr.Markdown("---")

    gr.Markdown("### Clinical Guidance")
    response_output = gr.Markdown()

    with gr.Accordion("Retrieved Protocol", open=False):
        protocol_output = gr.Markdown("")

    # ------------------------------------------------------------------
    # Example queries
    # ------------------------------------------------------------------
    gr.Markdown("### Example Queries")
    gr.Examples(
        examples=[
            ["Child 4 years old, temperature 38.8°C, crying but alert, drinking fluids, no rash or stiff neck. What antipyretic and dose?"],
            ["Adult patient collapsed, hives across chest and arms, wheezing, BP 80/50 after eating peanuts. What do I do immediately?"],
            ["3-year-old with 3 days of watery diarrhoea, sunken eyes, drinking eagerly, skin pinch returns in 1 second. Plan?"],
            ["Machete laceration to forearm, 4 cm long, occurred 2 hours ago, bleeding controlled, clean wound. How do I manage this?"],
            ["Adult male, fever for 3 days, chills, headache, in malaria-endemic region, RDT Pf positive. Treatment?"],
        ],
        inputs=[situation_input],
    )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    submit_btn.click(
        fn=query_agent,
        inputs=[situation_input],
        outputs=[response_output, protocol_output],
    )
    situation_input.submit(
        fn=query_agent,
        inputs=[situation_input],
        outputs=[response_output, protocol_output],
    )
    reload_btn.click(fn=reload_protocols, outputs=[reload_status])


if __name__ == "__main__":
    demo.launch()
