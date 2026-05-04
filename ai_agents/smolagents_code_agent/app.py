import os
import queue
import threading
import gradio as gr
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

from tools import WikipediaTool

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


# -- Formatting ----------------------------------------------------------------

def format_step(step_log) -> str:
    """Convert a smolagents step log object into a human-readable trace block."""
    parts = []
    step_num = getattr(step_log, "step_number", None)
    step_type = type(step_log).__name__

    header_label = f"STEP {step_num}" if step_num is not None else step_type.upper()
    parts.append(f"\n{'-' * 60}")
    parts.append(header_label)
    parts.append("-" * 60)

    # Planning text (PlanningStep)
    plan = getattr(step_log, "plan", None)
    if plan:
        parts.append(f"\n[PLAN]\n{plan.strip()}")

    # LLM output -- contains the agent's reasoning and generated code
    llm_output = getattr(step_log, "llm_output", None)
    if llm_output:
        parts.append(f"\n[THINK / ACT]\n{llm_output.strip()}")

    # Explicit tool calls (non-code agents)
    tool_calls = getattr(step_log, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            name = getattr(tc, "name", "?")
            args = getattr(tc, "arguments", None)
            parts.append(f"\n[TOOL CALL] {name}")
            if args:
                parts.append(f"Arguments: {args}")

    # Observations -- execution result / tool output
    observations = getattr(step_log, "observations", None)
    if observations:
        parts.append(f"\n[OBSERVE]\n{str(observations).strip()}")

    # Error
    error = getattr(step_log, "error", None)
    if error:
        parts.append(f"\n[ERROR]\n{error}")

    return "\n".join(parts) + "\n"


# -- Agent runner --------------------------------------------------------------

def run_agent(task: str):
    """
    Gradio generator: yields (trace, final_answer, spinner_update) after every
    agent step so the UI updates in real time.
    """
    if not task.strip():
        yield "Please enter a task.", "", gr.update(visible=False)
        return

    step_queue: queue.Queue = queue.Queue()
    result_container: list = []
    error_container: list = []

    def _step_callback(step_log):
        step_queue.put(("step", step_log))

    def _run():
        try:
            agent = CodeAgent(
                tools=[DuckDuckGoSearchTool(), WikipediaTool()],
                model=LiteLLMModel(
                    model_id="mistral/mistral-small-latest",
                    api_key=MISTRAL_API_KEY,
                ),
                step_callbacks=[_step_callback],
                max_steps=10,
                verbosity_level=0,
            )
            result = agent.run(task)
            result_container.append(result)
        except Exception as exc:
            error_container.append(str(exc))
        finally:
            step_queue.put(("done", None))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    trace = f"Task: {task}\n\nStarting agent...\n"
    final_answer = ""

    # Show spinner immediately
    yield trace, final_answer, gr.update(visible=True)

    while True:
        try:
            msg_type, step_log = step_queue.get(timeout=120)
        except queue.Empty:
            trace += "\n[TIMEOUT] Agent did not respond within 120 s.\n"
            yield trace, final_answer, gr.update(visible=False)
            return

        if msg_type == "done":
            break

        trace += format_step(step_log)
        yield trace, final_answer, gr.update(visible=True)

    thread.join(timeout=5)

    if error_container:
        trace += f"\n[ERROR]\n{error_container[0]}\n"
        yield trace, final_answer, gr.update(visible=False)
        return

    if result_container:
        final_answer = str(result_container[0])

    trace += "\n" + "-" * 60 + "\nAgent finished.\n"
    yield trace, final_answer, gr.update(visible=False)


# -- Gradio UI -----------------------------------------------------------------

with gr.Blocks(title="Smolagents Code Agent") as demo:
    gr.Markdown(
        """
# Smolagents Code Agent
Powered by **Mistral Small** via LiteLLM and **HuggingFace smolagents**.
Enter a task in plain English and watch the agent reason through it step by step.
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            task_input = gr.Textbox(
                label="Task",
                placeholder=(
                    'e.g. "Find the latest news about AI agents and summarize the top 3 stories"'
                ),
                lines=3,
            )
        with gr.Column(scale=1, min_width=140):
            run_btn = gr.Button("Run Agent", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

    spinner = gr.HTML(
        value=(
            '<div style="text-align:center;padding:10px;background:#eef2ff;'
            'border-radius:8px;font-weight:600;color:#4338ca;">'
            "Agent is running...</div>"
        ),
        visible=False,
    )

    trace_output = gr.Textbox(
        label="Reasoning Trace (Think -> Act -> Observe)",
        lines=20,
        max_lines=40,
        interactive=False,
    )

    final_output = gr.Textbox(
        label="Final Answer",
        lines=6,
        interactive=False,
    )

    run_btn.click(
        fn=run_agent,
        inputs=[task_input],
        outputs=[trace_output, final_output, spinner],
    )

    clear_btn.click(
        fn=lambda: ("", "", gr.update(visible=False)),
        inputs=[],
        outputs=[trace_output, final_output, spinner],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
