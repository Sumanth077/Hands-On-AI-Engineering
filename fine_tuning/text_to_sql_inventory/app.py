"""Gradio UI for Text-to-SQL Inventory Specialist."""

import sqlite3

import gradio as gr
import pandas as pd

from database import execute_query, init_db
from inference import generate_sql, load_model

MODEL_NAME = "Hecodes/text-to-sql-inventory"

EXAMPLE_QUESTIONS = [
    "How many products are in each category?",
    "Which products have stock below their reorder point?",
    "Which brake pads have less than 10 units in stock?",
    "What is the total inventory value by warehouse?",
    "Show pending orders with product names and supplier names.",
    "Which supplier has the most orders?",
    "List the top 5 most expensive products.",
    "How many orders were placed in May 2025?",
]

init_db()
_tokenizer, _model = load_model()


def process_question(question: str) -> tuple[str, pd.DataFrame | None, str]:
    """Generate SQL, execute it, and return results or friendly errors."""
    if not question or not question.strip():
        return "", None, "Please enter a question."

    try:
        sql = generate_sql(question, _tokenizer, _model)
    except Exception as exc:
        return (
            "",
            None,
            f"Could not generate SQL. Ensure the Hugging Face model "
            f"'{MODEL_NAME}' is available and set HF_TOKEN in .env if the model "
            f"is gated. Error: {exc}",
        )

    if not sql:
        return "", None, "The model did not return a valid SQL query. Try rephrasing your question."

    try:
        columns, rows = execute_query(sql)
        if not columns:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(rows, columns=columns)
        return sql, df, ""
    except ValueError as exc:
        return sql, None, f"Query blocked for safety: {exc}"
    except sqlite3.Error as exc:
        return sql, None, f"SQL error: {exc}. Try rephrasing your question."
    except Exception as exc:
        return sql, None, f"Unexpected error: {exc}"


def create_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    with gr.Blocks(title="Text-to-SQL Inventory Specialist") as demo:
        gr.Markdown(
            "# Text-to-SQL Inventory Specialist\n"
            "Ask questions about retail inventory in plain English. "
            "A fine-tuned Qwen3.5-2B model generates SQL and runs it against the inventory database."
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Which products need reordering?",
                lines=2,
            )

        with gr.Row():
            submit_btn = gr.Button("Run Query", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=question_input)

        sql_output = gr.Code(label="Generated SQL", language="sql", lines=4)
        error_output = gr.Textbox(label="Status", interactive=False)
        results_output = gr.Dataframe(label="Query Results", wrap=True)

        submit_btn.click(
            fn=process_question,
            inputs=question_input,
            outputs=[sql_output, results_output, error_output],
        )
        question_input.submit(
            fn=process_question,
            inputs=question_input,
            outputs=[sql_output, results_output, error_output],
        )
        clear_btn.click(
            fn=lambda: ("", None, ""),
            outputs=[question_input, sql_output, results_output],
        ).then(
            fn=lambda: "",
            outputs=error_output,
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
