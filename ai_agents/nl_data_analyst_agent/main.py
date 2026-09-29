"""Gradio interface for the Eve analyst."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from jev_review import Review, review_answer, review_question
from seed_data import seed_database

ROOT = Path(__file__).resolve().parent
DEMO_DATABASE = ROOT / "data" / "demo.sqlite"
REGISTRY = ROOT / "data" / "connections.json"
EVE_URL = os.getenv("EVE_URL", "http://127.0.0.1:3000").rstrip("/")
DEMO_METRICS = ["completed-order revenue = quantity * unit_price for orders where status = 'completed'"]


def schema_text(path: Path) -> str:
    if not path.is_file():
        raise ValueError("Database file not found.")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        result = []
        for (table,) in tables:
            quoted = table.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            result.append(f"{table}: " + ", ".join(f"{col[1]} {col[2]}" for col in columns))
        return "\n".join(result)
    finally:
        connection.close()


def connect_database(uploaded_path: str | None, local_path: str, demo: bool = False) -> tuple[Any, ...]:
    path = Path(uploaded_path or local_path.strip()).expanduser() if uploaded_path or local_path.strip() else None
    if path is None:
        raise gr.Error("Upload a SQLite file or enter its local path first.")
    try:
        schema = schema_text(path)
    except Exception as error:
        raise gr.Error(f"Could not open that SQLite database: {error}") from error
    if not schema:
        raise gr.Error("The database has no user tables.")
    database_id = "demo" if demo else uuid.uuid4().hex
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    registry[database_id] = str(path.resolve())
    REGISTRY.write_text(json.dumps(registry, indent=2))
    tables = [line.split(":", 1)[0] for line in schema.splitlines()]
    suggestions = (["Which region had the most completed-order revenue in 2025?",
                    "What were monthly completed-order revenues in 2025?",
                    "Show the first 20 orders with customer and product names."] if demo else
                   [f"How many rows are in {table}?" for table in tables[:2]])
    return (database_id, f"Connected: `{path.name}` (read-only)", schema, [], "", pd.DataFrame(),
            None, None, gr.update(visible=False), gr.update(visible=False),
            gr.update(choices=suggestions, value=None))


def use_demo_database() -> tuple[Any, ...]:
    if not DEMO_DATABASE.exists():
        seed_database(DEMO_DATABASE)
    return connect_database(None, str(DEMO_DATABASE), demo=True)


def eve_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(EVE_URL + path, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def present_grounding_review(answer: str, grounding: Review) -> tuple[str, str | None]:
    if grounding.actionable:
        return (
            f"⚠️ **Unverified draft — awaiting user review.** {grounding.message()}\n\n"
            "The explanation below is not treated as final. Compare it with the displayed rows, or submit: "
            "“Revise the previous answer using only the displayed SQL rows and explicitly correct unsupported claims.”\n\n"
            f"**Eve draft (unverified):**\n\n{answer}",
            None,
        )
    if grounding.status != "ok":
        return answer, grounding.message() + " The answer is shown because uncertain or failed reviews do not block it."
    return answer, None


def project_eve_events(
    events: list[dict[str, Any]], question: str, focus_call_id: str | None = None,
) -> tuple[str, str, pd.DataFrame, dict[str, Any] | None]:
    answer, pending = "", None
    proposed_sql: dict[str, str] = {}
    completed: dict[str, dict[str, Any]] = {}
    completed_order: list[str] = []
    sql_reviews: list[Review] = []
    for event in events:
        kind, data = event.get("type"), event.get("data") or {}
        if kind == "actions.requested":
            for action in data.get("actions", []):
                if action.get("toolName", "").endswith("run_sql"):
                    call_id = action.get("callId")
                    sql = action.get("input", {}).get("sql")
                    if call_id and isinstance(sql, str):
                        proposed_sql[call_id] = sql
        elif kind == "action.result":
            result = data.get("result") or {}
            output = result.get("output") or {}
            call_id = result.get("callId")
            if isinstance(output, dict):
                sql = output.get("sql") or proposed_sql.get(call_id, "")
                if call_id and "rows" in output:
                    completed[call_id] = {
                        "callId": call_id,
                        "sql": sql,
                        "rows": output["rows"],
                    }
                    if call_id not in completed_order:
                        completed_order.append(call_id)
                review = output.get("review")
                if isinstance(review, dict) and (review.get("actionable") or review.get("status") != "ok"):
                    sql_reviews.append(Review(**review))
        elif kind == "input.requested":
            for request in data.get("requests") or []:
                action = request.get("action") or {}
                if request.get("kind") == "tool-approval" and action.get("toolName", "").endswith("run_sql"):
                    call_id = action.get("callId")
                    sql = action.get("input", {}).get("sql") or proposed_sql.get(call_id, "")
                    pending = {"requestId": request["requestId"], "callId": call_id, "sql": sql}
                    break
        elif kind == "message.completed" and data.get("finishReason") != "tool-calls":
            answer = data.get("message") or answer
        elif kind in {"turn.failed", "session.failed"}:
            answer = f"Eve could not complete this question: {data.get('message', 'Unknown error')}"

    selected: dict[str, Any] | None = None
    if pending:
        prior_ids = [call_id for call_id in completed_order if call_id != pending.get("callId")]
        if prior_ids:
            pending["priorCompleted"] = completed[prior_ids[-1]]
        sql = pending.get("sql", "")
        rows = pd.DataFrame()
    else:
        selected_id = focus_call_id if focus_call_id in completed else (completed_order[-1] if completed_order else None)
        selected = completed.get(selected_id) if selected_id else None
        sql = selected.get("sql", "") if selected else ""
        rows = pd.DataFrame(selected.get("rows", [])) if selected else pd.DataFrame()

    notes = [review.message() for review in sql_reviews]
    if answer and sql and not pending:
        grounding = review_answer(question, sql, rows.to_dict(orient="records"), answer)
        answer, grounding_note = present_grounding_review(answer, grounding)
        if grounding_note:
            notes.append(grounding_note)
    if notes:
        answer = answer + ("\n\n" if answer else "") + "\n\n".join(f"> {note}" for note in notes)
    return answer, sql, rows, pending


def consume_events(
    session: dict[str, Any], question: str, focus_call_id: str | None = None,
) -> tuple[str, str, pd.DataFrame, dict[str, Any] | None]:
    events: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(15, read=180)) as client:
        with client.stream("GET", EVE_URL + f"/eve/v1/session/{session['id']}/stream",
                           params={"startIndex": session["cursor"]}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                session["cursor"] += 1
                events.append(event)
                if event.get("type") == "session.waiting":
                    break
    return project_eve_events(events, question, focus_call_id)


def ask(question: str, chat: list[dict[str, str]], pending: dict[str, Any] | None,
        database_id: str | None, session: dict[str, Any] | None) -> tuple[Any, ...]:
    if not database_id:
        raise gr.Error("Connect a SQLite database or select the demo first.")
    if pending:
        raise gr.Error("Approve or reject the pending query first.")
    if not question.strip():
        return chat, "", "", pd.DataFrame(), None, session, gr.update(visible=False), gr.update(visible=False)
    chat = chat + [{"role": "user", "content": question.strip()}]
    schema = schema_text(Path(json.loads(REGISTRY.read_text())[database_id]))
    clarity = review_question(question.strip(), schema, DEMO_METRICS if database_id == "demo" else [])
    if clarity.actionable:
        chat.append({"role": "assistant", "content": clarity.message() + " Please clarify that point, then submit the question again."})
        return chat, question, "", pd.DataFrame(), None, session, gr.update(visible=False), gr.update(visible=False)
    try:
        message = f"Database ID: {database_id}\nQuestion: {question.strip()}"
        if session is None:
            created = eve_post("/eve/v1/session", {"message": message})
            session = {"id": created["sessionId"], "cursor": 0, "question": question.strip()}
        else:
            eve_post(f"/eve/v1/session/{session['id']}", {"message": message})
            session["question"] = question.strip()
        answer, sql, rows, pending = consume_events(session, question.strip())
        if clarity.status != "ok":
            answer = (answer + "\n\n" if answer else "") + f"> {clarity.message()} The question proceeded because this review was not confident."
        content = (
            "This query needs approval before Eve runs it. Review the SQL and choose Approve or Reject."
            if pending else answer or "Eve returned no answer. Check its terminal for details.")
        if pending and pending.get("priorCompleted"):
            content += (
                "\n\nA different SQL call completed in the same turn. Its rows are hidden while you review "
                "the SQL awaiting approval and will be restored with a clear label if you reject it."
            )
        if pending and clarity.status != "ok":
            content += f"\n\n> {clarity.message()} The question proceeded because this review was not confident."
        chat.append({"role": "assistant", "content": content})
        return chat, "", sql, rows, pending, session, gr.update(visible=bool(pending)), gr.update(visible=bool(pending))
    except (httpx.HTTPError, OSError, ValueError) as error:
        chat.append({"role": "assistant", "content": f"Could not reach Eve: {error}. Check the Eve server and gateway key."})
        return chat, "", "", pd.DataFrame(), None, session, gr.update(visible=False), gr.update(visible=False)


def decide(approved: bool, chat: list[dict[str, str]], pending: dict[str, Any] | None,
           session: dict[str, Any] | None) -> tuple[Any, ...]:
    if not pending or not session:
        raise gr.Error("No query is waiting for approval.")
    try:
        eve_post(f"/eve/v1/session/{session['id']}", {"inputResponses": [{"requestId": pending["requestId"],
                  "optionId": "approve" if approved else "cancel"}]})
        answer, sql, rows, next_pending = consume_events(
            session, session.get("question", ""), focus_call_id=pending.get("callId"),
        )
        if not approved and not next_pending and pending.get("priorCompleted"):
            prior = pending["priorCompleted"]
            sql = prior.get("sql", "")
            rows = pd.DataFrame(prior.get("rows", []))
            label = (
                "The displayed SQL and rows belong to an earlier completed query "
                f"(`{prior.get('callId', 'unknown call')}`), not the rejected query."
            )
            answer = answer + ("\n\n" if answer else "") + label
        chat = chat + [{"role": "assistant", "content": answer or ("Query rejected." if not approved else "Eve returned no answer.")}]
        return chat, sql, rows, next_pending, session, gr.update(visible=bool(next_pending)), gr.update(visible=bool(next_pending))
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise gr.Error(f"Could not send approval to Eve: {error}") from error


def clear() -> tuple[Any, ...]:
    return [], "", "", pd.DataFrame(), None, None, gr.update(visible=False), gr.update(visible=False)


with gr.Blocks(title="NL Data Analyst Agent") as demo:
    gr.Markdown("# NL Data Analyst Agent\nConnect a SQLite database, ask questions, inspect Eve's SQL, and approve broad queries.")
    with gr.Row():
        upload = gr.File(label="Upload SQLite", file_types=[".sqlite", ".db", ".sqlite3"], type="filepath")
        local_path = gr.Textbox(label="Or local SQLite path", value=os.getenv("DATABASE_PATH", ""))
    with gr.Row():
        connect_button = gr.Button("Connect database", variant="primary")
        demo_button = gr.Button("Use demo database")
    connection_status = gr.Markdown("No database connected.")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=460, label="Conversation")
            question = gr.Textbox(label="Your question")
            suggestions = gr.Dropdown(label="Suggested questions", choices=[], interactive=True)
            with gr.Row():
                submit = gr.Button("Analyze", variant="primary")
                clear_button = gr.Button("Clear conversation")
            with gr.Row():
                approve_button = gr.Button("Approve query", visible=False, variant="primary")
                reject_button = gr.Button("Reject query", visible=False)
        with gr.Column(scale=2):
            gr.Markdown("### Database schema")
            schema_view = gr.Code(language="sql", interactive=False)
            sql_view = gr.Code(label="Generated SQL", language="sql", interactive=False)
            results = gr.Dataframe(label="Query results", interactive=False)
    database_state, pending_state, session_state = gr.State(None), gr.State(None), gr.State(None)
    connect_outputs = [database_state, connection_status, schema_view, chatbot, sql_view, results,
                       pending_state, session_state, approve_button, reject_button, suggestions]
    connect_button.click(connect_database, [upload, local_path], connect_outputs)
    demo_button.click(use_demo_database, outputs=connect_outputs)
    suggestions.change(lambda value: value or "", suggestions, question)
    ask_outputs = [chatbot, question, sql_view, results, pending_state, session_state, approve_button, reject_button]
    submit.click(ask, [question, chatbot, pending_state, database_state, session_state], ask_outputs)
    question.submit(ask, [question, chatbot, pending_state, database_state, session_state], ask_outputs)
    clear_button.click(clear, outputs=ask_outputs)
    decide_outputs = [chatbot, sql_view, results, pending_state, session_state, approve_button, reject_button]
    approve_button.click(lambda chat, pending, session: decide(True, chat, pending, session),
                         [chatbot, pending_state, session_state], decide_outputs)
    reject_button.click(lambda chat, pending, session: decide(False, chat, pending, session),
                        [chatbot, pending_state, session_state], decide_outputs)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
