"""Gradio UI for the AI Debate Agent.

Two LLM agents argue opposing sides of a user-defined topic for a configurable
number of rounds, and a judge agent scores each argument and declares a winner.
The transcript streams in live as each agent takes its turn.
"""

from __future__ import annotations

import gradio as gr

from debate import (
    AGAINST,
    DEBATER_A,
    DEBATER_A_MODEL,
    DEBATER_B,
    DEBATER_B_MODEL,
    FOR,
    JUDGE_MODEL,
    run_debate,
)


def _render_transcript(transcript: list[dict]) -> str:
    """Format the running transcript as markdown."""
    if not transcript:
        return "_The debate will appear here…_"
    blocks = []
    for arg in transcript:
        if arg["speaker"] == DEBATER_A:
            badge = f"🟦 **{DEBATER_A} · {FOR}**"
        else:
            badge = f"🟥 **{DEBATER_B} · {AGAINST}**"
        blocks.append(
            f"### Round {arg['round']} — {badge}\n\n{arg['content']}"
        )
    return "\n\n---\n\n".join(blocks)


def _render_verdict(verdict: dict) -> str:
    """Format the judge's verdict as markdown."""
    if not verdict:
        return "_Awaiting the judge's verdict…_"

    totals = verdict.get("totals", {})
    a_total = totals.get(DEBATER_A, 0)
    b_total = totals.get(DEBATER_B, 0)
    winner = verdict.get("winner", "Tie")

    if winner == DEBATER_A:
        headline = f"🏆 Winner: **{DEBATER_A} ({FOR})**"
    elif winner == DEBATER_B:
        headline = f"🏆 Winner: **{DEBATER_B} ({AGAINST})**"
    else:
        headline = "🤝 Result: **Tie**"

    lines = [
        f"## ⚖️ Judge's Verdict ({JUDGE_MODEL})",
        "",
        headline,
        "",
        f"**Total scores** — {DEBATER_A} (For): `{a_total}` · {DEBATER_B} (Against): `{b_total}`",
        "",
    ]

    args = verdict.get("arguments", [])
    if args:
        lines.append("### Score breakdown")
        lines.append("")
        lines.append("| Round | Debater | Logic | Evidence | Persuasion | Notes |")
        lines.append("|:-----:|:--------|:-----:|:--------:|:----------:|:------|")
        for a in args:
            lines.append(
                f"| {a.get('round', '-')} | {a.get('speaker', '-')} "
                f"| {a.get('logic', '-')} | {a.get('evidence', '-')} "
                f"| {a.get('persuasiveness', '-')} | {a.get('comment', '')} |"
            )
        lines.append("")

    if verdict.get("verdict"):
        lines.append("### Reasoning")
        lines.append("")
        lines.append(verdict["verdict"])

    return "\n".join(lines)


def debate_handler(topic: str, rounds: int):
    """Gradio streaming callback: yields (status, transcript_md, verdict_md)."""
    if not (topic or "").strip():
        yield "⚠️ Please enter a debate topic.", "", ""
        return

    yield "🚀 Starting debate…", "_The debate will appear here…_", ""

    try:
        for state in run_debate(topic, rounds):
            transcript = state.get("transcript", [])
            verdict = state.get("verdict", {})

            if verdict:
                status = "✅ Debate complete — verdict delivered."
            elif transcript:
                latest = transcript[-1]
                status = (
                    f"🗣️ Round {latest['round']}: {latest['speaker']} "
                    f"({latest['position']}) just argued…"
                )
            else:
                status = "🚀 Starting debate…"

            yield status, _render_transcript(transcript), _render_verdict(verdict)
    except Exception as exc:  # surface configuration / API errors in the UI
        detail = str(exc).strip() or type(exc).__name__
        cause = exc.__cause__
        if cause and str(cause) not in detail:
            detail = f"{detail} — {cause}"
        yield f"❌ Error: {detail}", "", ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Debate Agent") as demo:
        gr.Markdown(
            f"""
            # 🎙️ AI Debate Agent
            Two LLM agents argue opposing sides of your topic; a judge scores every
            argument and declares a winner.

            - 🟦 **Debater A — For** · `{DEBATER_A_MODEL}`
            - 🟥 **Debater B — Against** · `{DEBATER_B_MODEL}`
            - ⚖️ **Judge** · `{JUDGE_MODEL}`

            All three agents are routed through [Orq.ai](https://orq.ai/).
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                topic = gr.Textbox(
                    label="Debate topic",
                    placeholder="e.g. Social media does more harm than good",
                    lines=2,
                )
            with gr.Column(scale=1):
                rounds = gr.Slider(
                    label="Number of rounds",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=3,
                )

        run_btn = gr.Button("Start Debate", variant="primary")

        status = gr.Markdown("_Ready when you are._")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🗣️ Debate transcript")
                transcript_out = gr.Markdown("_The debate will appear here…_")
            with gr.Column():
                gr.Markdown("## 🏆 Result")
                verdict_out = gr.Markdown("_Awaiting the judge's verdict…_")

        run_btn.click(
            fn=debate_handler,
            inputs=[topic, rounds],
            outputs=[status, transcript_out, verdict_out],
        )
        topic.submit(
            fn=debate_handler,
            inputs=[topic, rounds],
            outputs=[status, transcript_out, verdict_out],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="rose"))
