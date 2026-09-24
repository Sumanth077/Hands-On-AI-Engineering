import gradio as gr

from agent.confirmation import build_default_review_fields, build_incident_from_review
from agent.harness import run_investigation
from memory.incident_store import save_incident
from models.schemas import MachineReading
from simulator.machine import Machine
from simulator.scenarios import trigger_fault
from util.network_status import is_online

machine = Machine()

TICK_SECONDS = 1
NETWORK_CHECK_SECONDS = 5

TOOL_LABELS = {
    "get_current_readings": "Checked current readings",
    "get_recent_history": "Reviewed recent history",
    "search_manual": "Searched equipment manual",
    "search_past_incidents": "Searched previous incidents",
}

INITIAL_INVESTIGATION_MD = "### Investigation\n\n" + "\n".join(
    f"- ○ {label}" for label in TOOL_LABELS.values()
)
INITIAL_MEMORY_MD = "### Retrieved Memory\n\nNo similar previous incident found."
INITIAL_RECOMMENDATION_MD = "### Recommendation\n\n_Not started yet._"


def _format_status(reading: MachineReading) -> tuple[str, str, str, str, str]:
    return (
        f"{reading.temperature_c:.1f}",
        f"{reading.vibration_mm_s:.2f}",
        f"{reading.pressure_bar:.2f}",
        reading.status,
        reading.error_code or "None",
    )


def _current_status() -> tuple[str, str, str, str, str]:
    return _format_status(machine.get_current_readings())


def _network_status_markdown() -> str:
    if is_online():
        return "🟢 **Online**"
    return "🔴 **Offline** (running locally)"


def _tick() -> tuple[str, str, str, str, str]:
    machine.advance_simulation(1)
    return _current_status()


def _trigger_bearing() -> tuple[str, str, str, str, str]:
    trigger_fault(machine, "bearing")
    return _current_status()


def _trigger_cooling() -> tuple[str, str, str, str, str]:
    trigger_fault(machine, "cooling")
    return _current_status()


def _trigger_pressure() -> tuple[str, str, str, str, str]:
    trigger_fault(machine, "pressure")
    return _current_status()


def _reset() -> tuple[str, str, str, str, str]:
    machine.reset_machine()
    return _current_status()


def _build_investigation_objective() -> str:
    reading = machine.get_current_readings()
    if reading.error_code:
        return (
            f"Investigate {reading.error_code} fault on Machine A "
            f"(status: {reading.status})."
        )
    return f"Investigate current condition of Machine A (status: {reading.status})."


def _format_investigation_markdown(state) -> str:
    completed_tool_names = {call["name"] for call in state.tool_calls}
    checklist_lines = [
        f"- {'✓' if name in completed_tool_names else '○'} {label}"
        for name, label in TOOL_LABELS.items()
    ]
    sections = ["### Investigation", "\n".join(checklist_lines)]

    if state.findings:
        finding_lines = [f"- {finding}" for finding in state.findings]
        sections.append("**Findings:**\n" + "\n".join(finding_lines))

    return "\n\n".join(sections)


def _format_retrieved_memory_markdown(state) -> str:
    if not state.retrieved_incidents:
        return "### Retrieved Memory\n\nNo similar previous incident found."

    lines = ["### Retrieved Memory", ""]
    for incident in state.retrieved_incidents:
        created_at = incident.get("created_at", "unknown date")
        symptoms = incident.get("symptoms") or []
        symptoms_text = ", ".join(symptoms) if symptoms else "not recorded"
        score = incident.get("score")
        similarity = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"

        lines.append("**Similar incident**")
        lines.append(f"- Date: {created_at}")
        lines.append(f"- Symptoms: {symptoms_text}")
        lines.append(f"- Cause: {incident.get('confirmed_cause', 'unknown')}")
        lines.append(f"- Fix: {incident.get('fix_applied', 'unknown')}")
        lines.append(f"- Similarity: {similarity}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_recommendation_markdown(state) -> str:
    if state.final_result is None:
        return "### Recommendation\n\n_Investigation in progress..._"

    result = state.final_result
    lines = [
        "### Recommendation",
        "",
        f"**Likely cause:** {result.get('likely_cause')}",
        f"**Confidence:** {result.get('confidence')}",
        f"**Recommendation:** {result.get('recommendation')}",
    ]

    evidence = result.get("supporting_evidence") or []
    if evidence:
        lines.append("")
        lines.append("**Supporting evidence:**")
        lines.extend(f"- {item}" for item in evidence)

    manual_sources = result.get("manual_sources") or []
    if manual_sources:
        lines.append("")
        lines.append("**Manual references:**")
        lines.extend(f"- {item}" for item in manual_sources)

    incident_ids = result.get("retrieved_incident_ids") or []
    if incident_ids:
        lines.append("")
        lines.append("**Past incidents used:**")
        lines.extend(f"- {item}" for item in incident_ids)

    return "\n".join(lines)


OUTCOME_OPTIONS = ["resolved", "unresolved", "needs follow-up"]


def _run_investigation():
    objective = _build_investigation_objective()
    for state in run_investigation(machine, machine.history, objective):
        review_ready = state.final_result is not None

        if review_ready:
            defaults = build_default_review_fields(state, state.current_readings)
            confirmed_cause_update = defaults["confirmed_cause"]
            fix_applied_update = defaults["fix_applied"]
            outcome_update = defaults["outcome"]
            notes_update = defaults["notes"]
            symptoms_update = ", ".join(defaults["symptoms"])
        else:
            confirmed_cause_update = gr.update()
            fix_applied_update = gr.update()
            outcome_update = gr.update()
            notes_update = gr.update()
            symptoms_update = gr.update()

        yield (
            _format_investigation_markdown(state),
            _format_retrieved_memory_markdown(state),
            _format_recommendation_markdown(state),
            state,
            gr.update(visible=review_ready),
            confirmed_cause_update,
            fix_applied_update,
            outcome_update,
            notes_update,
            symptoms_update,
            "",
        )


def _confirm_and_save(
    investigation_state, confirmed_cause, fix_applied, outcome, notes, symptoms_text
):
    if investigation_state is None or investigation_state.final_result is None:
        return gr.update(visible=True), "No completed investigation to save."

    symptoms = [s.strip() for s in symptoms_text.split(",") if s.strip()]
    incident = build_incident_from_review(
        investigation_state,
        investigation_state.current_readings,
        confirmed_cause=confirmed_cause,
        fix_applied=fix_applied,
        outcome=outcome,
        notes=notes,
        symptoms=symptoms,
    )
    save_incident(incident)

    return gr.update(visible=False), f"Saved incident `{incident.incident_id}` to memory."


def _discard_review():
    return gr.update(visible=False), "Discarded — nothing was saved to memory."


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Offline Troubleshooting Agent",
        analytics_enabled=False,
    ) as app:
        gr.Markdown("# Offline Troubleshooting Agent — Machine A")
        network_status_md = gr.Markdown(value=_network_status_markdown())

        with gr.Row():
            temperature = gr.Textbox(label="Temperature (°C)", interactive=False)
            vibration = gr.Textbox(label="Vibration (mm/s)", interactive=False)
            pressure = gr.Textbox(label="Pressure (bar)", interactive=False)
            status = gr.Textbox(label="Status", interactive=False)
            error_code = gr.Textbox(label="Error code", interactive=False)

        status_outputs = [temperature, vibration, pressure, status, error_code]

        with gr.Row():
            bearing_btn = gr.Button("Trigger Bearing Fault")
            cooling_btn = gr.Button("Trigger Cooling Fault")
            pressure_btn = gr.Button("Trigger Pressure Fault")
            reset_btn = gr.Button("Reset Machine")

        bearing_btn.click(
            fn=_trigger_bearing, inputs=None, outputs=status_outputs, api_name="trigger_bearing"
        )
        cooling_btn.click(
            fn=_trigger_cooling, inputs=None, outputs=status_outputs, api_name="trigger_cooling"
        )
        pressure_btn.click(
            fn=_trigger_pressure, inputs=None, outputs=status_outputs, api_name="trigger_pressure"
        )
        reset_btn.click(fn=_reset, inputs=None, outputs=status_outputs, api_name="reset_machine")

        timer = gr.Timer(TICK_SECONDS)
        timer.tick(fn=_tick, inputs=None, outputs=status_outputs, api_name="tick")

        # Separate, slower timer purely for the network status badge, kept
        # decoupled from the 1-second simulation timer so a network check
        # never adds lag to the machine readings updating every second. This
        # is purely informational display, not a gate on any functionality.
        network_timer = gr.Timer(NETWORK_CHECK_SECONDS)
        network_timer.tick(
            fn=_network_status_markdown,
            inputs=None,
            outputs=[network_status_md],
            api_name="network_status_tick",
        )

        app.load(fn=_current_status, inputs=None, outputs=status_outputs, api_name="current_status")
        app.load(
            fn=_network_status_markdown,
            inputs=None,
            outputs=[network_status_md],
            api_name="network_status_on_load",
        )

        start_investigation_btn = gr.Button("Start Investigation")

        with gr.Row():
            investigation_md = gr.Markdown(value=INITIAL_INVESTIGATION_MD)
            retrieved_memory_md = gr.Markdown(value=INITIAL_MEMORY_MD)
            recommendation_md = gr.Markdown(value=INITIAL_RECOMMENDATION_MD)

        investigation_state = gr.State(None)

        with gr.Group(visible=False) as review_group:
            gr.Markdown(
                "### Review & Confirm\n\nEdit anything below before saving — nothing is "
                "written to memory until you click Confirm."
            )
            confirmed_cause_tb = gr.Textbox(label="Confirmed cause", interactive=True)
            fix_applied_tb = gr.Textbox(label="Fix applied", interactive=True)
            outcome_dd = gr.Dropdown(
                label="Outcome", choices=OUTCOME_OPTIONS, value="resolved", interactive=True
            )
            notes_tb = gr.Textbox(label="Notes", interactive=True, lines=3)
            symptoms_tb = gr.Textbox(
                label="Symptoms (comma-separated)", interactive=True
            )
            with gr.Row():
                confirm_btn = gr.Button("Confirm & Save to Memory", variant="primary")
                discard_btn = gr.Button("Discard (don't save)")
            save_status_md = gr.Markdown(value="")

        start_investigation_btn.click(
            fn=_run_investigation,
            inputs=None,
            outputs=[
                investigation_md,
                retrieved_memory_md,
                recommendation_md,
                investigation_state,
                review_group,
                confirmed_cause_tb,
                fix_applied_tb,
                outcome_dd,
                notes_tb,
                symptoms_tb,
                save_status_md,
            ],
            api_name="start_investigation",
        )

        confirm_btn.click(
            fn=_confirm_and_save,
            inputs=[
                investigation_state,
                confirmed_cause_tb,
                fix_applied_tb,
                outcome_dd,
                notes_tb,
                symptoms_tb,
            ],
            outputs=[review_group, save_status_md],
            api_name="confirm_and_save",
        )

        discard_btn.click(
            fn=_discard_review,
            inputs=None,
            outputs=[review_group, save_status_md],
            api_name="discard_review",
        )

    return app
