"""
Telnyx AI Appointment Booking Voice Agent -- webhook backend, tools, and CRM dashboard.

Telnyx runs the voice, the models, and the telephony in the portal. This FastAPI
server contributes the parts that have to live in your code:

  POST /webhooks/dynamic-variables  -> live context injected at call start
  POST /tools/check-availability    -> tool the scheduling agent calls to read open slots
  POST /tools/book-appointment      -> tool that books a slot, makes an .ics, writes to the CRM
  POST /webhooks/call-summary       -> post-call record + Inference-generated follow-up
  GET  /                            -> lightweight CRM dashboard
  GET  /health                      -> readiness check

Run with:  uvicorn main:app --reload   (or: python main.py)
"""

from __future__ import annotations

import html
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from appointment_agent import crm, inference, scheduling
from appointment_agent.calendar_invite import build_invite
from appointment_agent.config import SETTINGS

# Log every tool call and its result to the console. This is the "show the
# payload live" view: keep this terminal on screen during a demo.
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("appointment-agent")


def _trace(label: str, data: Any) -> None:
    log.info("%s %s", label, json.dumps(data, default=str)[:1000])


app = FastAPI(title="Telnyx AI Appointment Booking Voice Agent")

# Allow the local demo page (opened as a file, or from any localhost port) to
# read the live JSON endpoints. This is a local demo server, not a public API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    crm.init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_args(body: dict[str, Any]) -> dict[str, Any]:
    """
    Pull the tool arguments out of a Telnyx tool-call webhook body.

    Telnyx posts the tool arguments when the model calls a tool. Different
    surfaces nest them slightly differently, so we look in the common places
    and fall back to the top-level body.
    """
    for key in ("arguments", "parameters", "data", "payload"):
        value = body.get(key)
        if isinstance(value, dict):
            return value
    return body


def _caller_phone(body: dict[str, Any], args: dict[str, Any]) -> str:
    for key in ("telnyx_end_user_target", "from", "caller_number", "phone", "caller_phone"):
        value = body.get(key) or args.get(key)
        if value:
            return str(value)
    return ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "telnyx-appointment-agent"}


@app.get("/demo")
def demo_page() -> FileResponse:
    """Serve the demo page over http://localhost:8000/demo (a secure context, so
    the voice widget's microphone works, unlike opening the file directly)."""
    return FileResponse(Path(__file__).parent / "demo.html")


# ---------------------------------------------------------------------------
# JSON API (read-only) used by the live demo page
# ---------------------------------------------------------------------------

@app.get("/api/appointments")
def api_appointments() -> list[dict]:
    return crm.list_appointments()


@app.get("/api/summaries")
def api_summaries() -> list[dict]:
    return crm.list_summaries()


@app.get("/api/business")
def api_business() -> dict[str, Any]:
    return {
        "business_name": SETTINGS.business_name,
        "business_hours": SETTINGS.business_hours,
        "services": SETTINGS.services,
        "model": SETTINGS.inference_model,
    }


# ---------------------------------------------------------------------------
# Dynamic variables webhook: live context injected into the agent prompt
# ---------------------------------------------------------------------------

@app.post("/webhooks/dynamic-variables")
async def dynamic_variables(request: Request) -> JSONResponse:
    """
    Telnyx calls this at the start of every call. The returned values fill the
    {{placeholders}} in the assistant's instructions, so the agent always knows
    the current date, hours, and services without redeploying anything.
    """
    today = date.today()
    variables = {
        "business_name": SETTINGS.business_name,
        "business_hours": SETTINGS.business_hours,
        "business_timezone": SETTINGS.business_timezone,
        "services": ", ".join(SETTINGS.services),
        "support_email": SETTINGS.support_email,
        "today": today.isoformat(),
        "today_pretty": today.strftime("%A %B %d, %Y"),
        "booking_horizon_days": str(SETTINGS.booking_horizon_days),
    }
    # Telnyx reads the returned JSON object directly as the map of dynamic
    # variables (flat, top-level keys), so return it as-is -- not nested.
    return JSONResponse(variables)


# ---------------------------------------------------------------------------
# Tool: check availability
# ---------------------------------------------------------------------------

@app.post("/tools/check-availability")
async def check_availability(request: Request) -> JSONResponse:
    """Tool the scheduling agent calls to read open slots for a date."""
    body = await request.json()
    args = _extract_args(body)
    _trace("check_availability  <-", args)

    date_str = str(args.get("date") or date.today().isoformat())
    service = args.get("service")

    try:
        slots = scheduling.available_slots(date_str, service)
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "available": []})

    # Keep the payload small: the model only needs a few options to offer.
    top = slots[:6]
    result = {
        "date": date_str,
        "service": service,
        "count": len(slots),
        "available": top,
        "message": (
            f"{len(slots)} open slots on {date_str}."
            if slots
            else f"No open slots on {date_str}. Offer another day."
        ),
    }
    _trace("check_availability  ->", result)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Tool: book appointment
# ---------------------------------------------------------------------------

@app.post("/tools/book-appointment")
async def book_appointment(request: Request) -> JSONResponse:
    """Tool that books a slot, generates a calendar invite, and writes to the CRM."""
    body = await request.json()
    args = _extract_args(body)
    _trace("book_appointment    <-", args)

    name = str(args.get("name") or args.get("caller_name") or "").strip()
    service = str(args.get("service") or "").strip()
    start = str(args.get("start") or args.get("start_time") or "").strip()
    phone = _caller_phone(body, args)

    missing = [f for f, v in {"name": name, "service": service, "start": start}.items() if not v]
    if missing:
        return JSONResponse(
            {"success": False, "error": f"Missing required fields: {', '.join(missing)}."}
        )

    try:
        booking = scheduling.book(name=name, phone=phone, service=service, start=start)
    except ValueError as exc:
        return JSONResponse({"success": False, "error": str(exc)})

    _, ics_path = build_invite(booking)

    result = {
        "success": True,
        "booking_ref": booking["booking_ref"],
        "confirmed_for": booking["label"],
        "service": booking["service"],
        "calendar_invite": str(ics_path.name),
        "message": (
            f"Booked {booking['service']} for {booking['label']}. "
            f"Reference {booking['booking_ref']}."
        ),
    }
    _trace("book_appointment    ->", result)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Post-call webhook: structured summary + Inference-generated follow-up
# ---------------------------------------------------------------------------

@app.post("/webhooks/call-summary")
async def call_summary(request: Request) -> JSONResponse:
    """
    Telnyx posts a conversation insight / summary after the call ends (configure
    this URL under the assistant's Insights / post-conversation settings). We
    store the record, ask Telnyx Inference to draft a personalised follow-up,
    and optionally text it to the caller.
    """
    body = await request.json()
    data = body.get("data", body)
    payload = data.get("payload", data) if isinstance(data, dict) else {}

    conversation_id = payload.get("conversation_id") or payload.get("id")
    caller_phone = _caller_phone(payload, payload)
    summary = (
        payload.get("summary")
        or payload.get("conversation_summary")
        or payload.get("transcript")
        or ""
    )
    structured = payload.get("insights") or payload.get("extracted_data") or payload

    context = summary if summary else str(structured)[:2000]

    follow_up = ""
    try:
        follow_up = inference.generate_follow_up(context)
        if SETTINGS.send_sms and caller_phone:
            inference.send_sms(caller_phone, follow_up)
    except Exception as exc:  # noqa: BLE001 - never fail the webhook on a follow-up error
        follow_up = f"[follow-up generation skipped: {exc}]"

    crm.add_summary(
        conversation_id=conversation_id,
        caller_phone=caller_phone,
        summary=summary or None,
        structured=structured if isinstance(structured, dict) else None,
        follow_up=follow_up,
    )
    _trace("call_summary        ->", {"conversation_id": conversation_id, "follow_up": follow_up})
    return JSONResponse({"received": True, "follow_up": follow_up})


# ---------------------------------------------------------------------------
# CRM dashboard
# ---------------------------------------------------------------------------

def _fmt(iso: str) -> str:
    try:
        return datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S").strftime("%a %b %d, %I:%M %p")
    except (ValueError, TypeError):
        return iso or ""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    appts = crm.list_appointments()
    summaries = crm.list_summaries()

    appt_rows = "".join(
        f"<tr><td><code>{html.escape(a['booking_ref'])}</code></td>"
        f"<td>{html.escape(a['caller_name'])}</td>"
        f"<td>{html.escape(a['caller_phone'] or '')}</td>"
        f"<td>{html.escape(a['service'])}</td>"
        f"<td>{_fmt(a['start_iso'])}</td>"
        f"<td><span class='pill'>{html.escape(a['status'])}</span></td></tr>"
        for a in appts
    ) or "<tr><td colspan='6' class='empty'>No appointments yet. Call the agent to book one.</td></tr>"

    summary_cards = "".join(
        f"<div class='card'>"
        f"<div class='muted'>{_fmt(s['created_at'])} &middot; {html.escape(s['caller_phone'] or 'unknown caller')}</div>"
        f"<p>{html.escape(s['summary'] or 'No summary text.')}</p>"
        f"<div class='followup'><strong>Follow-up (Telnyx Inference)</strong><br>{html.escape(s['follow_up'] or '')}</div>"
        f"</div>"
        for s in summaries
    ) or "<div class='empty'>No call summaries yet.</div>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(SETTINGS.business_name)} · Front Desk CRM</title>
<style>
  :root {{ --bg:#0F172A; --ink:#1E293B; --accent:#4F46E5; --line:#E5E7EB; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink); background:#F8FAFC; }}
  header {{ background:linear-gradient(135deg,#0F172A,#1E1B4B); color:#fff; padding:28px 32px; }}
  header h1 {{ margin:0 0 4px; font-size:22px; }}
  header p {{ margin:0; opacity:.75; font-size:14px; }}
  main {{ max-width:1000px; margin:0 auto; padding:28px 32px; }}
  h2 {{ font-size:15px; text-transform:uppercase; letter-spacing:.05em; color:#64748B; margin:28px 0 12px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  th,td {{ text-align:left; padding:11px 14px; font-size:14px; border-bottom:1px solid var(--line); }}
  th {{ background:#F1F5F9; color:#475569; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  tr:last-child td {{ border-bottom:none; }}
  code {{ color:var(--accent); font-weight:600; }}
  .pill {{ background:#DCFCE7; color:#166534; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; }}
  .empty {{ color:#94A3B8; text-align:center; padding:24px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:12px; }}
  .muted {{ color:#64748B; font-size:12px; margin-bottom:6px; }}
  .card p {{ margin:0 0 10px; font-size:14px; line-height:1.5; }}
  .followup {{ background:#EEF2FF; color:#3730A3; padding:10px 12px; border-radius:8px; font-size:13px; line-height:1.5; }}
</style></head>
<body>
  <header>
    <h1>{html.escape(SETTINGS.business_name)} · Front Desk CRM</h1>
    <p>Appointments and post-call summaries booked by the Telnyx voice agent</p>
  </header>
  <main>
    <h2>Appointments ({len(appts)})</h2>
    <table>
      <tr><th>Ref</th><th>Name</th><th>Phone</th><th>Service</th><th>When</th><th>Status</th></tr>
      {appt_rows}
    </table>
    <h2>Call summaries ({len(summaries)})</h2>
    {summary_cards}
  </main>
</body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=SETTINGS.host, port=SETTINGS.port, reload=True)
