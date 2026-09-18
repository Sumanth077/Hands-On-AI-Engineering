"""
Generate an iCalendar (.ics) invite for a booking.

Hand-rolled so there is no extra dependency. Times are written as local
wall-clock (floating) time, which every calendar app reads as the viewer's
local time. Good enough for a demo invite; add a VTIMEZONE block if you need
strict timezone handling.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from appointment_agent.config import SETTINGS

_ICS_DIR = SETTINGS.db_path.parent / "invites"


def _ics_stamp(iso: str) -> str:
    return datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S").strftime("%Y%m%dT%H%M%S")


def build_invite(booking: dict) -> tuple[str, Path]:
    """Return (ics_text, path_on_disk) for a booking dict from scheduling.book()."""
    ref = booking["booking_ref"]
    summary = f"{booking['service']} at {SETTINGS.business_name}"
    description = (
        f"Booking reference {ref}. "
        f"Booked for {booking['name']} ({booking['phone']})."
    )

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Telnyx Appointment Agent//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{ref}@telnyx-appointment-agent",
        f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{_ics_stamp(booking['start'])}",
        f"DTEND:{_ics_stamp(booking['end'])}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{SETTINGS.business_name}",
        "STATUS:CONFIRMED",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    ics_text = "\r\n".join(lines) + "\r\n"

    _ICS_DIR.mkdir(parents=True, exist_ok=True)
    path = _ICS_DIR / f"{ref}.ics"
    path.write_text(ics_text, encoding="utf-8")
    return ics_text, path
