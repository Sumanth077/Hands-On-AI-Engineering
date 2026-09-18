"""
Slot logic for the scheduling agent.

Slots are generated from the business hours in config, filtered against what is
already booked in the CRM store. Times are treated as local wall-clock time for
the business; that is all a front-desk demo needs, and it keeps the calendar
invite readable. Swap this module for a real calendar API (Google, Outlook,
Cal.com) without touching the webhook layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from appointment_agent import crm
from appointment_agent.config import SETTINGS

_TIME_FMT = "%Y-%m-%dT%H:%M:%S"


def _parse_date(date_str: str) -> datetime:
    """Accept YYYY-MM-DD (preferred) or a full ISO datetime and return midnight of that day."""
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.strptime(date_str[: len(fmt) + 2] if "T" in fmt else date_str, fmt)
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {date_str!r}. Use YYYY-MM-DD.")


def available_slots(date_str: str, service: str | None = None) -> list[dict]:
    """Return the open slots for a given date as a list of {start, end, label}."""
    day = _parse_date(date_str)
    now = datetime.now()

    slots: list[dict] = []
    cursor = day.replace(hour=SETTINGS.open_hour)
    end_of_day = day.replace(hour=SETTINGS.close_hour)
    step = timedelta(minutes=SETTINGS.slot_length_min)

    while cursor + step <= end_of_day:
        start_iso = cursor.strftime(_TIME_FMT)
        end_iso = (cursor + step).strftime(_TIME_FMT)
        is_past = cursor <= now
        if not is_past and not crm.is_slot_taken(start_iso):
            slots.append(
                {
                    "start": start_iso,
                    "end": end_iso,
                    "label": cursor.strftime("%A %B %d, %I:%M %p"),
                }
            )
        cursor += step

    return slots


def book(name: str, phone: str, service: str, start: str) -> dict:
    """Book a slot. `start` is an ISO datetime (YYYY-MM-DDTHH:MM:SS). Idempotent per slot."""
    start_dt = datetime.strptime(start[:19], _TIME_FMT)
    end_dt = start_dt + timedelta(minutes=SETTINGS.slot_length_min)
    start_iso = start_dt.strftime(_TIME_FMT)
    end_iso = end_dt.strftime(_TIME_FMT)

    if crm.is_slot_taken(start_iso):
        raise ValueError("That time was just taken. Offer the caller another slot.")

    booking_ref = f"APT-{uuid.uuid4().hex[:8].upper()}"
    crm.add_appointment(booking_ref, name, phone, service, start_iso, end_iso)

    return {
        "booking_ref": booking_ref,
        "name": name,
        "phone": phone,
        "service": service,
        "start": start_iso,
        "end": end_iso,
        "label": start_dt.strftime("%A %B %d, %I:%M %p"),
    }
