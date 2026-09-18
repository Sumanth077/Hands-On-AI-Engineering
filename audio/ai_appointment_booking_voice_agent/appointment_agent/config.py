"""
Runtime configuration, resolved from environment variables (see .env.example).

Nothing is hardcoded: the business profile, model, and scheduling rules all come
from the environment so the same code runs for any front desk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """All runtime settings, resolved from the environment with sane defaults."""

    # Telnyx (cloud). The API key is used for Inference and optional SMS.
    telnyx_api_key: str = os.getenv("TELNYX_API_KEY", "")
    inference_base_url: str = os.getenv("INFERENCE_BASE_URL", "https://api.telnyx.com/v2/ai/openai")
    inference_model: str = os.getenv("INFERENCE_MODEL", "moonshotai/Kimi-K2.6")

    # Business profile, injected into the agent prompt at call start.
    business_name: str = os.getenv("BUSINESS_NAME", "Brightsmile Dental")
    business_hours: str = os.getenv("BUSINESS_HOURS", "Mon-Fri, 9:00 AM to 5:00 PM")
    business_timezone: str = os.getenv("BUSINESS_TIMEZONE", "America/New_York")
    support_email: str = os.getenv("SUPPORT_EMAIL", "front-desk@example.com")
    services: list[str] = field(
        default_factory=lambda: _csv(
            os.getenv("BUSINESS_SERVICES", "Checkup, Cleaning, Whitening, Emergency visit")
        )
    )

    # Scheduling rules.
    slot_length_min: int = int(os.getenv("SLOT_LENGTH_MIN", "30"))
    booking_horizon_days: int = int(os.getenv("BOOKING_HORIZON_DAYS", "14"))
    open_hour: int = int(os.getenv("OPEN_HOUR", "9"))
    close_hour: int = int(os.getenv("CLOSE_HOUR", "17"))

    # Optional SMS follow-up. Off by default so the demo sends nothing.
    send_sms: bool = os.getenv("SEND_SMS", "false").lower() == "true"
    sms_from: str = os.getenv("SMS_FROM", "")

    # Storage and server.
    db_path: Path = Path(os.getenv("DB_PATH", ".data/appointments.db"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


SETTINGS = Settings()


def require_api_key() -> str:
    """Return the Telnyx API key or raise a clear error if it is missing."""
    if not SETTINGS.telnyx_api_key:
        raise RuntimeError(
            "TELNYX_API_KEY is not set. Copy .env.example to .env and add your key "
            "from portal.telnyx.com -> API Keys."
        )
    return SETTINGS.telnyx_api_key
