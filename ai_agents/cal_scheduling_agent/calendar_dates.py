"""Date helpers for calendar scheduling prompts."""

from datetime import datetime

import pytz


def today_in_timezone(timezone_name: str, now: datetime | None = None) -> str:
    """Format today's date in the selected timezone.

    ``now`` is injectable so the date boundary can be tested without waiting
    for the host clock or depending on its local timezone.
    """
    selected_timezone = pytz.timezone(timezone_name)
    instant = now or datetime.now(pytz.utc)
    if instant.tzinfo is None:
        instant = pytz.utc.localize(instant)
    return instant.astimezone(selected_timezone).strftime("%A, %B %d, %Y")
