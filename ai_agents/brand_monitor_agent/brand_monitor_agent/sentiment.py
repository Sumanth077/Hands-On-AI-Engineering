"""Parse the explicit sentiment label from a brand report."""

import re


_SENTIMENT_LINE = re.compile(r"Sentiment:\s*(Positive|Neutral|Negative|Mixed)", re.IGNORECASE)


def detect_sentiment(report: str) -> str:
    """Return the first-line sentiment label, or ``Unrated`` when absent/invalid."""
    if not report:
        return "Unrated"

    first_line = report.splitlines()[0].strip()
    match = _SENTIMENT_LINE.fullmatch(first_line)
    return match.group(1).title() if match else "Unrated"
