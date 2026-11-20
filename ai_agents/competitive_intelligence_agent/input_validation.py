"""Validation helpers for the Competitive Intelligence Agent form."""


def missing_required_inputs(my_company, competitor, pain_point, goal):
    """Return labels for required values that are empty or whitespace-only."""
    required = (
        ("Your Company Name", my_company),
        ("Competitor Name", competitor),
        ("Main Pain Point", pain_point),
        ("Strategic Goal", goal),
    )
    return tuple(
        label
        for label, value in required
        if not isinstance(value, str) or not value.strip()
    )
