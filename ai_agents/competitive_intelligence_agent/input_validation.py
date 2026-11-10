"""Validation helpers for the competitive intelligence form."""


def missing_required_fields(my_company, competitor, pain_point, goal):
    """Return labels for required battlecard fields that are empty or whitespace-only."""
    fields = (
        ("Your Company Name", my_company),
        ("Competitor Name", competitor),
        ("Main Pain Point (What do you lose deals over?)", pain_point),
        ("Strategic Goal (e.g. Win enterprise deals)", goal),
    )
    return [
        label
        for label, value in fields
        if not isinstance(value, str) or not value.strip()
    ]
