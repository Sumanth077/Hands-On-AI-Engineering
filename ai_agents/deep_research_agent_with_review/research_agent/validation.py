from __future__ import annotations

import re

from .models import Source

CITATION_PATTERN = re.compile(r"(?<!\[)\[(\d+)\](?!\])")


def validate_citations(answer: str, sources: list[Source]) -> list[str]:
    """Catch malformed or out-of-range numeric citations before showing a report."""
    cited = {int(number) for number in CITATION_PATTERN.findall(answer)}
    available = {source.number for source in sources}
    issues: list[str] = []

    if not cited:
        issues.append("The report did not include any numbered citations.")

    unknown = sorted(cited - available)
    if unknown:
        issues.append(f"The report cites unavailable source numbers: {unknown}.")

    return issues

