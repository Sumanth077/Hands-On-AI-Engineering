from research_agent.models import Source
from research_agent.validation import validate_citations


def test_validate_citations_accepts_available_numbers() -> None:
    sources = [Source(1, "Source", "https://example.com", "Evidence")]
    assert validate_citations("Supported claim [1].", sources) == []


def test_validate_citations_rejects_unknown_or_missing_citations() -> None:
    sources = [Source(1, "Source", "https://example.com", "Evidence")]
    assert validate_citations("Claim [2].", sources) == [
        "The report cites unavailable source numbers: [2]."
    ]
    assert validate_citations("Uncited claim.", sources) == [
        "The report did not include any numbered citations."
    ]

