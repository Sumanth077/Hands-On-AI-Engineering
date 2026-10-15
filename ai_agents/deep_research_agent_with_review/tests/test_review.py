from research_agent.review import parse_review_response


def test_parse_review_response_reads_answer_values() -> None:
    result = parse_review_response(
        {
            "answers": {
                "evidence_coverage": {"probability": 0.92},
                "claim_grounding": {"probability": 0.21},
                "action": {"choice": "research_again"},
            }
        }
    )

    assert result.evidence_coverage is True
    assert result.claim_grounding is False
    assert result.action == "research_again"
    assert result.needs_revision is True

