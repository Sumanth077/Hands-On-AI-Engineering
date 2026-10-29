from legal_intake.agent import draft_summary
from legal_intake.models import IntakeFacts


def test_draft_summary_includes_known_facts_and_open_questions():
    facts = IntakeFacts(
        matter_type="Housing dispute",
        narrative="The landlord changed the locks.",
        people_or_organisations=["Tenant", "Landlord"],
        missing_information=["Date of the lock change"],
    )

    summary = draft_summary(facts)

    assert "Housing dispute" in summary
    assert "The landlord changed the locks." in summary
    assert "Date of the lock change" in summary
    assert "not legal advice" in summary.lower()
