from legal_intake.models import IntakeDecision


def test_intake_decision_accepts_a_single_follow_up_question():
    decision = IntakeDecision.model_validate(
        {
            "facts": {"narrative": "A payment was missed."},
            "next_question": "When did the payment become overdue?",
            "status": "collecting",
            "reason": "The timeline is missing.",
        }
    )

    assert decision.status == "collecting"
    assert "When" in decision.next_question
