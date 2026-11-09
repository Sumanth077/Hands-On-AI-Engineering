import unittest

from verdict import reconcile_scores


DEBATER_A = "Debater A"
DEBATER_B = "Debater B"


class ReconcileScoresTests(unittest.TestCase):
    def test_winner_follows_scores_when_model_declares_the_other_debater(self):
        parsed = {
            "arguments": [
                {"speaker": DEBATER_A, "logic": 5, "evidence": 5, "persuasiveness": 5},
                {"speaker": DEBATER_B, "logic": 8, "evidence": 9, "persuasiveness": 8},
            ],
            "totals": {DEBATER_A: 99, DEBATER_B: 1},
            "winner": DEBATER_A,
        }

        result = reconcile_scores(parsed, DEBATER_A, DEBATER_B)

        self.assertEqual(result["totals"], {DEBATER_A: 15, DEBATER_B: 25})
        self.assertEqual(result["winner"], DEBATER_B)

    def test_equal_scores_override_a_declared_winner_with_tie(self):
        parsed = {
            "arguments": [
                {"speaker": DEBATER_A, "logic": 5, "evidence": 5, "persuasiveness": 5},
                {"speaker": DEBATER_B, "logic": 5, "evidence": 5, "persuasiveness": 5},
            ],
            "winner": DEBATER_A,
        }

        result = reconcile_scores(parsed, DEBATER_A, DEBATER_B)

        self.assertEqual(result["winner"], "Tie")

    def test_no_scores_leave_existing_verdict_unchanged(self):
        parsed = {"arguments": [], "winner": DEBATER_B}

        result = reconcile_scores(parsed, DEBATER_A, DEBATER_B)

        self.assertEqual(result["winner"], DEBATER_B)
        self.assertEqual(result["totals"], {DEBATER_A: 0, DEBATER_B: 0})


if __name__ == "__main__":
    unittest.main()
