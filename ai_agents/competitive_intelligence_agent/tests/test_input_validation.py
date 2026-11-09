import unittest

from input_validation import missing_required_inputs


class MissingRequiredInputsTests(unittest.TestCase):
    def test_reports_empty_and_whitespace_only_fields(self):
        missing = missing_required_inputs("Acme", "  ", None, "Win renewals")

        self.assertEqual(missing, ("Competitor Name", "Main Pain Point"))

    def test_accepts_complete_context(self):
        missing = missing_required_inputs(
            "Acme", "Rival Co", "slow onboarding", "Win enterprise deals"
        )

        self.assertEqual(missing, ())

    def test_reports_every_field_when_form_is_untouched(self):
        missing = missing_required_inputs("", "", "", "")

        self.assertEqual(
            missing,
            (
                "Your Company Name",
                "Competitor Name",
                "Main Pain Point",
                "Strategic Goal",
            ),
        )


if __name__ == "__main__":
    unittest.main()
