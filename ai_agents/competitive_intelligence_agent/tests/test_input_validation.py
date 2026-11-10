import unittest
from unittest.mock import patch

from main import run_analysis


class RunAnalysisValidationTests(unittest.TestCase):
    def test_reports_all_fields_when_form_is_empty(self):
        with patch("main._run_valid_analysis") as run_valid_analysis:
            logs, output = run_analysis("", "", "", "")

        self.assertEqual(
            output,
            "Please complete these required fields before generating a battlecard: "
            "Your Company Name, Competitor Name, Main Pain Point (What do you lose deals over?), "
            "Strategic Goal (e.g. Win enterprise deals).",
        )
        self.assertEqual(logs, "")
        run_valid_analysis.assert_not_called()

    def test_reports_only_empty_and_whitespace_fields(self):
        with patch("main._run_valid_analysis") as run_valid_analysis:
            _, output = run_analysis("Acme", "  ", "support costs", None)

        self.assertIn("Competitor Name", output)
        self.assertIn("Strategic Goal (e.g. Win enterprise deals)", output)
        self.assertNotIn("Your Company Name", output)
        self.assertNotIn("Main Pain Point", output)
        run_valid_analysis.assert_not_called()

    def test_accepts_non_whitespace_values(self):
        expected = ("analysis log", "battlecard")
        with patch("main._run_valid_analysis", return_value=expected) as run_valid_analysis:
            result = run_analysis(" Acme ", "Rival", "Support costs", "Win deals")

        self.assertEqual(result, expected)
        run_valid_analysis.assert_called_once_with(" Acme ", "Rival", "Support costs", "Win deals")


if __name__ == "__main__":
    unittest.main()
