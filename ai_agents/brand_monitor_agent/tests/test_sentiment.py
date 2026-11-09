import importlib.util
import unittest
from pathlib import Path

SENTIMENT_PATH = Path(__file__).parents[1] / "brand_monitor_agent" / "sentiment.py"
SPEC = importlib.util.spec_from_file_location("brand_monitor_sentiment", SENTIMENT_PATH)
assert SPEC is not None and SPEC.loader is not None
SENTIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SENTIMENT)
detect_sentiment = SENTIMENT.detect_sentiment


class DetectSentimentTests(unittest.TestCase):
    def test_uses_explicit_label_instead_of_conflicting_body_keywords(self):
        report = (
            "Sentiment: Negative\n"
            "The report describes strong criticism and several complaints; "
            "positive examples were rare."
        )

        self.assertEqual(detect_sentiment(report), "Negative")

    def test_normalizes_label_case(self):
        self.assertEqual(detect_sentiment("Sentiment: mixed\nBalanced evidence."), "Mixed")

    def test_missing_or_invalid_label_is_unrated(self):
        self.assertEqual(detect_sentiment("A negative report with positive examples."), "Unrated")
        self.assertEqual(detect_sentiment("Sentiment: Mostly positive"), "Unrated")
        self.assertEqual(detect_sentiment(""), "Unrated")


if __name__ == "__main__":
    unittest.main()
