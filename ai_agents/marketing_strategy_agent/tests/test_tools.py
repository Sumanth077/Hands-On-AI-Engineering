import os
import unittest
from unittest.mock import patch

import httpx

from marketing_strategy_agent.tools import search_web


class SearchWebTests(unittest.TestCase):
    def test_missing_api_key_stops_before_search(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": ""}):
            with patch("marketing_strategy_agent.tools.httpx.post") as post:
                with self.assertRaisesRegex(RuntimeError, "SERPER_API_KEY not set"):
                    search_web("market research")

        post.assert_not_called()

    def test_request_failure_propagates(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test-key"}):
            with patch(
                "marketing_strategy_agent.tools.httpx.post",
                side_effect=httpx.TimeoutException("offline"),
            ):
                with self.assertRaises(httpx.TimeoutException):
                    search_web("market research")


if __name__ == "__main__":
    unittest.main()
