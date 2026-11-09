import unittest
from datetime import datetime

import pytz

from calendar_dates import today_in_timezone


class TodayInTimezoneTests(unittest.TestCase):
    def test_date_uses_selected_timezone_across_midnight(self):
        now = datetime(2026, 11, 10, 0, 30, tzinfo=pytz.utc)

        self.assertEqual(
            today_in_timezone("America/Los_Angeles", now),
            "Monday, November 09, 2026",
        )
        self.assertEqual(
            today_in_timezone("Asia/Tokyo", now),
            "Tuesday, November 10, 2026",
        )


if __name__ == "__main__":
    unittest.main()
