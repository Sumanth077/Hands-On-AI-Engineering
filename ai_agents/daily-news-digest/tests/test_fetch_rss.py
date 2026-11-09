"""Provider-free tests for RSS date conversion."""

from __future__ import annotations

import calendar
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


def load_fetch_rss_module():
    project_dir = Path(__file__).resolve().parents[1]
    requests = ModuleType("requests")
    feedparser = ModuleType("feedparser")
    spec = importlib.util.spec_from_file_location(
        "daily_news_digest_fetch_rss_under_test",
        project_dir / "scripts" / "fetch_rss.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/fetch_rss.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"requests": requests, "feedparser": feedparser}):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class ParseEntryDateTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fetch_rss = load_fetch_rss_module()

    def test_feedparser_utc_dates_ignore_host_timezone(self) -> None:
        parsed = time.struct_time((2026, 11, 10, 0, 0, 0, 1, 314, 0))
        expected = datetime(2026, 11, 10, tzinfo=timezone.utc)

        def singapore_local_mktime(value) -> float:
            return calendar.timegm(value) - 8 * 60 * 60

        for field in ("published_parsed", "updated_parsed"):
            entry = SimpleNamespace(
                published_parsed=parsed if field == "published_parsed" else None,
                updated_parsed=parsed if field == "updated_parsed" else None,
            )
            with self.subTest(field=field), patch(
                "time.mktime", side_effect=singapore_local_mktime
            ):
                self.assertEqual(self.fetch_rss._parse_entry_date(entry), expected)
