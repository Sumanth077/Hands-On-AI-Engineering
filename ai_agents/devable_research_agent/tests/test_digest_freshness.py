import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_pipeline


class DigestFreshnessTests(unittest.TestCase):
    def validate(self, generated_at):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            digest_path = temp_path / "digest.json"
            digest_path.write_text(
                json.dumps({"items": [{"id": "T001"}], "generated_at": generated_at}),
                encoding="utf-8",
            )
            with (
                patch.object(run_pipeline, "DIGEST_FILE", digest_path),
                patch.object(run_pipeline, "LOG_FILE", temp_path / "pipeline.log"),
            ):
                return run_pipeline._validate_digest_input()

    def test_rejects_future_generated_at(self):
        future = datetime.now(timezone.utc) + timedelta(days=2)

        self.assertFalse(self.validate(future.isoformat()))

    def test_accepts_recent_generated_at_from_previous_lagos_day(self):
        recent = datetime.now(run_pipeline.TZ_LAGOS) - timedelta(hours=23)

        self.assertTrue(self.validate(recent.isoformat()))


if __name__ == "__main__":
    unittest.main()
