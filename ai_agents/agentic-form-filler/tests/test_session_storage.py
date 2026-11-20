import tempfile
import unittest
from pathlib import Path

from session_storage import session_file_path


class SessionFilePathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.temp_dir.name) / "sessions"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accepts_safe_session_ids(self):
        self.assertEqual(session_file_path(self.session_dir, "a1b2c3d4"), self.session_dir / "a1b2c3d4.json")
        self.assertEqual(session_file_path(self.session_dir, "legacy_session-2"), self.session_dir / "legacy_session-2.json")

    def test_rejects_traversal_and_invalid_ids(self):
        for session_id in ("../probe", r"..\probe", "nested/path", "", None):
            with self.subTest(session_id=session_id):
                self.assertIsNone(session_file_path(self.session_dir, session_id))


if __name__ == "__main__":
    unittest.main()
