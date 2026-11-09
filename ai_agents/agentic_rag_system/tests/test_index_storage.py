import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from index_storage import replace_directory


class ReplaceDirectoryTests(unittest.TestCase):
    def test_failed_build_keeps_existing_index(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent, "chroma_db")
            target.mkdir()
            (target / "existing.sqlite").write_text("previous index")

            def fail_build(staging_path):
                Path(staging_path, "partial.sqlite").write_text("partial")
                raise ValueError("invalid PDF")

            with self.assertRaisesRegex(ValueError, "invalid PDF"):
                replace_directory(str(target), fail_build)

            self.assertEqual((target / "existing.sqlite").read_text(), "previous index")
            self.assertEqual(list(Path(parent).iterdir()), [target])

    def test_success_replaces_existing_index_and_cleans_backup(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent, "chroma_db")
            target.mkdir()
            (target / "existing.sqlite").write_text("previous index")

            def build(staging_path):
                Path(staging_path, "replacement.sqlite").write_text("new index")

            self.assertIsNone(replace_directory(str(target), build))

            self.assertFalse((target / "existing.sqlite").exists())
            self.assertEqual((target / "replacement.sqlite").read_text(), "new index")
            self.assertEqual(list(Path(parent).iterdir()), [target])

    def test_failed_install_restores_existing_index(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent, "chroma_db")
            target.mkdir()
            (target / "existing.sqlite").write_text("previous index")
            real_replace = os.replace

            def fail_staging_install(source, destination):
                if ".staging-" in str(source) and str(destination) == str(target):
                    raise OSError("simulated install failure")
                return real_replace(source, destination)

            with patch("index_storage.os.replace", side_effect=fail_staging_install):
                with self.assertRaisesRegex(OSError, "simulated install failure"):
                    replace_directory(str(target), lambda _staging_path: None)

            self.assertEqual((target / "existing.sqlite").read_text(), "previous index")
            self.assertEqual(list(Path(parent).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
