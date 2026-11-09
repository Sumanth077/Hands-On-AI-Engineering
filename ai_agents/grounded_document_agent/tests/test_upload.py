import unittest
from pathlib import Path

from grounded_agent.upload import parse_uploaded_pdf


class ParseUploadedPdfTests(unittest.TestCase):
    def test_removes_temporary_upload_after_successful_parse(self):
        paths = []

        def parse_pdf(path, *, file_name):
            path = Path(path)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), b"pdf bytes")
            self.assertEqual(file_name, "report.pdf")
            paths.append(path)
            return ["parsed page"]

        result = parse_uploaded_pdf(b"pdf bytes", "report.pdf", parse_pdf)

        self.assertEqual(result, ["parsed page"])
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists())

    def test_removes_temporary_upload_when_parse_fails(self):
        paths = []

        def parse_pdf(path, *, file_name):
            path = Path(path)
            self.assertTrue(path.exists())
            paths.append(path)
            raise RuntimeError("parse failed")

        with self.assertRaisesRegex(RuntimeError, "parse failed"):
            parse_uploaded_pdf(b"pdf bytes", "report.pdf", parse_pdf)

        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
