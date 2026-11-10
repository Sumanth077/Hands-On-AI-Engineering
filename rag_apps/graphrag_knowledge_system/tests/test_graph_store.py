import tempfile
import unittest
from pathlib import Path

from src.graph_store import GraphStore


class GraphStoreLoadTests(unittest.TestCase):
    def test_invalid_existing_graph_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph_store.json"
            original_contents = '{"nodes": ['
            graph_path.write_text(original_contents, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                GraphStore(graph_path)

            self.assertEqual(
                graph_path.read_text(encoding="utf-8"),
                original_contents,
            )


if __name__ == "__main__":
    unittest.main()
