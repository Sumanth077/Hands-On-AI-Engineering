import sqlite3
import tempfile
import unittest
from pathlib import Path

from db_tools import create_database_functions


class DatabaseFunctionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp_dir.name)
        self.db_a = self.root / "a.sqlite"
        self.db_b = self.root / "b.sqlite"
        self._create_database(self.db_a, "private_a", "alpha-secret")
        self._create_database(self.db_b, "private_b", "beta-secret")

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _create_database(path, table, secret):
        conn = sqlite3.connect(path)
        try:
            conn.execute(f"CREATE TABLE {table} (value TEXT NOT NULL)")
            conn.execute(f"INSERT INTO {table} (value) VALUES (?)", (secret,))
            conn.commit()
        finally:
            conn.close()

    def test_each_tool_pair_stays_bound_to_its_database_when_interleaved(self):
        schema_a, query_a = create_database_functions(self.db_a)
        schema_b, query_b = create_database_functions(self.db_b)

        self.assertIn("private_a", schema_a())
        self.assertIn("private_b", schema_b())
        self.assertIn("alpha-secret", query_a("SELECT value FROM private_a"))
        self.assertIn("beta-secret", query_b("SELECT value FROM private_b"))


if __name__ == "__main__":
    unittest.main()
