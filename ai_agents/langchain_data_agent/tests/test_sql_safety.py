import unittest

from sql_safety import _validate_read_only


class ReadOnlySQLValidationTests(unittest.TestCase):
    def test_accepts_select(self):
        self.assertEqual(_validate_read_only("SELECT 1;"), "SELECT 1")

    def test_accepts_common_table_expression(self):
        query = "WITH counts AS (SELECT 1 AS value) SELECT value FROM counts"
        self.assertEqual(_validate_read_only(query), query)

    def test_rejects_mutating_pragma(self):
        with self.assertRaisesRegex(ValueError, "Only read-only SELECT queries"):
            _validate_read_only("PRAGMA user_version = 735")

    def test_rejects_explain_statement_outside_documented_query_forms(self):
        with self.assertRaisesRegex(ValueError, "Only read-only SELECT queries"):
            _validate_read_only("EXPLAIN SELECT 1")


if __name__ == "__main__":
    unittest.main()
