import json
import unittest

from script_json import serialize_inline_script_data


class InlineScriptDataTests(unittest.TestCase):
    def test_escapes_script_terminator_and_round_trips_original_text(self) -> None:
        value = "</script><script>window.pwned=1</script>"

        serialized = serialize_inline_script_data(value)

        self.assertNotIn("<", serialized)
        self.assertNotIn("</script>", serialized.lower())
        self.assertEqual(json.loads(serialized), value)

    def test_preserves_normal_text(self) -> None:
        value = "Senior Technical Writer — Research & Editorial"

        self.assertEqual(json.loads(serialize_inline_script_data(value)), value)


if __name__ == "__main__":
    unittest.main()
