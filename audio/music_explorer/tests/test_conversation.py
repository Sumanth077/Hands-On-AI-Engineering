import unittest
from types import SimpleNamespace

from conversation import build_contents, reset_audio_context


class FakePart:
    @staticmethod
    def from_text(*, text):
        return {"kind": "text", "text": text}

    @staticmethod
    def from_uri(*, file_uri, mime_type):
        return {"kind": "file", "uri": file_uri, "mime_type": mime_type}


class FakeContent:
    def __init__(self, *, role, parts):
        self.role = role
        self.parts = parts


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.types = SimpleNamespace(Content=FakeContent, Part=FakePart)
        self.audio = SimpleNamespace(uri="files/audio-123", mime_type="audio/wav")

    def test_first_turn_includes_audio(self):
        contents = build_contents(self.types, self.audio, [], "Describe this audio")

        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0]["text"], "Describe this audio")
        self.assertEqual(contents[0].parts[1]["uri"], "files/audio-123")

    def test_follow_up_turn_includes_audio_and_text_history(self):
        history = [
            {"role": "user", "content": "Describe this audio"},
            {"role": "assistant", "content": "It is a jazz recording."},
        ]

        contents = build_contents(
            self.types, self.audio, history, "What instruments enter at 2:30?"
        )

        self.assertEqual([item.role for item in contents], ["user", "model", "user"])
        self.assertEqual(contents[1].parts[0]["text"], "It is a jazz recording.")
        self.assertEqual(contents[2].parts[0]["text"], "What instruments enter at 2:30?")
        self.assertEqual(contents[2].parts[1]["uri"], "files/audio-123")

    def test_audio_source_change_clears_chat_and_file_reference(self):
        self.assertEqual(reset_audio_context(), ([], ""))


if __name__ == "__main__":
    unittest.main()
