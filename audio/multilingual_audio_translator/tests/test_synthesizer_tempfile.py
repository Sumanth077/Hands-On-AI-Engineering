import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "synthesizer.py"


def load_synthesizer(write_audio=None):
    numpy = types.ModuleType("numpy")
    numpy.concatenate = lambda chunks: b"".join(chunks)

    soundfile = types.ModuleType("soundfile")

    def write_audio_file(path, audio, _sample_rate):
        if write_audio is None:
            Path(path).write_bytes(audio)
        else:
            write_audio(path, audio)

    soundfile.write = write_audio_file

    kokoro = types.ModuleType("kokoro")

    class FakeKPipeline:
        def __init__(self, lang_code):
            self.lang_code = lang_code

        def __call__(self, _text, voice):
            self.voice = voice
            yield None, None, b"audio"

    kokoro.KPipeline = FakeKPipeline

    module_spec = importlib.util.spec_from_file_location(
        "synthesizer_under_test", MODULE_PATH
    )
    synthesizer = importlib.util.module_from_spec(module_spec)
    with patch.dict(
        sys.modules,
        {"numpy": numpy, "soundfile": soundfile, "kokoro": kokoro},
    ):
        module_spec.loader.exec_module(synthesizer)
    return synthesizer


class SynthesizeTempfileTests(unittest.TestCase):
    def test_closes_descriptor_and_keeps_output_for_playback(self):
        synthesizer = load_synthesizer()
        created_descriptors = []
        real_mkstemp = tempfile.mkstemp

        def track_mkstemp(*args, **kwargs):
            file_descriptor, path = real_mkstemp(*args, **kwargs)
            created_descriptors.append(file_descriptor)
            return file_descriptor, path

        with patch.object(
            synthesizer.tempfile, "mkstemp", side_effect=track_mkstemp
        ), patch.object(synthesizer.os, "close", wraps=os.close) as close_file:
            output_path = synthesizer.synthesize("Hello", "en")

        try:
            self.assertEqual(len(created_descriptors), 1)
            closed_descriptors = [call.args[0] for call in close_file.call_args_list]
            self.assertIn(created_descriptors[0], closed_descriptors)
            self.assertEqual(Path(output_path).read_bytes(), b"audio")
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_removes_partial_file_when_writing_fails(self):
        def fail_write(_path, _audio):
            raise OSError("write failed")

        synthesizer = load_synthesizer(write_audio=fail_write)
        created_paths = []
        real_mkstemp = tempfile.mkstemp

        def track_mkstemp(*args, **kwargs):
            file_descriptor, path = real_mkstemp(*args, **kwargs)
            created_paths.append(path)
            return file_descriptor, path

        with patch.object(synthesizer.tempfile, "mkstemp", side_effect=track_mkstemp):
            with self.assertRaisesRegex(OSError, "write failed"):
                synthesizer.synthesize("Hello", "en")

        self.assertEqual(len(created_paths), 1)
        self.assertFalse(Path(created_paths[0]).exists())


if __name__ == "__main__":
    unittest.main()
