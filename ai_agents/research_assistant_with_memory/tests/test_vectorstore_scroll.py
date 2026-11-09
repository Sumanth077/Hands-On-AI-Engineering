"""Focused tests for paging through VectorAI DB document points."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_vectorstore_module():
    """Load vectorstore with SDK/config stubs so tests need no services."""
    actian = types.ModuleType("actian_vectorai")
    actian.Distance = types.SimpleNamespace(Cosine="cosine")
    actian.PointStruct = object
    actian.VectorAIClient = object
    actian.VectorParams = object

    config = types.ModuleType("research_assistant.config")
    config.COLLECTIONS = ["research_documents", "research_memory"]
    config.DOCUMENTS_COLLECTION = "research_documents"
    config.EMBEDDING_DIM = 384
    config.MEMORY_COLLECTION = "research_memory"
    config.VECTORAI_ACCESS_TOKEN = None
    config.VECTORAI_URL = "localhost:6574"

    module_path = Path(__file__).resolve().parents[1] / "research_assistant" / "vectorstore.py"
    module_name = "research_assistant.vectorstore_scroll_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, {
        "actian_vectorai": actian,
        "research_assistant.config": config,
    }):
        spec.loader.exec_module(module)

    return module


vectorstore = load_vectorstore_module()


class FakePoints:
    def __init__(self, pages):
        self.pages = list(pages)
        self.offsets = []

    def count(self, collection):
        return sum(len(points) for points, _ in self.pages)

    def scroll(self, collection, *, limit, offset=None):
        self.offsets.append(offset)
        return self.pages.pop(0)


class ScrollDocumentsTests(unittest.TestCase):
    def test_reads_each_page_using_the_returned_cursor(self):
        first = object()
        second = object()
        points = FakePoints([([first], "page-2"), ([second], None)])

        result = vectorstore.scroll_documents(types.SimpleNamespace(points=points), limit=1)

        self.assertEqual(result, [first, second])
        self.assertEqual(points.offsets, [None, "page-2"])

    def test_empty_collection_does_not_scroll(self):
        points = FakePoints([])

        result = vectorstore.scroll_documents(types.SimpleNamespace(points=points))

        self.assertEqual(result, [])
        self.assertEqual(points.offsets, [])

    def test_repeated_cursor_fails_instead_of_looping_or_truncating(self):
        points = FakePoints([([object()], "page-2"), ([object()], "page-2")])

        with self.assertRaisesRegex(RuntimeError, "repeated scroll cursor"):
            vectorstore.scroll_documents(types.SimpleNamespace(points=points), limit=1)


if __name__ == "__main__":
    unittest.main()
