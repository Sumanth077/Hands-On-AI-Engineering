import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeDocument:
    def __init__(self, text, source):
        self.page_content = text
        self.metadata = {"source": source}


class FakeLoader:
    def __init__(self, documents):
        self.documents = documents

    def load(self):
        return self.documents


class FakeSplitter:
    def split_documents(self, documents):
        return documents


def import_rag_module():
    embedder = types.ModuleType("embedder")

    class FakeGeminiEmbedder:
        def __init__(self, api_key):
            pass

        def embed_batch(self, texts):
            return [[0.1, 0.2] for _ in texts]

    embedder.GeminiEmbedder = FakeGeminiEmbedder

    community = types.ModuleType("langchain_community")
    community.__path__ = []
    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = lambda path: None
    loaders.TextLoader = lambda path, encoding: None
    splitters = types.ModuleType("langchain_text_splitters")
    splitters.RecursiveCharacterTextSplitter = lambda **kwargs: None

    fake_modules = {
        "embedder": embedder,
        "langchain_community": community,
        "langchain_community.document_loaders": loaders,
        "langchain_text_splitters": splitters,
    }
    return fake_modules


class IngestReplacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(cls.project_dir))
        import google

        cls.genai_patch = patch.object(
            google,
            "genai",
            types.SimpleNamespace(Client=lambda **kwargs: object()),
            create=True,
        )
        cls.genai_patch.start()
        cls.modules_patch = patch.dict(sys.modules, import_rag_module())
        cls.modules_patch.start()
        sys.modules.pop("rag", None)
        cls.rag = importlib.import_module("rag")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("rag", None)
        cls.modules_patch.stop()
        cls.genai_patch.stop()
        sys.path.remove(str(cls.project_dir))

    def make_pipeline(self):
        pipeline = self.rag.HyDERAG(google_api_key="test-key")
        pipeline._collection.add(
            embeddings=[[0.5, 0.5]],
            documents=["existing document"],
            metadatas=[{"source": "existing.txt"}],
            ids=["existing-id"],
        )
        pipeline.doc_info = {"filename": "existing.txt", "total_chunks": 1}
        return pipeline

    def patch_loader_and_splitter(self, documents):
        self.loader_patch = patch.object(
            self.rag, "TextLoader", return_value=FakeLoader(documents)
        )
        self.splitter_patch = patch.object(
            self.rag,
            "RecursiveCharacterTextSplitter",
            return_value=FakeSplitter(),
        )
        self.loader_patch.start()
        self.splitter_patch.start()
        self.addCleanup(self.loader_patch.stop)
        self.addCleanup(self.splitter_patch.stop)

    def test_failed_replacement_keeps_previous_collection_and_metadata(self):
        pipeline = self.make_pipeline()
        previous_collection = pipeline._collection
        previous_name = pipeline._collection_name
        previous_info = pipeline.doc_info.copy()
        self.patch_loader_and_splitter([FakeDocument("replacement", "new.txt")])
        pipeline.embedder.embed_batch = unittest.mock.Mock(
            side_effect=RuntimeError("embedding unavailable")
        )

        with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
            pipeline.ingest("new.txt")

        self.assertIs(pipeline._collection, previous_collection)
        self.assertEqual(pipeline._collection_name, previous_name)
        self.assertEqual(pipeline._collection.get()["documents"], ["existing document"])
        self.assertEqual(pipeline.doc_info, previous_info)
        collection_names = {
            getattr(collection, "name", collection)
            for collection in pipeline._chroma.list_collections()
        }
        self.assertEqual(collection_names, {previous_name})

    def test_successful_replacement_activates_new_collection_and_clear_resets_it(self):
        pipeline = self.make_pipeline()
        previous_name = pipeline._collection_name
        self.patch_loader_and_splitter([FakeDocument("replacement", "new.txt")])

        info = pipeline.ingest("new.txt")

        self.assertEqual(info["filename"], "new.txt")
        self.assertEqual(pipeline._collection.get()["documents"], ["replacement"])
        self.assertNotEqual(pipeline._collection_name, previous_name)
        collection_names = {
            getattr(collection, "name", collection)
            for collection in pipeline._chroma.list_collections()
        }
        self.assertNotIn(previous_name, collection_names)

        pipeline.clear()

        self.assertEqual(pipeline._collection_name, pipeline.COLLECTION)
        self.assertEqual(pipeline._collection.count(), 0)
        self.assertIsNone(pipeline.doc_info)


if __name__ == "__main__":
    unittest.main()
