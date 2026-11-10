from unittest import TestCase
from unittest.mock import Mock, patch

from rag import MultimodalRAG


class ApiKeyRefreshTests(TestCase):
    def setUp(self):
        self.rag = object.__new__(MultimodalRAG)
        self.rag.api_key = "old-key"
        self.rag.client = Mock(name="old-client")
        self.rag.embeddings = Mock()
        self.rag._collection = Mock(name="collection")
        self.rag.sources = [Mock(name="source")]

    def test_updated_key_replaces_gemini_client_and_preserves_index(self):
        new_client = Mock(name="new-client")
        collection = self.rag._collection
        sources = self.rag.sources

        with patch("rag.genai.Client", return_value=new_client) as client_factory:
            self.rag.update_api_key("new-key")

        client_factory.assert_called_once_with(api_key="new-key")
        self.assertIs(self.rag.client, new_client)
        self.assertIs(self.rag.embeddings.client, new_client)
        self.assertEqual(self.rag.api_key, "new-key")
        self.assertIs(self.rag._collection, collection)
        self.assertIs(self.rag.sources, sources)

    def test_unchanged_key_keeps_existing_client(self):
        client = self.rag.client

        with patch("rag.genai.Client") as client_factory:
            self.rag.update_api_key("old-key")

        client_factory.assert_not_called()
        self.assertIs(self.rag.client, client)
