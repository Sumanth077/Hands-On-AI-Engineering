import json
from unittest.mock import patch, MagicMock
import pytest
from tools.research_tools import web_search, fetch_page_content


def _mock_urlopen(response_body: bytes):
    """Create a mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_web_search_returns_valid_json():
    body = json.dumps({"RelatedTopics": [{"Text": "Python info", "FirstURL": "https://python.org"}],
                        "AbstractText": "Python is a language", "Heading": "Python",
                        "AbstractURL": "https://python.org"}).encode()
    with patch("tools.research_tools.urllib.request.urlopen", return_value=_mock_urlopen(body)):
        result = web_search("Python programming language", num_results=3)
    data = json.loads(result)
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 1


def test_web_search_handles_empty_query():
    body = json.dumps({"RelatedTopics": []}).encode()
    with patch("tools.research_tools.urllib.request.urlopen", return_value=_mock_urlopen(body)):
        result = web_search("")
    data = json.loads(result)
    assert isinstance(data, dict)


def test_fetch_page_content_rejects_file_url():
    result = fetch_page_content("file:///etc/passwd")
    data = json.loads(result)
    assert "error" in data
    assert "Unsupported URL scheme" in data["error"]


def test_fetch_page_content_strips_html():
    body = b"<html><body><h1>Title</h1><p>Content here</p></body></html>"
    with patch("tools.research_tools.urllib.request.urlopen", return_value=_mock_urlopen(body)):
        result = fetch_page_content("https://example.com", max_chars=100)
    data = json.loads(result)
    assert "content" in data
    assert "<" not in data["content"]
    assert len(data["content"]) <= 100


def test_fetch_page_content_handles_error():
    with patch("tools.research_tools.urllib.request.urlopen", side_effect=ConnectionError("timeout")):
        result = fetch_page_content("https://unreachable.test")
    data = json.loads(result)
    assert "error" in data
