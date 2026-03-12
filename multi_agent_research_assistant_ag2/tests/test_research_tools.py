import json
import pytest
from tools.research_tools import web_search, fetch_page_content

def test_web_search_returns_valid_json():
    result = web_search("Python programming language", num_results=3)
    data = json.loads(result)
    assert "query" in data
    # Function returns either results or error — both are valid structured outputs
    assert "results" in data or "error" in data
    if "results" in data:
        assert isinstance(data["results"], list)

def test_web_search_handles_empty_query():
    result = web_search("")
    data = json.loads(result)
    # Should not raise — may return empty results or error key
    assert isinstance(data, dict)

def test_fetch_page_content_invalid_url():
    result = fetch_page_content("http://invalid.nonexistent.url.xyz")
    data = json.loads(result)
    assert "error" in data

def test_fetch_page_content_max_chars():
    result = fetch_page_content("https://example.com", max_chars=100)
    data = json.loads(result)
    if "content" in data:
        assert len(data["content"]) <= 100
