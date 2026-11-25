"""Display repair must not change citation data or introduce image requests."""

from repo_scout.evidence_display import evidence_preview

URL = "https://example.org/docs/readme.md"


def test_formatting_and_relative_links_are_preserved():
    raw = '# Heading\n**Important**: [guide](guide.md).\n\n```python\nprint("ok")\n```\n'
    result = evidence_preview(raw, raw, URL)
    assert "# Heading" in result and "**Important**" in result
    assert "[guide](<https://example.org/docs/guide.md>)" in result
    assert '```python\nprint("ok")\n```' in result


def test_excerpt_inside_code_block_recovers_language_and_closes_it():
    raw = "# Setup\n```python\nfirst()\nsecond()\nthird()\n```\n## Next\n"
    excerpt = "second()\nthird()"
    result = evidence_preview(excerpt, raw, URL)
    assert result == "```python\nsecond()\nthird()\n```\n"
    assert excerpt == "second()\nthird()" and raw.endswith("## Next\n")


def test_leading_partial_code_then_closing_fence_does_not_swallow_headings():
    raw = "```go\nfirst()\nlast()\n```\n\n## Install\n**Local** use\n"
    excerpt = "last()\n```\n\n## Install\n**Local** use\n"
    assert evidence_preview(excerpt, raw, URL) == "```go\n" + excerpt


def test_clipped_links_are_completed_only_from_original_source():
    raw = "See [the **guide**](https://example.org/a(b)/page) for details."
    for excerpt in [
        "**guide**](https://example.org/a(b)/page) for",
        "See [the **guide**](https://exam",
    ]:
        preview = evidence_preview(excerpt, raw, URL)
        assert "[the **guide**](<https://example.org/a(b)/page>)" in preview
    assert raw == "See [the **guide**](https://example.org/a(b)/page) for details."


def test_reference_link_definition_outside_excerpt_is_resolved():
    raw = "Read [manual][docs].\n\n[docs]: ../manual\n"
    preview = evidence_preview("Read [manual][docs].", raw, URL)
    assert preview == "Read [manual](<https://example.org/manual>)."


def test_images_badges_multiline_and_html_embeds_do_not_load():
    raw = (
        "[![Build](https://images.example/badge.svg)](https://example.org/build)\n"
        "![Logo][logo]\n![inline](\nhttps://images.example/logo.png\n)\n"
        '<img src="https://images.example/pixel">\n'
        "[logo]: https://images.example/logo.svg\n"
    )
    preview = evidence_preview(raw, raw, URL)
    assert "![" not in preview
    assert "<img" not in preview
    assert "images.example" not in preview
    assert "https://example.org/build" in preview


def test_code_images_stay_literal_and_unsafe_links_are_not_clickable():
    raw = "```md\n![literal](https://example.org/image.png)\n```\n[click](javascript:alert(1))\n[local](file:///etc/passwd)\n<img src=x onerror=alert(1)>\n"
    preview = evidence_preview(raw, raw, URL)
    assert "```md\n![literal](https://example.org/image.png)\n```" in preview
    assert "javascript:" not in preview and "file:" not in preview
    assert "<img" not in preview


def test_unclosed_fence_without_source_context_is_closed_for_display():
    assert evidence_preview("~~~text\nexample", "", URL).endswith("\n~~~\n")


def test_boundaries_inside_fence_markers_are_recovered_from_source():
    raw = "```python\nprint(1)\n```\n## After\n"
    assert evidence_preview(raw[1:20], raw, URL) == "```python\nprint(1)\n```\n"
