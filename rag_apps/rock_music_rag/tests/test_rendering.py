import unittest

from rendering import escape_html, render_chunk_card_html


class RenderingTests(unittest.TestCase):
    def test_escape_html_escapes_tags_and_quotes(self):
        self.assertEqual(
            escape_html('<img src="x"> &'),
            "&lt;img src=&quot;x&quot;&gt; &amp;",
        )

    def test_chunk_card_escapes_wikipedia_fields(self):
        card = render_chunk_card_html(
            {
                "content": '<script>alert("x")</script>',
                "band": 'Band"><img src=x onerror="alert(1)">',
                "url": 'https://en.wikipedia.org/wiki/Band" onclick="alert(1)',
            }
        )

        self.assertIn('<div class="chunk-card">', card)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", card)
        self.assertIn("Band&quot;&gt;&lt;img", card)
        self.assertIn('href="https://en.wikipedia.org/wiki/Band&quot; onclick=&quot;alert(1)"', card)
        self.assertIn('rel="noopener noreferrer"', card)
        self.assertNotIn("<script>", card)
        self.assertNotIn('<img src=x', card)


if __name__ == "__main__":
    unittest.main()
