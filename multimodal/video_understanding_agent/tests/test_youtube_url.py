import unittest

from youtube_url import is_valid_youtube_url


class IsValidYoutubeUrlTests(unittest.TestCase):
    def test_accepts_supported_hosts_and_paths(self):
        urls = (
            "https://www.youtube.com/watch?v=abc123",
            "https://youtube.com/shorts/abc_123",
            "https://youtu.be/abc-123",
            "https://www.youtu.be/abc123?feature=shared",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertTrue(is_valid_youtube_url(url))

    def test_rejects_youtube_text_inside_another_host(self):
        self.assertFalse(
            is_valid_youtube_url(
                "https://example.com/path/youtube.com/watch?v=abc123"
            )
        )

    def test_rejects_deceptive_hosts_and_userinfo(self):
        urls = (
            "https://youtube.com.example.org/watch?v=abc123",
            "https://youtube.com@example.com/watch?v=abc123",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertFalse(is_valid_youtube_url(url))

    def test_rejects_unsupported_paths_and_missing_ids(self):
        urls = (
            "https://www.youtube.com/channel/abc123",
            "https://www.youtube.com/watch",
            "https://youtu.be/",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertFalse(is_valid_youtube_url(url))


if __name__ == "__main__":
    unittest.main()
