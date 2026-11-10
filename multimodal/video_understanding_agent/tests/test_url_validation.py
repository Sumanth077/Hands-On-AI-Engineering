import unittest

from url_validation import is_valid_youtube_url


class YouTubeUrlValidationTests(unittest.TestCase):
    def test_accepts_supported_youtube_urls(self):
        urls = [
            "https://www.youtube.com/watch?v=abc_DEF-1234&t=30",
            "https://youtu.be/abc_DEF-1234?si=share",
            "youtube.com/shorts/abc_DEF-1234",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertTrue(is_valid_youtube_url(url))

    def test_rejects_non_youtube_hosts_and_embedded_matches(self):
        urls = [
            "https://example.com/path/youtube.com/watch?v=abc123",
            "https://youtube.com.example.org/watch?v=abc123",
            "text before youtube.com/watch?v=abc123",
            "ftp://youtube.com/watch?v=abc123",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertFalse(is_valid_youtube_url(url))

    def test_rejects_missing_video_ids_and_credentialed_urls(self):
        urls = [
            "https://www.youtube.com/watch?v=",
            "https://youtu.be/",
            "https://user@youtube.com/watch?v=abc123",
            "https://youtube.com:8443/watch?v=abc123",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertFalse(is_valid_youtube_url(url))


if __name__ == "__main__":
    unittest.main()
