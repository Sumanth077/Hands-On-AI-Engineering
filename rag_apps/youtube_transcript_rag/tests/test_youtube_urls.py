import unittest

from youtube_urls import extract_video_id, youtube_watch_url


VIDEO_ID = "dQw4w9WgXcQ"


class YouTubeUrlTests(unittest.TestCase):
    def test_extracts_video_id_from_supported_youtube_urls(self):
        urls = (
            f"https://www.youtube.com/watch?v={VIDEO_ID}",
            f"https://m.youtube.com/watch?v={VIDEO_ID}&list=PL123",
            f"https://youtu.be/{VIDEO_ID}",
            f"https://www.youtube.com/embed/{VIDEO_ID}",
            f"https://www.youtube.com/v/{VIDEO_ID}",
            f"https://youtube.com/shorts/{VIDEO_ID}",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), VIDEO_ID)

    def test_rejects_non_youtube_hosts_and_deceptive_hosts(self):
        urls = (
            f"https://example.com/watch?v={VIDEO_ID}",
            f"https://youtube.com.example.org/watch?v={VIDEO_ID}",
            f"https://www.youtube.com@evil.example/watch?v={VIDEO_ID}",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertIsNone(extract_video_id(url))

    def test_rejects_unsupported_schemes_and_malformed_ids(self):
        self.assertIsNone(extract_video_id(f"file:///tmp/watch?v={VIDEO_ID}"))
        self.assertIsNone(extract_video_id(f"https://youtube.com:8443/watch?v={VIDEO_ID}"))
        self.assertIsNone(extract_video_id(f"https://youtube.com:invalid/watch?v={VIDEO_ID}"))
        self.assertIsNone(extract_video_id("https://youtu.be/short"))
        self.assertIsNone(extract_video_id("https://www.youtube.com/watch?v=abc&v=def"))

    def test_builds_canonical_youtube_watch_url(self):
        self.assertEqual(
            youtube_watch_url(VIDEO_ID),
            f"https://www.youtube.com/watch?v={VIDEO_ID}",
        )
        with self.assertRaises(ValueError):
            youtube_watch_url("not-a-valid-id")


if __name__ == "__main__":
    unittest.main()
