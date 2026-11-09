import unittest
from unittest.mock import patch

from email_utils import fetch_unread_emails


RAW_EMAIL = (
    b"From: Example Sender <sender@example.com>\r\n"
    b"Subject: Order update\r\n"
    b"\r\n"
    b"Please check my order."
)


class FakeIMAP:
    def __init__(self):
        self.fetch_calls = []
        self.search_calls = []
        self.logged_out = False

    def login(self, email_address, app_password):
        return "OK", []

    def select(self, mailbox):
        return "OK", [b"1"]

    def search(self, charset, criteria):
        self.search_calls.append((charset, criteria))
        return "OK", [b"1"]

    def fetch(self, message_id, query):
        self.fetch_calls.append((message_id, query))
        return "OK", [(b"1 BODY[] {message}", RAW_EMAIL), b")"]

    def logout(self):
        self.logged_out = True
        return "BYE", []


class FetchUnreadEmailsTests(unittest.TestCase):
    def test_fetches_message_without_marking_it_seen(self):
        mail = FakeIMAP()

        with patch("email_utils.imaplib.IMAP4_SSL", return_value=mail):
            messages = fetch_unread_emails(
                email_address="user@example.com",
                app_password="test-password",
            )

        self.assertEqual(mail.search_calls, [(None, "UNSEEN")])
        self.assertEqual(mail.fetch_calls, [(b"1", "(BODY.PEEK[])")])
        self.assertTrue(mail.logged_out)
        self.assertEqual(messages[0]["body"], "Please check my order.")


if __name__ == "__main__":
    unittest.main()
