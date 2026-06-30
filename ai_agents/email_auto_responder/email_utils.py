import email
import imaplib
import os
from email.header import decode_header
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def decode_mime_header(value: str | None) -> str:
    """Decode a MIME-encoded email header into plain text."""
    if not value:
        return ""

    decoded_parts = decode_header(value)
    result_parts: list[str] = []

    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            result_parts.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            result_parts.append(part)

    return " ".join(result_parts).strip()


def parse_email_body(message: email.message.Message) -> str:
    """Extract plain text or HTML body content from an email message."""
    if message.is_multipart():
        text_parts: list[str] = []
        html_parts: list[str] = []

        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition:
                continue

            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            charset = part.get_content_charset() or "utf-8"
            decoded_text = payload.decode(charset, errors="replace")

            if content_type == "text/plain":
                text_parts.append(decoded_text)
            elif content_type == "text/html" and not text_parts:
                html_parts.append(decoded_text)

        if text_parts:
            return "\n".join(text_parts).strip()

        if html_parts:
            return html_parts[0].strip()

        return ""

    payload = message.get_payload(decode=True)
    if payload is None:
        return ""

    charset = message.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace").strip()


def parse_email_message(raw_email: bytes) -> dict[str, str]:
    """Parse raw RFC822 bytes into sender, subject, and body fields."""
    message = email.message_from_bytes(raw_email)

    sender = decode_mime_header(message.get("From"))
    subject = decode_mime_header(message.get("Subject"))
    body = parse_email_body(message)

    return {
        "sender": sender,
        "subject": subject or "(No subject)",
        "body": body or "(No body content)",
    }


def fetch_unread_emails(
    email_address: str | None = None,
    app_password: str | None = None,
    max_emails: int = 5,
) -> list[dict[str, Any]]:
    """Fetch unread inbox messages from Gmail over IMAP."""
    email_address = email_address or os.getenv("EMAIL_ADDRESS")
    app_password = app_password or os.getenv("APP_PASSWORD")

    if not email_address or not app_password:
        raise ValueError(
            "EMAIL_ADDRESS and APP_PASSWORD must be set in the environment or passed as arguments."
        )

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(email_address, app_password)
    mail.select("inbox")

    try:
        _, message_numbers = mail.search(None, "UNSEEN")
        unread_ids = message_numbers[0].split()

        if not unread_ids:
            return []

        selected_ids = unread_ids[-max_emails:]
        emails: list[dict[str, Any]] = []

        for message_id in selected_ids:
            _, message_data = mail.fetch(message_id, "(RFC822)")
            if not message_data or not message_data[0]:
                continue

            raw_email = message_data[0][1]
            if not isinstance(raw_email, bytes):
                continue

            parsed = parse_email_message(raw_email)
            parsed["message_id"] = message_id.decode()
            emails.append(parsed)

        return emails
    finally:
        mail.logout()


def get_email_credentials() -> tuple[str, str]:
    """Load Gmail address and app password from environment variables."""
    email_address = os.getenv("EMAIL_ADDRESS")
    app_password = os.getenv("APP_PASSWORD")

    if not email_address or not app_password:
        raise ValueError(
            "Missing email credentials. Set EMAIL_ADDRESS and APP_PASSWORD in your .env file."
        )

    return email_address, app_password
