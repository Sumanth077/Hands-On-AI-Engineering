import logging
import os
import re
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import trafilatura
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def fetch_top_stories(count: int = 10) -> list[dict]:
    """Fetch the most recent tech stories from Hacker News via the Algolia Search API and return them as a list of dicts."""
    url = "https://hn.algolia.com/api/v1/search_by_date?query=tech&tags=story&hitsPerPage=10"
    response = requests.get(
        url,
        headers={"Cache-Control": "no-cache"},
        timeout=15,
    )
    response.raise_for_status()

    stories = []
    for hit in response.json().get("hits", []):
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
        stories.append({
            "id": hit["objectID"],
            "title": hit.get("title", "No title"),
            "url": url,
            "score": hit.get("points", 0),
            "author": hit.get("author", "Unknown"),
            "comments": hit.get("num_comments", 0),
        })

    logger.info("Algolia returned %d stories", len(stories))
    return stories


def extract_content(url: str) -> str:
    """Download and extract plain-text article content from a URL using Trafilatura, truncated to 1800 characters."""
    if not url or "news.ycombinator.com" in url:
        return f"[Full content unavailable — read at: {url}]"
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            if extracted:
                return extracted[:1800]
    except Exception:
        pass
    return f"[Full content unavailable — read at: {url}]"


def generate_newsletter(stories: list[dict]) -> str:
    """Send the list of HN stories to Gemma 4 and return a raw HTML newsletter document."""
    date_str = datetime.now().strftime("%B %d, %Y")

    stories_text = ""
    for i, story in enumerate(stories, 1):
        stories_text += (
            f"\nStory {i}:\n"
            f"Title: {story['title']}\n"
            f"URL: {story['url']}\n"
            f"Score: {story['score']} points | Author: {story['author']} | Comments: {story['comments']}\n"
            f"Content:\n{story.get('content', '[No content available]')}\n"
            f"---\n"
        )

    prompt = f"""You are a seasoned tech newsletter editor. Write a complete HTML newsletter covering the top Hacker News stories below.

Today's date is {date_str}. You have been given {len(stories)} top Hacker News stories.

EDITORIAL RULES:
- Summarise each story clearly and accurately in 2-3 sentences. No forced angle — just report what the story is about.
- The "Key insight" callout should highlight the single most interesting or actionable takeaway from the story.
- Write in a clear, confident voice — informative but not dry, no hype, no filler.
- If any article's content appears clearly outdated relative to today, note that briefly.

OUTPUT RULES — follow exactly:
- Output ONLY a raw HTML document. Do NOT wrap it in markdown code fences.
- Do NOT include any text, explanation, or preamble before <!DOCTYPE html>.
- Start the output with <!DOCTYPE html> and end with </html>.

Use this exact HTML structure and inline styles:

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hacker News Tech Digest</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:-apple-system,Arial,sans-serif;">
<div style="max-width:680px;margin:0 auto;background:#ffffff;">

  <div style="background:#ff6600;padding:32px 40px;text-align:center;">
    <h1 style="margin:0;color:#ffffff;font-size:26px;letter-spacing:-0.5px;">Hacker News Tech Digest</h1>
    <p style="margin:8px 0 0;color:rgba(255,255,255,0.9);font-size:14px;">{date_str}</p>
  </div>

  <div style="padding:24px 40px 0;">
    <p style="color:#555;font-size:15px;line-height:1.6;margin:0;">
      Your curated summary of today&#39;s top stories from the Hacker News community.
    </p>
  </div>

  <div style="padding:16px 40px 32px;">

    <div style="margin-bottom:28px;padding-bottom:28px;border-bottom:1px solid #eeeeee;">
      <span style="font-size:11px;font-weight:700;text-transform:uppercase;color:#ff6600;letter-spacing:0.5px;">Story #N</span>
      <h2 style="margin:6px 0 4px;font-size:18px;line-height:1.3;">
        <a href="ARTICLE_URL" style="color:#1a1a1a;text-decoration:none;">ARTICLE_TITLE</a>
      </h2>
      <p style="margin:0 0 10px;font-size:13px;color:#999;">
        &#x2B06; SCORE points &nbsp;&#xb7;&nbsp; by AUTHOR &nbsp;&#xb7;&nbsp; COMMENTS comments
      </p>
      <p style="margin:0 0 12px;font-size:15px;color:#444;line-height:1.65;">
        2-3 sentence summary of the story.
      </p>
      <div style="background:#fff8f0;border-left:3px solid #ff6600;padding:10px 14px;font-size:14px;color:#555;line-height:1.5;">
        <strong style="color:#ff6600;">Key insight:</strong> The most interesting or actionable takeaway from this story.
      </div>
    </div>

  </div>

  <div style="background:#f9f9f9;padding:20px 40px;text-align:center;border-top:1px solid #eeeeee;">
    <p style="margin:0;color:#aaa;font-size:12px;">Generated by HN Newsletter Agent on {date_str}</p>
    <p style="margin:4px 0 0;color:#aaa;font-size:12px;">Powered by Gemini via Google AI Studio</p>
  </div>

</div>
</body>
</html>

Fill in all {len(stories)} stories using the article block pattern above. Ensure every link is the real article URL.

STORIES DATA:
{stories_text}"""

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )
    return response.text


def send_email(newsletter: str, recipient: str) -> str:
    """Send the HTML newsletter to the recipient via Gmail SMTP and return a status message string."""
    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_address or not gmail_password:
        return "Error: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set in environment."

    html_content = re.sub(r"<think>.*?</think>", "", newsletter, flags=re.DOTALL).strip()
    date_str = datetime.now().strftime("%B %d, %Y")
    subject = f"Hacker News Daily Digest — {date_str}"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"HN Newsletter Agent <{gmail_address}>"
        msg["To"] = recipient
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, recipient, msg.as_string())

        return f"Newsletter successfully sent to {recipient}."

    except smtplib.SMTPAuthenticationError:
        return (
            "Authentication failed. Verify GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env. "
            "Use a Gmail App Password, not your regular account password."
        )
    except smtplib.SMTPException as exc:
        return f"SMTP error while sending: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"
