import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Video Understanding Agent",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Video Understanding Agent")
st.markdown("Paste a public YouTube URL to get a structured AI-powered summary powered by **Gemini Flash**.")

YOUTUBE_PATTERN = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)"
    r"[\w\-]+"
)

def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_PATTERN.search(url))

def parse_sections(text: str) -> dict:
    sections = {"chapters": "", "takeaways": "", "actions": ""}

    chapter_match = re.search(
        r"(?:#{1,3}\s*)?(?:chapter[- ]by[- ]chapter summary|chapters?.*?summary).*?\n(.*?)(?=\n#{1,3}|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    takeaway_match = re.search(
        r"(?:#{1,3}\s*)?key takeaways.*?\n(.*?)(?=\n#{1,3}|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    action_match = re.search(
        r"(?:#{1,3}\s*)?action items.*?\n(.*?)(?=\n#{1,3}|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if chapter_match:
        sections["chapters"] = chapter_match.group(1).strip()
    if takeaway_match:
        sections["takeaways"] = takeaway_match.group(1).strip()
    if action_match:
        sections["actions"] = action_match.group(1).strip()

    if not any(sections.values()):
        sections["chapters"] = text.strip()

    return sections

PROMPT = """Analyze this YouTube video and return a structured response with exactly three sections using these headings:

## Chapter-by-Chapter Summary
List each chapter or major segment with its timestamp (e.g. "0:00 - Introduction", "3:45 - Main concept explained"). Write 2–3 sentences describing what happens in each chapter. If the video has no clear chapters, break it into logical segments with approximate timestamps.

## Key Takeaways
Provide 5–8 bullet points summarizing the most important insights, facts, or lessons from the video.

## Action Items and Recommendations
List 4–6 concrete, actionable steps the viewer can take based on the video's content.

Be concise, specific, and use the viewer's perspective. Do not include any preamble before the first heading."""

def analyze_video(youtube_url: str) -> dict:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-3-flash-preview")

    video_part = {
        "file_data": {
            "file_uri": youtube_url,
            "mime_type": "video/*",
        }
    }

    response = model.generate_content([video_part, PROMPT])
    return parse_sections(response.text)


if not GEMINI_API_KEY:
    st.error(
        "**GEMINI_API_KEY not found.** Create a `.env` file with `GEMINI_API_KEY=your_key_here` and restart the app."
    )
    st.stop()

url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste any public YouTube video URL",
)

analyze_btn = st.button("Analyze Video", type="primary", use_container_width=True)

if analyze_btn:
    if not url.strip():
        st.warning("Please enter a YouTube URL.")
    elif not is_valid_youtube_url(url.strip()):
        st.error("That doesn't look like a valid YouTube URL. Please check and try again.")
    else:
        with st.spinner("Gemini is watching your video... this may take 30–60 seconds."):
            try:
                sections = analyze_video(url.strip())

                st.success("Analysis complete!")

                with st.expander("📚 Chapter-by-Chapter Summary", expanded=True):
                    if sections["chapters"]:
                        st.markdown(sections["chapters"])
                    else:
                        st.info("No chapter data returned.")

                with st.expander("💡 Key Takeaways", expanded=True):
                    if sections["takeaways"]:
                        st.markdown(sections["takeaways"])
                    else:
                        st.info("No takeaways returned.")

                with st.expander("✅ Action Items & Recommendations", expanded=True):
                    if sections["actions"]:
                        st.markdown(sections["actions"])
                    else:
                        st.info("No action items returned.")

            except Exception as e:
                err = str(e).lower()
                if "private" in err or "unavailable" in err or "not found" in err:
                    st.error(
                        "Could not access this video. It may be **private**, **age-restricted**, or **unavailable**. "
                        "Please try a different public YouTube URL."
                    )
                elif "invalid" in err or "uri" in err or "url" in err:
                    st.error(
                        "Gemini could not process this URL. Make sure it is a valid, publicly accessible YouTube link."
                    )
                elif "api_key" in err or "permission" in err or "auth" in err:
                    st.error(
                        "Authentication error. Check that your `GEMINI_API_KEY` in `.env` is correct and has access to Gemini Flash."
                    )
                else:
                    st.error(f"An unexpected error occurred: {e}")
