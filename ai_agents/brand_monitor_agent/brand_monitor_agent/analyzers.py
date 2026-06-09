"""
Platform analyzers for brand monitoring.
Each function collects raw data via Scrapingdog then makes a single
LLM call via Orq.ai to produce a structured intelligence brief.
Total cost: 4 LLM calls for a full brand monitor run.
"""

from openai import OpenAI

from brand_monitor_agent.tools import (
    _google_news,
    _google_search,
    _youtube_search,
)

MODEL = "alibaba/deepseek-v4-flash"

# ---------------------------------------------------------------------------
# Shared LLM helper
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, orq_key: str) -> str:
    client = OpenAI(
        base_url="https://my.orq.ai/v3/router",
        api_key=orq_key,
    )
    response = client.responses.create(
        model=MODEL,
        instructions=system,
        input=user,
    )
    return response.output[0].content[0].text.strip()


# ---------------------------------------------------------------------------
# Shared system prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a brand intelligence analyst. You receive raw data collected from {platform}
about a brand and produce a concise, structured report. Be specific and evidence-based.
Use **bold** for brand names, product names, and key figures.
Never invent information — only use what is present in the data provided.\
"""

# ---------------------------------------------------------------------------
# Web analyzer
# ---------------------------------------------------------------------------

def analyze_web(brand: str, scrapingdog_key: str, orq_key: str) -> str:
    """Collect web search + news data and produce a brand intelligence brief."""
    serp = _google_search(f"{brand} review site reputation", scrapingdog_key, results=8)
    news = _google_news(brand, scrapingdog_key, results=6)

    serp_text = "\n".join(
        f"- {r.get('title','')}: {r.get('snippet', r.get('description',''))}"
        for r in serp[:8]
    )
    news_text = "\n".join(
        f"- [{a.get('source','')} | {a.get('lastUpdated','')}] {a.get('title','')}: {a.get('snippet','')}"
        for a in news[:6]
    )

    user_prompt = f"""Brand: {brand}

--- WEB SEARCH RESULTS ---
{serp_text or "No results found."}

--- RECENT NEWS (past month) ---
{news_text or "No news found."}

Produce a structured brand intelligence brief for the WEB channel with these sections:

### Overall Web Sentiment
Positive / Neutral / Negative assessment with supporting evidence from the data.

### Key Themes
Top 3-5 recurring themes across web coverage.

### Notable Mentions
Most impactful articles, reviews, or discussions found.

### Recent Developments
News and announcements from the data above.

### Brand Health Assessment
One concise paragraph summarising the brand's current web presence and reputation."""

    return _call_llm(
        SYSTEM_PROMPT.format(platform="the web (Google SERP + Google News)"),
        user_prompt,
        orq_key,
    )


# ---------------------------------------------------------------------------
# YouTube analyzer
# ---------------------------------------------------------------------------

def analyze_youtube(brand: str, scrapingdog_key: str, orq_key: str) -> str:
    """Collect YouTube video data and produce a brand intelligence brief."""
    videos = _youtube_search(brand, scrapingdog_key)
    reviews = _youtube_search(f"{brand} review", scrapingdog_key)

    all_videos = list({v.get("title", ""): v for v in (videos + reviews)}.values())

    if all_videos:
        video_text = "\n".join(
            f"- [{v.get('channel', v.get('channel_name',''))} | {v.get('views', v.get('view_count',''))} views] "
            f"{v.get('title','')}: {str(v.get('description', v.get('snippet','')))[:150]}"
            for v in all_videos[:10]
        )
    else:
        # Fall back to Google SERP filtered to youtube.com
        serp = _google_search(f"site:youtube.com {brand} review", scrapingdog_key, results=10)
        video_text = "\n".join(
            f"- {r.get('title','')}: {r.get('snippet', r.get('description',''))}"
            for r in serp[:10]
        ) or "No YouTube results found."

    user_prompt = f"""Brand: {brand}

--- YOUTUBE VIDEO RESULTS ---
{video_text or "No YouTube videos found."}

Produce a structured brand intelligence brief for the YOUTUBE channel with these sections:

### YouTube Sentiment Overview
Overall tone across video content (positive / mixed / negative) with evidence.

### Top Video Topics
Most common themes in videos about the brand.

### Notable Creators
Key YouTubers covering the brand and their reach.

### High-Impact Videos
Most-viewed or most-discussed videos with brief summaries.

### YouTube Presence Assessment
One concise paragraph on the brand's overall YouTube footprint and reputation."""

    return _call_llm(
        SYSTEM_PROMPT.format(platform="YouTube"),
        user_prompt,
        orq_key,
    )


# ---------------------------------------------------------------------------
# Twitter/X analyzer
# ---------------------------------------------------------------------------

def analyze_twitter(brand: str, scrapingdog_key: str, orq_key: str) -> str:
    """Collect Twitter/X mention data and produce a brand intelligence brief."""
    results = _google_search(f"site:twitter.com {brand}", scrapingdog_key, results=10)
    results_neg = _google_search(
        f"site:twitter.com {brand} bad OR problem OR hate", scrapingdog_key, results=5
    )

    all_results = list(
        {r.get("link", r.get("url", "")): r for r in (results + results_neg)}.values()
    )

    tweets_text = "\n".join(
        f"- {r.get('title','')}: {r.get('snippet', r.get('description',''))}"
        for r in all_results[:10]
    )

    user_prompt = f"""Brand: {brand}

--- TWITTER/X MENTIONS ---
{tweets_text or "No Twitter/X results found."}

Produce a structured brand intelligence brief for the TWITTER/X channel with these sections:

### Twitter/X Sentiment Breakdown
Qualitative split of positive / neutral / negative posts with evidence.

### Dominant Conversations
Top 3-5 recurring topics or narratives in brand discussions.

### Influential Voices
Notable accounts or communities discussing the brand.

### Viral or High-Engagement Moments
Any posts or threads generating outsized engagement.

### Social Pulse Assessment
One concise paragraph summarising the brand's current Twitter/X reputation."""

    return _call_llm(
        SYSTEM_PROMPT.format(platform="Twitter/X"),
        user_prompt,
        orq_key,
    )


# ---------------------------------------------------------------------------
# LinkedIn analyzer
# ---------------------------------------------------------------------------

def analyze_linkedin(brand: str, scrapingdog_key: str, orq_key: str) -> str:
    """Collect LinkedIn mention data and produce a brand intelligence brief."""
    results = _google_search(f"site:linkedin.com {brand}", scrapingdog_key, results=10)
    results_emp = _google_search(
        f"site:linkedin.com {brand} company employees jobs", scrapingdog_key, results=5
    )

    all_results = list(
        {r.get("link", r.get("url", "")): r for r in (results + results_emp)}.values()
    )

    posts_text = "\n".join(
        f"- {r.get('title','')}: {r.get('snippet', r.get('description',''))}"
        for r in all_results[:10]
    )

    user_prompt = f"""Brand: {brand}

--- LINKEDIN RESULTS ---
{posts_text or "No LinkedIn results found."}

Produce a structured brand intelligence brief for the LINKEDIN channel with these sections:

### LinkedIn Presence Overview
How active and prominent the brand is on LinkedIn.

### Professional Sentiment
How professionals and employees perceive the brand.

### Company Activity
Recent posts, announcements, and updates visible on LinkedIn.

### B2B Reputation Signals
Partnerships, client testimonials, industry recognition.

### Professional Presence Assessment
One concise paragraph summarising the brand's LinkedIn standing and B2B reputation."""

    return _call_llm(
        SYSTEM_PROMPT.format(platform="LinkedIn"),
        user_prompt,
        orq_key,
    )
