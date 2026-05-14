"""
Marketing Strategy Agent — Agent definitions.

Three specialist agents run sequentially via the OpenAI SDK:
  1. Market Analyst     — researches market, competitors, and audience (uses Serper)
  2. Strategy Officer   — formulates the marketing strategy
  3. Creative Director  — writes the full campaign content
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from .tools import search_web

ORQ_BASE_URL = "https://my.orq.ai/v3/router"
MODEL_ID = "alibaba/deepseek-v4-flash"

# ── Tool schema for the Market Analyst ────────────────────────────────────────

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current market data, competitor info, and audience insights.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
}

# ── Shared LLM client factory ──────────────────────────────────────────────────

def _client(api_key: str) -> OpenAI:
    return OpenAI(base_url=ORQ_BASE_URL, api_key=api_key)


# ── Agent 1: Market Analyst ────────────────────────────────────────────────────

ANALYST_SYSTEM = """\
You are a senior market researcher with 15 years of experience across consumer, B2B, and SaaS markets.
You are known for tight, actionable research briefs that cut straight to what matters.
Use web search to ground every claim in current, real data.
"""

def run_market_analyst(api_key: str, product: str, audience: str) -> str:
    """Research the market using Serper search and return a structured brief."""
    client = _client(api_key)
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Conduct market research for this product and target audience.\n\n"
                f"Product: {product}\n"
                f"Target Audience: {audience}\n\n"
                "Research and cover:\n"
                "1. Market size and growth trends\n"
                "2. Top 3–5 competitors — positioning, strengths, and weaknesses\n"
                "3. Target audience pain points, motivations, and buying behaviour\n"
                "4. Key market opportunities and gaps\n"
                "5. Relevant industry trends that affect the campaign\n\n"
                "Use search_web to find current data. Make at least 3 searches.\n"
                "Return a structured markdown brief with sections for: Market Overview, "
                "Competitor Analysis, Audience Insights, Opportunities, and Key Trends."
            ),
        },
    ]

    # Tool-calling loop
    while True:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tools=[SEARCH_TOOL],
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = search_web(args["query"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return msg.content or ""


# ── Agent 2: Strategy Officer ──────────────────────────────────────────────────

STRATEGIST_SYSTEM = """\
You are a Chief Marketing Strategist who has launched products across consumer tech, fintech, and enterprise software.
You think in frameworks but write in plain language, and you never produce a strategy that can't be executed.
"""

def run_strategy_officer(api_key: str, product: str, audience: str, research: str) -> str:
    """Turn market research into a full marketing strategy."""
    client = _client(api_key)
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": STRATEGIST_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Using the market research below, develop a comprehensive marketing strategy.\n\n"
                    f"Product: {product}\n"
                    f"Target Audience: {audience}\n\n"
                    f"## Market Research\n{research}\n\n"
                    "Your strategy must define:\n"
                    "1. Positioning statement — what makes this product uniquely valuable\n"
                    "2. Core messaging pillars (3–4 themes across all comms)\n"
                    "3. Target channels — ranked by priority with rationale\n"
                    "4. Campaign goals and KPIs (awareness, acquisition, retention)\n"
                    "5. Budget allocation guidance (% split across channels)\n"
                    "6. 90-day phased rollout plan\n\n"
                    "Return a complete markdown strategy document with all sections."
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


# ── Agent 3: Creative Director ─────────────────────────────────────────────────

CREATIVE_SYSTEM = """\
You are a Creative Director who has led campaigns for challenger brands and Fortune 500 companies.
You write copy that converts and ideas that stick, always keeping the brand voice and target audience front of mind.
"""

def run_creative_director(api_key: str, product: str, audience: str, strategy: str) -> str:
    """Translate the strategy into a full creative campaign plan."""
    client = _client(api_key)
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": CREATIVE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Using the marketing strategy below, produce the full creative campaign plan.\n\n"
                    f"Product: {product}\n"
                    f"Target Audience: {audience}\n\n"
                    f"## Marketing Strategy\n{strategy}\n\n"
                    "Deliver:\n"
                    "1. Campaign name and tagline\n"
                    "2. Hero headline and 3 supporting headlines\n"
                    "3. Elevator pitch (2–3 sentences for ads and landing pages)\n"
                    "4. Channel-specific copy:\n"
                    "   - 3 LinkedIn post drafts\n"
                    "   - 3 short-form social posts (X/Instagram)\n"
                    "   - 1 email subject line + preview text\n"
                    "   - 1 Google Ads headline set (3 headlines, 2 descriptions)\n"
                    "5. 5 content marketing ideas (blog posts, videos, or case studies)\n"
                    "6. Launch week playbook — day-by-day activity plan\n\n"
                    "Return everything in clean markdown."
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""
