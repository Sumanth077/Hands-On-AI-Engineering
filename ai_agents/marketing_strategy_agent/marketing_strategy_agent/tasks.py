"""
Marketing Strategy Agent — Task definitions.

Tasks run sequentially. Each task receives the previous task's output as context.
"""

from crewai import Task

from .agents import creative_director, market_analyst, strategy_officer

# ── Task 1: Market Research ────────────────────────────────────────────────────

research_task = Task(
    description=(
        "Conduct market research for the following product and target audience.\n\n"
        "Product: {product_description}\n"
        "Target Audience: {target_audience}\n\n"
        "Research and cover:\n"
        "1. Market size and growth trends\n"
        "2. Top 3–5 competitors — their positioning, strengths, and weaknesses\n"
        "3. Target audience pain points, motivations, and buying behaviour\n"
        "4. Key market opportunities and gaps\n"
        "5. Relevant industry trends or shifts that affect the campaign\n\n"
        "Use web search to ground your findings in real, current data."
    ),
    expected_output=(
        "A structured market research brief in markdown with sections for: "
        "Market Overview, Competitor Analysis, Audience Insights, Opportunities, "
        "and Key Trends. Include specific data points and sources where possible."
    ),
    agent=market_analyst,
)

# ── Task 2: Marketing Strategy ─────────────────────────────────────────────────

strategy_task = Task(
    description=(
        "Using the market research brief above, develop a comprehensive marketing "
        "strategy for the following product.\n\n"
        "Product: {product_description}\n"
        "Target Audience: {target_audience}\n\n"
        "Your strategy must define:\n"
        "1. Positioning statement — what makes this product uniquely valuable\n"
        "2. Core messaging pillars (3–4 themes that will run through all comms)\n"
        "3. Target channels — ranked by priority with rationale\n"
        "4. Campaign goals and KPIs (awareness, acquisition, retention)\n"
        "5. Budget allocation guidance (% split across channels)\n"
        "6. 90-day phased rollout plan\n\n"
        "Be specific and practical. Avoid generic advice."
    ),
    expected_output=(
        "A complete marketing strategy document in markdown with sections for: "
        "Positioning, Messaging Pillars, Channel Strategy, Goals & KPIs, "
        "Budget Guidance, and 90-Day Rollout Plan."
    ),
    agent=strategy_officer,
    context=[research_task],
)

# ── Task 3: Creative Campaign ──────────────────────────────────────────────────

creative_task = Task(
    description=(
        "Using the marketing strategy above, produce the full creative campaign plan "
        "for the following product.\n\n"
        "Product: {product_description}\n"
        "Target Audience: {target_audience}\n\n"
        "Deliver:\n"
        "1. Campaign name and tagline\n"
        "2. Hero headline and 3 supporting headlines\n"
        "3. Elevator pitch (2–3 sentences for paid ads and landing pages)\n"
        "4. Channel-specific copy:\n"
        "   - 3 LinkedIn post drafts\n"
        "   - 3 short-form social posts (X/Instagram)\n"
        "   - 1 email subject line + preview text\n"
        "   - 1 Google Ads headline set (3 headlines, 2 descriptions)\n"
        "5. 5 content marketing ideas (blog posts, videos, or case studies)\n"
        "6. Launch week playbook — day-by-day activity plan\n\n"
        "Keep the voice consistent, on-brand, and tailored to the target audience."
    ),
    expected_output=(
        "A complete creative campaign document in markdown with all deliverables "
        "listed above: campaign name, headlines, ad copy, channel-specific posts, "
        "content ideas, and the launch week playbook."
    ),
    agent=creative_director,
    context=[strategy_task],
)
