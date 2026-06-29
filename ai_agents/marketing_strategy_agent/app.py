"""
Marketing Strategy Agent - Gradio UI.

Takes a product description and target audience, runs three agents
sequentially (Market Analyst, Strategy Officer, Creative Director), and
returns the full campaign plan split into three tabs.
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# -- Constants -----------------------------------------------------------------

PLACEHOLDER = "_Output will appear here once the campaign is generated._"

SAMPLES = [
    {
        "label": "AI Finance App",
        "product": "An AI-powered personal finance app that automatically categorises spending, sets savings goals, and nudges users toward better habits.",
        "audience": "Millennials aged 25 to 35 who earn well but feel they have little control over where their money goes.",
    },
    {
        "label": "HR Onboarding SaaS",
        "product": "A B2B SaaS tool that integrates with existing HR systems to reduce employee onboarding from 2 weeks to 2 days using automated workflows.",
        "audience": "HR directors and COOs at mid-market companies (200 to 2000 employees) frustrated with slow, manual onboarding.",
    },
    {
        "label": "Cold Brew Subscription",
        "product": "A premium cold brew coffee subscription that delivers small-batch, single-origin cold brew concentrate to your door every two weeks.",
        "audience": "Coffee enthusiasts aged 28 to 45 who drink specialty coffee daily and care about quality, sustainability, and convenience.",
    },
    {
        "label": "Fitness Coaching App",
        "product": "An AI personal trainer app that builds adaptive workout plans, tracks form via camera, and adjusts difficulty in real time.",
        "audience": "Busy professionals aged 28 to 42 who want structured workouts but cannot commit to a gym or personal trainer.",
    },
    {
        "label": "Legal AI for SMBs",
        "product": "A legal assistant tool that helps small business owners draft contracts, review NDAs, and understand compliance requirements without needing a lawyer.",
        "audience": "Small business owners and founders with 1 to 50 employees who find legal costs prohibitive and legal jargon confusing.",
    },
    {
        "label": "Sustainable Fashion Brand",
        "product": "A sustainable clothing brand that uses only deadstock and recycled fabrics, ships carbon-neutral, and lets customers track the full supply chain.",
        "audience": "Environmentally conscious shoppers aged 22 to 38 who want to look good without compromising their values.",
    },
]

# -- CSS -----------------------------------------------------------------------

CSS = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

#title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 4px;
    color: #111827;
}

#subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 15px;
    margin-bottom: 24px;
}

/* Sample cards */
.sample-card button {
    background: #f9fafb !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 52px !important;
}
.sample-card button:hover {
    background: #ede9fe !important;
    border-color: #7c3aed !important;
    color: #5b21b6 !important;
}

/* Inputs */
#product_input textarea, #audience_input textarea {
    min-height: 110px;
    font-size: 14px;
    border-radius: 8px;
}

/* Generate button */
#run_btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 8px !important;
    height: 50px !important;
    margin-top: 4px;
    width: 100% !important;
}
#run_btn:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.35) !important;
}

/* Status */
#status_bar {
    text-align: center;
    font-size: 13px;
    color: #6b7280;
    min-height: 22px;
}

/* Output tabs */
#research_out, #strategy_out, #creative_out {
    min-height: 480px;
    font-size: 14px;
    line-height: 1.75;
}

/* Agent pipeline badges */
.pipeline-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin: 12px 0 20px;
}
.pipe-badge {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #374151;
}
.pipe-arrow {
    color: #9ca3af;
    font-size: 14px;
    padding-top: 5px;
}
"""

# -- Core function -------------------------------------------------------------

def generate_campaign(product_description: str, target_audience: str):
    """Run the three-agent pipeline and yield progressive outputs for each Gradio tab."""
    product_description = product_description.strip()
    target_audience = target_audience.strip()

    if not product_description or not target_audience:
        yield PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, "Please fill in both fields."
        return

    api_key = os.getenv("ORQ_API_KEY", "")
    if not api_key:
        yield PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, "ORQ_API_KEY not set in .env"
        return

    yield (
        "_Researching market, competitors, and audience..._",
        "_Waiting for market research..._",
        "_Waiting for strategy..._",
        "Step 1 of 3: Market Analyst researching",
    )

    try:
        from marketing_strategy_agent.agents import (
            run_creative_director,
            run_market_analyst,
            run_strategy_officer,
        )

        research = run_market_analyst(api_key, product_description, target_audience)

        yield (
            research,
            "_Building positioning, channels, and rollout plan..._",
            "_Waiting for strategy..._",
            "Step 2 of 3: Strategy Officer formulating plan",
        )

        strategy = run_strategy_officer(api_key, product_description, target_audience, research)

        yield (
            research,
            strategy,
            "_Writing headlines, copy, and launch playbook..._",
            "Step 3 of 3: Creative Director writing campaign",
        )

        creative = run_creative_director(api_key, product_description, target_audience, strategy)

        yield research, strategy, creative, "Campaign generated successfully."

    except Exception as exc:
        err = f"**Error:** {exc}"
        yield err, PLACEHOLDER, PLACEHOLDER, f"Error: {exc}"


# -- UI ------------------------------------------------------------------------

with gr.Blocks(css=CSS, title="Marketing Strategy Agent") as demo:

    gr.Markdown("# Marketing Strategy Agent", elem_id="title")
    gr.Markdown(
        "Describe your product and target audience. Three specialist AI agents will "
        "research, strategise, and write your full campaign plan.",
        elem_id="subtitle",
    )

    gr.HTML("""
    <div class="pipeline-row">
        <span class="pipe-badge">🔍 Market Analyst</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-badge">📐 Strategy Officer</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-badge">🎨 Creative Director</span>
    </div>
    """)

    # Sample prompt cards
    gr.Markdown("**Try a sample**")
    with gr.Row():
        sample_btns = []
        for s in SAMPLES:
            with gr.Column(scale=1, min_width=160, elem_classes="sample-card"):
                btn = gr.Button(s["label"])
                sample_btns.append((btn, s))

    gr.Markdown("**Or describe your own**")

    with gr.Row():
        with gr.Column(scale=1):
            product_input = gr.Textbox(
                label="Product Description",
                placeholder="What does your product do? What problem does it solve? What makes it different?",
                lines=5,
                elem_id="product_input",
            )
        with gr.Column(scale=1):
            audience_input = gr.Textbox(
                label="Target Audience",
                placeholder="Who is your ideal customer? Describe their job, pain points, goals, and buying behaviour.",
                lines=5,
                elem_id="audience_input",
            )

    run_btn = gr.Button("Generate Campaign Plan", elem_id="run_btn")
    status = gr.Markdown("", elem_id="status_bar")

    with gr.Tabs():
        with gr.Tab("🔍 Market Research"):
            research_out = gr.Markdown(PLACEHOLDER, elem_id="research_out")
        with gr.Tab("📐 Marketing Strategy"):
            strategy_out = gr.Markdown(PLACEHOLDER, elem_id="strategy_out")
        with gr.Tab("🎨 Creative Campaign"):
            creative_out = gr.Markdown(PLACEHOLDER, elem_id="creative_out")

    # Wire sample buttons to fill the input fields
    for btn, sample in sample_btns:
        btn.click(
            fn=lambda p=sample["product"], a=sample["audience"]: (p, a),
            outputs=[product_input, audience_input],
        )

    run_btn.click(
        fn=generate_campaign,
        inputs=[product_input, audience_input],
        outputs=[research_out, strategy_out, creative_out, status],
    )

if __name__ == "__main__":
    demo.launch()
