"""
Stock Portfolio Analyst — Gradio UI
Run: python app.py  or  gradio app.py
"""

import pandas as pd
import gradio as gr
from agent import run_analysis, MODEL_ID

# ── Default example portfolio ──────────────────────────────────────────────────
DEFAULT_PORTFOLIO = [
    ["AAPL",  10,  150.00],
    ["MSFT",   5,  300.00],
    ["NVDA",   8,  450.00],
    ["GOOGL",  3,  140.00],
    ["AMZN",   4,  180.00],
    ["TSLA",   6,  200.00],
    ["META",   7,  320.00],
]

QUICK_QUESTIONS = [
    "Full analysis: P&L, risk, and rebalancing recommendations",
    "Which positions should I trim or exit?",
    "What are my biggest concentration and valuation risks?",
    "How is my portfolio diversified across sectors?",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_portfolio(df) -> list[dict]:
    """Parse a Gradio Dataframe into a list of holding dicts."""
    holdings = []
    if not isinstance(df, pd.DataFrame):
        return holdings
    for _, row in df.iterrows():
        try:
            ticker = str(row.iloc[0]).strip().upper()
            shares = float(row.iloc[1])
            avg_cost = float(row.iloc[2])
            if ticker and ticker not in ("", "NAN", "NONE") and shares > 0 and avg_cost > 0:
                holdings.append({"ticker": ticker, "shares": shares, "avg_cost": avg_cost})
        except (ValueError, TypeError):
            continue
    return holdings


def analyze(portfolio_df, question):
    """Generator — streams the analysis report into the Markdown output."""
    holdings = parse_portfolio(portfolio_df)

    if not holdings:
        yield "⚠️ Please add at least one valid holding (ticker, shares, avg cost) before analyzing."
        return

    tickers = ", ".join(h["ticker"] for h in holdings)
    yield f"*Fetching live data for {tickers} and running analysis...*\n\n"

    report = ""
    try:
        for chunk in run_analysis(holdings, question or ""):
            report += chunk
            yield report
    except Exception as e:
        yield f"❌ Analysis error: {e}"


def reset():
    return (
        pd.DataFrame(DEFAULT_PORTFOLIO, columns=["Ticker (e.g. AAPL)", "Shares Owned", "Avg Purchase Price ($)"]),
        "",
        "*Your analysis will appear here once you click Analyze Portfolio.*",
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Stock Portfolio Analyst — DeepSeek-V4-Flash",
    css="""
        .model-badge {
            display: inline-block;
            background: #1a1a2e;
            color: #00d4ff;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78em;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .footer-note { text-align: center; opacity: 0.55; font-size: 0.82em; margin-top: 8px; }
    """,
) as demo:

    # ── Header ──
    gr.Markdown(f"""
    # 📈 Stock Portfolio Analyst
    <span class="model-badge">⚡ DeepSeek-V4-Flash</span>
    &nbsp; Powered by **Agno** · **YFinance** · **DuckDuckGo**

    Enter your holdings, ask a question (or leave blank for a full review), and hit **Analyze**.
    The agent fetches live market data, calculates P&L per position, and generates a structured report — streamed in real time.
    """)

    gr.Markdown("---")

    with gr.Row(equal_height=False):

        # ── Left column: inputs ──
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Your Stock Holdings")
            gr.Markdown(
                "Click any cell to edit. "
                "**Ticker** = stock symbol (e.g. AAPL), "
                "**Shares** = number of shares you own, "
                "**Avg Cost** = your average purchase price in USD."
            )
            portfolio_table = gr.Dataframe(
                headers=["Ticker (e.g. AAPL)", "Shares Owned", "Avg Purchase Price ($)"],
                datatype=["str", "number", "number"],
                row_count=(8, "dynamic"),
                col_count=(3, "fixed"),
                interactive=True,
                value=pd.DataFrame(
                    DEFAULT_PORTFOLIO,
                    columns=["Ticker (e.g. AAPL)", "Shares Owned", "Avg Purchase Price ($)"],
                ),
                label="",
            )

            gr.Markdown("### 💬 Analysis Question")
            question_input = gr.Textbox(
                placeholder="Leave blank for a full portfolio review, or ask something specific...",
                label="",
                lines=2,
            )

            gr.Markdown("**Quick Questions:**")
            with gr.Row():
                btn1 = gr.Button(QUICK_QUESTIONS[0], size="sm")
                btn2 = gr.Button(QUICK_QUESTIONS[1], size="sm")
            with gr.Row():
                btn3 = gr.Button(QUICK_QUESTIONS[2], size="sm")
                btn4 = gr.Button(QUICK_QUESTIONS[3], size="sm")

            gr.Markdown("---")
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Portfolio", variant="primary", scale=3)
                reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)

        # ── Right column: output ──
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Analysis Report")
            output = gr.Markdown(
                value="*Your analysis will appear here once you click Analyze Portfolio.*",
                height=680,
            )

    # ── Footer ──
    gr.Markdown(
        f"""<div class="footer-note">
        Model: <strong>DeepSeek-V4-Flash</strong> (deepseek-ai/deepseek-v4-flash) via NVIDIA API &nbsp;|&nbsp;
        Framework: Agno &nbsp;|&nbsp; Market Data: YFinance &nbsp;|&nbsp;
        This is not financial advice.
        </div>""",
    )

    # ── Event wiring ──
    for btn, q in zip([btn1, btn2, btn3, btn4], QUICK_QUESTIONS):
        btn.click(fn=lambda q=q: q, outputs=question_input)

    analyze_btn.click(
        fn=analyze,
        inputs=[portfolio_table, question_input],
        outputs=output,
    )

    reset_btn.click(
        fn=reset,
        outputs=[portfolio_table, question_input, output],
    )


if __name__ == "__main__":
    demo.launch()
