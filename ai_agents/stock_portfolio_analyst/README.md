# 📈 Stock Portfolio Analyst

> **Powered by [DeepSeek-V4-Flash](https://build.nvidia.com/deepseek-ai/deepseek-v4-flash) via NVIDIA API — fast, cost-efficient reasoning for real-time financial analysis**

![Stock Portfolio Analyst Demo](assets/demo.png)

## Overview

An AI-powered portfolio analysis agent that evaluates stock holdings using live market data. Enter your tickers, share counts, and average purchase prices, and the agent fetches real-time pricing and fundamentals via YFinance, searches for recent news via DuckDuckGo, and generates a comprehensive portfolio review — streamed live in the UI as it builds.

The report covers P&L per position, portfolio-level metrics, concentration and valuation risk flags, sector breakdown, and specific rebalancing recommendations.

## Features

- **Live Market Data:** Fetches real-time prices, fundamentals, and analyst ratings via YFinance for every holding
- **P&L Calculations:** Unrealized gain/loss per position (absolute and percentage) plus portfolio totals
- **Risk Assessment:** Flags concentration risk (positions > 20%), valuation extremes (P/E > 40), and negative news sentiment
- **Sector Breakdown:** Visualizes portfolio weight across sectors to identify diversification gaps
- **Rebalancing Recommendations:** Specific, data-driven suggestions on what to trim, hold, or add
- **Streaming Output:** Analysis streams in real time as the agent works through each holding
- **Pre-loaded Example Portfolio:** A 7-stock portfolio is ready to analyze on first launch
- **Quick Question Buttons:** One-click common analysis queries in the sidebar

## Tech Stack

**Framework & Tools:**
- [Agno](https://docs.agno.com/): Agent framework orchestrating tool use and LLM reasoning
- [YFinanceTools](https://docs.agno.com/): Live stock prices, fundamentals, analyst recommendations, company news
- [DuckDuckGoTools](https://docs.agno.com/): Real-time web search for market context and headlines
- [CalculatorTools](https://docs.agno.com/): Precise P&L and concentration calculations

**Model:**
- **[DeepSeek-V4-Flash](https://build.nvidia.com/deepseek-ai/deepseek-v4-flash)** (`deepseek-ai/deepseek-v4-flash`) — fast and cost-efficient reasoning via the NVIDIA API

**UI:**
- [Gradio](https://www.gradio.app/): Interactive web interface with streaming markdown output

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [NVIDIA API key](https://build.nvidia.com/) (free tier available)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/stock_portfolio_analyst
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your NVIDIA API key:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Get your API key at [build.nvidia.com](https://build.nvidia.com/).

### 3. Install Dependencies

```bash
uv sync
```

## Usage

```bash
uv run python app.py
```

Navigate to `http://localhost:7860` in your browser.

1. **Edit the portfolio table** — update tickers, share counts, and average purchase prices
2. **Type a question** or leave it blank for a full portfolio review
3. Click **Analyze Portfolio** — the report streams in real time as the agent fetches data
4. Use the **Quick Question** buttons for common one-click analyses
5. Click **Reset** to restore the example portfolio

## How It Works

```
User Portfolio Input
        │
        ▼
  YFinanceTools          ← live price, fundamentals, analyst ratings per ticker
        │
        ▼
  DuckDuckGoTools        ← recent news and market context
        │
        ▼
  CalculatorTools        ← P&L, weight, concentration calculations
        │
        ▼
  DeepSeek-V4-Flash      ← synthesizes data into structured markdown report
        │
        ▼
  Streamed Report        ← Summary · Holdings · Risk Flags · Rebalancing
```

## Project Structure

```text
stock_portfolio_analyst/
├── app.py            # Gradio UI and streaming interface
├── agent.py          # Agno agent, DeepSeek-V4-Flash model, tool config, prompt
├── assets/
│   └── demo.png      # App screenshot
├── .env.example      # API key template
├── pyproject.toml    # Project dependencies (uv)
└── README.md
```

## Disclaimer

This tool is for educational and informational purposes only. Nothing in this application constitutes financial advice. Always consult a qualified financial advisor before making investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

---

[⬆ Back to Top](#-stock-portfolio-analyst)
