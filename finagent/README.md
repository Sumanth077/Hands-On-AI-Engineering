# Financial Analyst (FastMCP Server)

A high-performance financial analysis tool that exposes deep market insights via the Model Context Protocol (MCP). It allows other AI tools (or the terminal) to query real-time stock data using natural language.

## Principal Functionalities
- **Natural Language Analysis**: Ask questions like "Is Apple a good buy right now?" or "Compare Nvidia and AMD's recent trends" without needing to write code.
- **High-Speed Execution**: Optimized to deliver professional financial reports in under 20 seconds using streamlined `yfinance` data streams.
- **Model Context Protocol (MCP)**: Built with **FastMCP**, allowing this analyst to be plugged directly into any MCP-compatible environment as a specialized tool.
- **Professional Assessment**: Provides technical levels, market trends, and risk factors in a structured, institutional-grade format.

## Technical Context
- **Infrastructure**: **FastMCP** for tool definition and transport.
- **Intelligence**: OpenAI `gpt-4o-mini` for fast, concise financial reasoning.
- **Data Source**: Optimized `yfinance` historical price engine.

## Setup & Execution

### 1. Requirements
This tool runs within the shared repository environment. Ensure dependencies are met:
```bash
pip install -r requirements.txt
```

### 2. Environment
The server automatically loads your `OPENAI_API_KEY` from the root `.env` file.

### 3. Launch
```bash
python main.py
```

## How to Use
Once the server is running (Stdio transport), you can interact with the `analyze_stock` tool. It is designed to interpret a wide variety of financial queries:
- "Analyze TSLA and give me the key technical levels"
- "Give me a risk assessment for Microsoft"
- "Should I hold or sell Nvidia based on the last month?"
