# Multi-Agent Financial Research Team

A coordinated AI "Crew" that mimics an institutional research desk. It uses specialized agents to conduct deep-dive equity research and generate professional-grade investment reports.

## Principal Functionalities
- **Multi-Agent Coordination**: Utilizing the **CrewAI** framework, the system manages a specialized workflow:
    - **Sr. Financial Analyst**: Scans real-time market data to identify technical levels and trends.
    - **Investment Writer**: Synthesizes raw data into a polished, executive-ready report.
- **Institutional Reporting**: Generates markdown reports featuring formatted data tables, trend indicators (📈/📉), and risk assessments.
- **Premium User Dashboard**: A streamlined Streamlit interface that manages the complex multi-agent process behind the scenes.
- **Thread-Safe Signals**: Specifically configured to run reliably in multi-threaded dashboard environments.

## Technical Context
- **Framework**: **CrewAI** for multi-agent process management.
- **Intelligence**: OpenAI `gpt-4o-mini` (Standardized for high-speed agentic coordination).
- **Security**: Automated `.env` mapping hides sensitive API keys from the UI while maintaining full functionality.

## Setup & Execution

### 1. Requirements
This project uses the shared repository environment. Ensure root dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Environment
The agents pull configuration and keys (`OPENAI_API_KEY`) from the root `.env` file.

### 3. Launch
```bash
streamlit run financial_analyst.py
```

## How to Use
1. **Input**: Enter a stock ticker (e.g., TSLA, NVDA) in the sidebar.
2. **Execute**: Click "Generate Report". You can follow the agent logs in the terminal to see the Analyst and Writer collaborating.
3. **Review**: The final institutional report will appear on the main dashboard for review and download.