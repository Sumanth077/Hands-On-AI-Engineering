# Deep Research Agent

A multi-stage research workflow agent powered by Agno, Firecrawl, and Nebius AI Cloud. This application enables users to perform deep, structured research on any topic, with automated data collection, analysis, and report generation.

## Features
- **Streamlit UI**: Simple interface for entering research queries and API keys.
- **Agentic Workflow**: Multi-stage process (search, analyze, write) using Agno agents.
- **Web Search**: Uses Firecrawl for data gathering.
- **AI Analysis**: Nebius-powered models for insight extraction and report writing.
- **References**: All sources and references are included in the final report.

---

## Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/DeepResearcherAgent
```

### 2. Create a Python Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the `DeepResearcherAgent` directory with your API keys:
```
FIRECRAWL_API_KEY=your_firecrawl_api_key
NEBIUS_API_KEY=your_nebius_api_key
```
You can obtain API keys from [Firecrawl](https://firecrawl.dev/) and [Nebius AI Cloud](https://nebius.ai/).

---

## Usage Guide

### 1. Run the Streamlit App
```bash
streamlit run app.py
```
- Enter your API keys in the sidebar.
- Type your research query in the main area and click **Submit**.
- The agent will search, analyze, and generate a detailed report with references.

### 2. Run as a Python Script
You can also run the agent directly from the command line:
```bash
python app.py
```

### 3. Run the MCP Server (Optional)
To expose the agent as an MCP tool:
```bash
python mcpserver.py
```

---

## How It Works
1. **Searcher Agent**: Uses Firecrawl to gather data from the web.
2. **Analyst Agent**: Analyzes findings, extracts insights, and lists actual references.
3. **Writer Agent**: Produces a polished, structured report with citations.

All agents are orchestrated using Agno's workflow system. The final output is a comprehensive markdown report.

---

## Troubleshooting
- Ensure your API keys are valid and set in the `.env` file or via the Streamlit sidebar.
- If you encounter missing dependencies, run `pip install -r requirements.txt` again.
- For issues with Streamlit, ensure you are using Python 3.8 or higher.

---

## License
MIT

## Author
Avikumar Talaviya

---

## References
- [Agno Documentation](https://github.com/agnolabs/agno)
- [Firecrawl](https://firecrawl.dev/)
- [Nebius AI Cloud](https://nebius.ai/)
- [Streamlit](https://streamlit.io/)
