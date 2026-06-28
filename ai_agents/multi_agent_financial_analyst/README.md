<<<<<<< HEAD
=======
<a id="top"></a>
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
# 🎯 Multi-Agent AI Financial Analyst

A powerful multi-agent system that performs comprehensive stock analysis and generates detailed financial reports.

<<<<<<< HEAD
=======
## Demo

![Demo](assets/demo.png)

## Tech Stack

- **Provider:** Orq.ai
- **Model:** alibaba/qwen3.6-flash via Orq.ai
- **Agent framework:** CrewAI
- **UI:** Streamlit
- **Market data:** yfinance

## Prerequisites

- Python 3.10 or higher
- An Orq.ai account and API key, available at https://orq.ai

>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
## Features

- Stock Analysis Agent: Performs thorough analysis of stocks using fundamental and technical indicators
- Report Writing Agent: Transforms analysis into comprehensive, reader-friendly reports
- Real-time market data integration
- Professional report generation in markdown format
- Interactive Streamlit interface

## Installation

1.  **Clone Repository & Navigate:**
    ```bash
    git clone https://github.com/Sumanth077/awesome-ai-apps-and-agents.git
    cd awesome-ai-apps-and-agents/multi_agent_financial_analyst
    ```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
3. Create a `.env` file in the root directory and add your SambaNova API key:
```bash
SAMBANOVA_API_KEY=your_api_key_here
=======
3. Create a `.env` file in the root directory and add your Orq.ai API key:
```bash
ORQ_API_KEY=your_orq_api_key_here
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run financial_analyst.py
```

2. In the web interface:
   - Enter your stock symbol (e.g., AAPL, GOOGL)
   - Click "Analyze Stock" to start the analysis
   - Wait for the agents to complete their analysis
   - Download the generated report

## How It Works

The application uses two specialized AI agents:

1. **Stock Analysis Agent**
   - Analyzes company fundamentals
   - Reviews market news
   - Evaluates technical indicators
   - Assesses market trends

2. **Report Writing Agent**
   - Compiles analysis into structured reports
   - Formats information for clarity
   - Generates actionable insights
   - Creates downloadable markdown reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

<<<<<<< HEAD
This project is licensed under the MIT License - see the LICENSE file for details. 
=======
This project is licensed under the MIT License - see the LICENSE file for details.

[Back to top](#top)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
