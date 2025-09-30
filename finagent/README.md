# Finagent - AI-Powered Financial Analysis Tool

A sophisticated financial analysis system that leverages Google's Gemini AI and real-time market data to provide comprehensive stock analysis, automated code generation, and investment insights.

## Features

- **Multi-Agent AI System**: Specialized agents for query parsing, code generation, and market analysis
- **Real-time Stock Data**: Fetches live market data using Yahoo Finance
- **Automated Code Generation**: Creates executable Python code for financial analysis
- **News Integration**: Incorporates latest market news into analysis (via Firecrawl)
- **MCP Server**: Modern Model Context Protocol server for seamless AI assistant integration
- **Professional Visualizations**: Generates matplotlib charts and technical analysis plots
- **Risk Assessment**: Provides balanced investment recommendations with proper disclaimers

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Firecrawl API key (optional, for enhanced news features)

## File Structure

```
finagent/
├── main.py                 # MCP server entry point
├── financial_agents.py     # Core AI agents and analysis logic
├── .env                    # Environment variables (create this)
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── logs/                 # Log files (auto-created)
```

## Installation

1. **Clone the repository**:

```bash
git clone <your-repo-url>
cd finagent
```

2. **Install required packages**:

```bash
pip install google-generativeai yfinance pandas matplotlib numpy mcp python-dotenv
```

3. **Set up environment variables**:
   Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here  # Optional
```

## API Keys Setup

### Gemini API Key (Required)

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

### Firecrawl API Key (Optional)

1. Sign up at [Firecrawl](https://firecrawl.dev)
2. Get your API key from the dashboard
3. Add it to your `.env` file for enhanced news features

## Usage

### As MCP Server

Start the MCP server:

```bash
python main.py
```

The server will expose three main tools:

- `analyze_stock`: Comprehensive stock analysis with AI insights
- `execute_code`: Execute generated Python analysis code
- `get_news`: Fetch latest market news for any stock

### Direct Python Usage

```python
from financial_agents import FinancialAnalysisTeam

# Initialize the analysis team
team = FinancialAnalysisTeam(
    gemini_api_key="your_gemini_key",
    firecrawl_api_key="your_firecrawl_key"  # Optional
)

# Analyze a stock
result = team.analyze("Analyze Apple stock over the last 6 months")

print(f"Insights: {result.insights}")
print(f"Recommendations: {result.recommendations}")
print(f"Generated Code:\n{result.code}")
```

## Example Queries

The system understands natural language queries:

- "Analyze Tesla stock performance over the last 3 months"
- "Compare Apple and Microsoft stocks this year"
- "Technical analysis of NVDA with RSI and MACD indicators"
- "What's the current sentiment on Bitcoin?"

## Architecture

### Core Components

1. **FinancialAnalysisTeam**: Main orchestrator class

   - Coordinates multiple AI agents
   - Manages data flow and analysis workflow

2. **GeminiAgent**: Base agent class using Gemini AI

   - Specialized agents for different analysis tasks
   - Configurable safety settings and generation parameters

3. **FinancialTools**: Data acquisition utilities

   - Yahoo Finance integration for market data
   - News fetching capabilities

4. **MCP Server**: Modern protocol server
   - Seamless integration with AI assistants
   - RESTful API endpoints for analysis tools

### Agent Specialization

- **Query Parser**: Extracts symbols, timeframes, and analysis requirements
- **Code Generator**: Creates executable Python analysis scripts
- **Market Analyst**: Provides professional investment insights and recommendations

## Generated Analysis Includes

- **Technical Indicators**: RSI, MACD, Moving averages, Bollinger bands
- **Price Analysis**: Trend analysis, support/resistance levels
- **Risk Metrics**: Volatility calculations, drawdown analysis
- **Visualizations**: Professional charts with technical overlays
- **News Integration**: Latest market sentiment and news impact
- **Investment Recommendations**: Buy/Hold/Sell with rationale

## Important Disclaimers

- This tool is for educational and research purposes only
- All analysis and recommendations are not financial advice
- Always consult with qualified financial advisors before making investment decisions
- Past performance does not guarantee future results

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed

```bash
pip install google-generativeai yfinance pandas mcp python-dotenv
```

2. **API Key Issues**: Verify your `.env` file is in the project root and properly formatted

3. **Data Fetching Errors**: Check internet connection and ticker symbol validity

4. **Code Execution Timeout**: Large datasets may require increased timeout values

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Security

- Never commit API keys to version control
- Use environment variables for all sensitive configuration
- Regularly rotate API keys
- Monitor API usage and costs

## Support

For questions and support:

- Open an issue on GitHub
- Check the troubleshooting section above
- Review the example queries for proper usage patterns

---

**Made with care using Google Gemini AI**
