# Finagent - AI-Powered Financial Analysis Tool

A sophisticated financial analysis system that leverages Google's Gemini AI and real-time market data to provide comprehensive stock analysis, automated code generation, and investment insights.

## Features

- **Multi-Agent AI System**: Specialized agents for query parsing, code generation, and market analysis
- **Real-time Stock Data**: Fetches live market data using Yahoo Finance
- **Automated Code Generation**: Creates executable Python code for financial analysis
- **News Integration**: Incorporates latest market news into analysis
- **MCP Server**: Modern Model Context Protocol server for Claude Desktop integration
- **Professional Visualizations**: Generates matplotlib charts and technical analysis plots
- **Risk Assessment**: Provides balanced investment recommendations with proper disclaimers

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Claude Desktop (for MCP integration)
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

## Claude Desktop Integration

### Setup MCP Server with Claude Desktop

1. **Install Claude Desktop**:

   - Download from [Claude.ai](https://claude.ai/download)
   - Install and launch the application

2. **Configure MCP Server**:

   Locate your Claude Desktop config file:

   - **MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

3. **Add Finagent Server**:

   Edit the config file and add:

   ```json
   {
     "mcpServers": {
       "financial-analyst": {
         "command": "python",
         "args": ["/absolute/path/to/finagent/main.py"],
         "env": {
           "GEMINI_API_KEY": "your_gemini_api_key_here",
           "FIRECRAWL_API_KEY": "your_firecrawl_api_key_here"
         }
       }
     }
   }
   ```

   Replace `/absolute/path/to/finagent/main.py` with the actual path to your main.py file.

4. **Restart Claude Desktop**:

   - Quit Claude Desktop completely
   - Relaunch the application
   - The MCP server should now be available

5. **Verify Connection**:

   In Claude Desktop, you should see the financial analysis tools available. Ask Claude:

   ```
   Can you analyze Apple stock for me?
   ```

### Available Tools in Claude Desktop

Once configured, Claude can use these tools:

- **analyze_stock**: Get comprehensive stock analysis with AI insights
- **execute_code**: Run generated Python analysis code
- **get_news**: Fetch latest market news for stocks

### Example Conversations with Claude Desktop

```
You: Analyze Tesla stock performance over the last 6 months

Claude: I'll analyze Tesla stock for you using the financial analysis tool...
[Uses analyze_stock tool]
```

```
You: What's the latest news on NVDA?

Claude: Let me fetch the latest news for NVIDIA...
[Uses get_news tool]
```

```
You: Can you execute this analysis code for me?

Claude: I'll run that code using the execution tool...
[Uses execute_code tool]
```

## Usage

### As MCP Server (Standalone)

Start the MCP server manually:

```bash
python main.py
```

### Direct Python Usage

```python
from financial_agents import FinancialAnalysisTeam

team = FinancialAnalysisTeam(
    gemini_api_key="your_gemini_key",
    firecrawl_api_key="your_firecrawl_key"
)

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
- "Show me the latest news for Amazon stock"

## Architecture

### Core Components

1. **FinancialAnalysisTeam**: Main orchestrator class
2. **GeminiAgent**: Base agent class using Gemini AI
3. **FinancialTools**: Data acquisition utilities
4. **MCP Server**: Model Context Protocol server for Claude Desktop

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

1. **MCP Server Not Showing in Claude Desktop**:

   - Verify the config file path is correct
   - Check that Python path in config is valid
   - Ensure main.py path is absolute, not relative
   - Restart Claude Desktop completely

2. **Import Errors**:

```bash
pip install google-generativeai yfinance pandas mcp python-dotenv
```

3. **API Key Issues**:

   - Verify your `.env` file is in the project root
   - Check API keys are valid and properly formatted
   - Ensure environment variables are set in Claude Desktop config

4. **Data Fetching Errors**:

   - Check internet connection
   - Verify ticker symbol validity
   - Yahoo Finance may have rate limits

5. **Code Execution Timeout**:
   - Large datasets may require increased timeout values
   - Check for infinite loops in generated code

### Debug Mode

To enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

## Security

- Never commit API keys to version control
- Use environment variables for all sensitive configuration
- Regularly rotate API keys
- Monitor API usage and costs
- Keep Claude Desktop config file secure

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:

- Open an issue on GitHub
- Check the troubleshooting section above
- Review the example queries for proper usage patterns

---

**Made with Google Gemini AI & Claude Desktop**
