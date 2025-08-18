# MCP Financial Analyst with Agno Framework Setup Guide

This guide will help you set up the enhanced MCP-powered Financial Analyst that uses the Agno agentic framework with GPT-OSS model and Firecrawl for news integration.

## 🏗️ Architecture Overview

### File Structure
```
financial-analyst-agno/
├── financial_agents.py          # Agentic system (separate file)
├── mcp_financial_main.py        # Main MCP server
├── requirements.txt             # Dependencies
├── .env                        # Environment variables
└── financial_analysis_agno_output/  # Generated output directory
```

### Agent System Components
1. **Query Parser Agent**: Converts natural language to structured queries
2. **Code Generator Agent**: Creates executable Python analysis code
3. **Market Analyst Agent**: Provides market insights and recommendations
4. **Financial Tools**: Integrates Firecrawl for news and market data

## 📋 Prerequisites

### 1. Install Ollama and GPT-OSS
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull GPT-OSS model
ollama pull gpt-oss

# Verify installation
ollama list
```

### 2. Get Firecrawl API Key (Optional)
1. Visit [Firecrawl](https://firecrawl.dev)
2. Sign up for an account
3. Get your API key from the dashboard
4. Store it securely for configuration

## 🚀 Installation

### 1. Install Python Dependencies
```bash
pip install mcp pandas matplotlib yfinance requests pydantic agno firecrawl-py
```

### 2. Create Environment File
Create a `.env` file in your project directory:
```env
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
GPT_OSS_MODEL=gpt-oss
```

### 3. Create Requirements File
```txt
# requirements.txt
mcp>=1.0.0
pandas>=2.0.0
matplotlib>=3.7.0
yfinance>=0.2.0
requests>=2.30.0
pydantic>=2.0.0
agno>=0.1.0
firecrawl-py>=0.0.8
python-dotenv>=1.0.0
```

## ⚙️ Configuration

### For Claude Desktop

**Location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "financial-analyst-agno": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_financial_main.py"],
      "env": {
        "FIRECRAWL_API_KEY": "your_api_key_here",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

### For Cursor IDE

1. Go to: **File → Preferences → Cursor Settings → MCP**
2. Add new global MCP server:

```json
{
  "mcpServers": {
    "financial-analyst-agno": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_financial_main.py"],
      "env": {
        "FIRECRAWL_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## 🎯 Usage Examples

### Basic Stock Analysis
```
Analyze Tesla stock performance over the last 3 months with market news
```

### Technical Analysis with Multiple Indicators
```
Show me Apple vs Microsoft comparison with SMA and RSI indicators for 1 year
```

### News-Focused Analysis
```
Get latest news and sentiment analysis for NVIDIA stock with price trends
```

### Comprehensive Investment Report
```
Generate a complete investment analysis for Amazon including risk assessment
```

## 🔧 Available Tools

### 1. `analyze_stock_with_agents`
**Purpose**: Complete multi-agent financial analysis
- **Input**: Natural language query
- **Process**: 
  1. Query Parser Agent structures the request
  2. Market Analyst Agent gathers insights
  3. Code Generator Agent creates analysis code
  4. News integration via Firecrawl
- **Output**: Comprehensive analysis with code, insights, and recommendations

### 2. `save_analysis_code`
**Purpose**: Save generated code with metadata
- **Features**:
  - Automatic timestamping
  - Metadata preservation
  - Organized file structure
  - Code headers with generation info

### 3. `execute_analysis_code`
**Purpose**: Secure code execution with visualization
- **Security**: Sandboxed execution environment
- **Timeout**: Configurable execution limits
- **Output**: Execution results and error handling

### 4. `get_market_news`
**Purpose**: Real-time news fetching with Firecrawl
- **Sources**: Multiple financial websites
- **Processing**: Content extraction and summarization
- **Integration**: Sentiment analysis with stock data

### 5. `generate_investment_report`
**Purpose**: Comprehensive PDF-ready reports
- **Format**: Professional investment analysis format
- **Content**: Executive summary, methodology, recommendations
- **Export**: Ready for external use

## 🤖 Agent System Details

### Query Parser Agent
```python
# Specialized in financial terminology
# Converts: "Show me Apple stock trends with moving averages"
# To: StockQuery(symbol="AAPL", analysis_type="trend", indicators=["sma"])
```

### Code Generator Agent
```python
# Creates production-ready analysis code
# Features: Error handling, professional visualizations, metrics calculation
# Libraries: yfinance, pandas, matplotlib, numpy
```

### Market Analyst Agent
```python
# Provides expert-level insights
# Analysis: Technical patterns, market sentiment, risk assessment
# Integration: News sentiment, economic indicators
```

## 📊 Enhanced Features

### Multi-Agent Collaboration
- **Parallel Processing**: Agents work simultaneously
- **Specialization**: Each agent has specific expertise
- **Quality Control**: Cross-validation between agents

### Real-Time Data Integration
- **Live Market Data**: Yahoo Finance integration
- **News Scraping**: Firecrawl web scraping
- **Sentiment Analysis**: AI-powered news sentiment

### Advanced Code Generation
- **Professional Quality**: Production-ready code
- **Error Handling**: Comprehensive error management
- **Visualization**: Advanced matplotlib charts
- **Metrics**: Financial KPIs and risk measures

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. "GPT-OSS model not found"
```bash
# Solution
ollama list
ollama pull gpt-oss
ollama serve
```

#### 2. "Agno import error"
```bash
# Solution
pip install agno
# Or try development version
pip install git+https://github.com/agno-framework/agno.git
```

#### 3. "Firecrawl authentication failed"
```bash
# Check API key
echo $FIRECRAWL_API_KEY

# Set environment variable
export FIRECRAWL_API_KEY="your_key_here"
```

#### 4. "MCP server connection failed"
- Verify file paths in configuration
- Check Python environment
- Review Claude Desktop/Cursor logs

### Performance Optimization

#### 1. Model Performance
```bash
# Allocate more resources to Ollama
OLLAMA_NUM_PARALLEL=4 ollama serve
```

#### 2. News Fetching Limits
```python
# Adjust in financial_agents.py
news_items = self.tools.get_stock_news(symbol, limit=3)  # Reduce limit
```

#### 3. Execution Timeout
```python
# Increase timeout for complex analysis
await self.execute_analysis_code(code, timeout=60)
```

## 🔐 Security Considerations

### Code Execution Safety
- **Sandboxed Environment**: Isolated execution
- **Timeout Protection**: Prevents infinite loops
- **Resource Limits**: Memory and CPU constraints
- **File System Protection**: Limited file access

### API Key Security
- **Environment Variables**: Never hardcode keys
- **File Permissions**: Secure .env files
- **Network Security**: HTTPS-only connections

## 📈 Advanced Configuration

### Custom Agent Behavior
```python
# Modify in financial_agents.py
class CustomMarketAnalyst(MarketAnalystAgent):
    def analyze_stock(self, query, news_data):
        # Custom analysis logic
        return super().analyze_stock(query, news_data)
```

### Extended News Sources
```python
# Add more news sources in FinancialTools
news_urls = [
    f"https://finance.yahoo.com/quote/{symbol}/news/",
    f"https://www.marketwatch.com/investing/stock/{symbol}",
    f"https://www.bloomberg.com/quote/{symbol}:US",  # New source
    f"https://www.reuters.com/companies/{symbol}"    # New source
]
```

### Custom Technical Indicators
```python
# Extend indicators in CodeGeneratorAgent
def add_custom_indicators(self, data):
    # Bollinger Bands
    data['BB_upper'] = data['Close'].rolling(20).mean() + 2*data['Close'].rolling(20).std()
    data['BB_lower'] = data['Close'].rolling(20).mean() - 2*data['Close'].rolling(20).std()
    return data
```

## 📚 Integration Examples

### With Jupyter Notebooks
```python
from financial_agents import FinancialAnalysisTeam

# Initialize team
team = FinancialAnalysisTeam(firecrawl_api_key="your_key")

# Run analysis
result = team.analyze("Analyze TSLA stock with news integration")

# Execute generated code
exec(result.code)
```

### With FastAPI Backend
```python
from fastapi import FastAPI
from financial_agents import FinancialAnalysisTeam

app = FastAPI()
team = FinancialAnalysisTeam()

@app.post("/analyze")
async def analyze_stock(query: str):
    result = team.analyze(query)
    return result.dict()
```

### With Streamlit Dashboard
```python
import streamlit as st
from financial_agents import FinancialAnalysisTeam

st.title("AI Financial Analyst")
query = st.text_input("Enter your analysis query:")

if query:
    team = FinancialAnalysisTeam()
    result = team.analyze(query)
    st.code(result.code)
    st.write(result.insights)
```

## 🎓 Best Practices

### 1. Query Formulation
- **Be Specific**: "AAPL 6-month trend analysis with SMA"
- **Include Timeframes**: Always specify time periods
- **Mention Indicators**: Request specific technical indicators
- **Multiple Symbols**: "Compare TSLA vs F stock performance"

### 2. Code Execution
- **Review First**: Always review generated code
- **Test Incrementally**: Run small parts first
- **Monitor Resources**: Watch CPU and memory usage
- **Save Results**: Use save_analysis_code tool

### 3. News Integration
- **API Limits**: Be mindful of Firecrawl rate limits
- **Relevance**: Focus on recent, relevant news
- **Verification**: Cross-check news sentiment with data

## 🤝 Support and Community

### Getting Help
- **Documentation**: Check agent docstrings and comments
- **Logs**: Review application logs for debugging
- **GitHub Issues**: Report bugs and feature requests
- **Community**: Join AI/ML finance communities

### Contributing
- **Code**: Submit PRs for improvements
- **Agents**: Create new specialized agents
- **Tools**: Add new financial data sources
- **Documentation**: Improve guides and examples

---

*Powered by Agno Framework, GPT-OSS, and Firecrawl for next-generation financial analysis*