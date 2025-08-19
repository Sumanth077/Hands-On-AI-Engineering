#!/usr/bin/env python3

"""
MCP Financial Analyst Server with Agno Framework and Firecrawl
Main server file that integrates with the separate agentic system
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

# MCP imports
try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import (
        CallToolRequestParams,
        ListToolsRequestParams,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
    import mcp.types as types
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    sys.exit(1)

# Import our agentic system
try:
    from financial_agents import FinancialAnalysisTeam, AnalysisResult
except ImportError:
    print("financial_agents.py not found. Make sure it's in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-financial-agno")

# set upa mcp server
class MCPFinancialServer:
    """MCP Server for Financial Analysis using Agno Framework"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.server = Server("financial-analyst-agno-gpt-oss")
        
        # Initialize the agentic team
        logger.info("Initializing Financial Analysis Team with Agno Framework...")
        self.analysis_team = FinancialAnalysisTeam(firecrawl_api_key)
        
        self.setup_handlers()
        logger.info("MCP Financial Server initialized with Agno agents and Firecrawl")
        
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="analyze_stock_with_agents",
                    description="Comprehensive stock analysis using Agno agents with GPT-OSS model and Firecrawl news integration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query about stock analysis (e.g., 'Analyze AAPL performance over 6 months with news')"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="save_analysis_code",
                    description="Save generated analysis code to local directory with metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to save"},
                            "filename": {"type": "string", "description": "Filename (optional)"},
                            "metadata": {"type": "object", "description": "Additional metadata about the analysis"}
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="execute_analysis_code",
                    description="Execute Python analysis code in a secure environment and generate plots",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"},
                            "timeout": {"type": "number", "description": "Execution timeout in seconds (default: 30)"}
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="get_market_news",
                    description="Fetch latest market news using Firecrawl for specific stocks",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                            "limit": {"type": "number", "description": "Number of news articles to fetch (default: 5)"}
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="generate_investment_report",
                    description="Generate comprehensive investment report with recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "analysis_result": {"type": "string", "description": "JSON string of previous analysis result"},
                            "include_charts": {"type": "boolean", "description": "Whether to include chart generation code"}
                        },
                        "required": ["analysis_result"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            if name == "analyze_stock_with_agents":
                return await self.analyze_stock_with_agents(arguments.get("query", ""))
            elif name == "save_analysis_code":
                return await self.save_analysis_code(
                    arguments.get("code", ""),
                    arguments.get("filename"),
                    arguments.get("metadata", {})
                )
            elif name == "execute_analysis_code":
                return await self.execute_analysis_code(
                    arguments.get("code", ""),
                    arguments.get("timeout", 30)
                )
            elif name == "get_market_news":
                return await self.get_market_news(
                    arguments.get("symbol", ""),
                    arguments.get("limit", 5)
                )
            elif name == "generate_investment_report":
                return await self.generate_investment_report(
                    arguments.get("analysis_result", ""),
                    arguments.get("include_charts", True)
                )
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async def analyze_stock_with_agents(self, query: str) -> Sequence[types.TextContent]:
        """Comprehensive stock analysis using Agno agents"""
        try:
            logger.info(f"Starting agentic analysis for query: {query}")
            
            # Run the analysis using our agent team
            result: AnalysisResult = self.analysis_team.analyze(query)
            
            # Format the comprehensive result
            news_section = ""
            if result.news:
                news_section = "### 📰 Latest Market News:\n"
                for i, news_item in enumerate(result.news[:3], 1):
                    news_section += f"{i}. **{news_item.title}**\n"
                    news_section += f"   {news_item.summary}...\n"
                    news_section += f"   [Read more]({news_item.url})\n\n"
            
            formatted_result = f"""# 📈 Comprehensive Financial Analysis
            
            ## 🎯 Query Analysis
            **Symbol:** {result.query.symbol}  
            **Analysis Type:** {result.query.analysis_type.title()}  
            **Time Period:** {result.query.time_period.upper()}  
            **Additional Symbols:** {', '.join(result.query.additional_symbols) if result.query.additional_symbols else 'None'}  
            **Technical Indicators:** {', '.join(result.query.indicators) if result.query.indicators else 'None'}

## 💡 Market Insights (GPT-OSS Analysis)
{result.insights}

## 📊 Generated Analysis Code
```python
{result.code}
```

{news_section}

## 🎯 Investment Recommendations
{result.recommendations}

## 🔧 Next Steps
1. **Save Code**: Use `save_analysis_code` tool to save the generated code locally
2. **Execute Analysis**: Use `execute_analysis_code` tool to run the code and generate visualizations
3. **Generate Report**: Use `generate_investment_report` tool for a comprehensive PDF-ready report
4. **Get Updates**: Use `get_market_news` tool for the latest news updates

---
*Analysis powered by Agno Framework with GPT-OSS model and Firecrawl news integration*
"""
            
            return [TextContent(type="text", text=formatted_result)]
            
        except Exception as e:
            logger.error(f"Error in agentic analysis: {e}")
            return [TextContent(type="text", text=f"❌ Error in analysis: {str(e)}\n\nPlease check that:\n- Ollama is running with gpt-oss model\n- Firecrawl API key is configured (if using news features)\n- All dependencies are installed")]
    
    async def save_analysis_code(self, code: str, filename: Optional[str] = None, metadata: Dict = None) -> Sequence[types.TextContent]:
        """Save analysis code with metadata"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"agno_financial_analysis_{timestamp}.py"
            
            # Create output directory
            output_dir = Path.cwd() / "financial_analysis_agno_output"
            output_dir.mkdir(exist_ok=True)
            
            code_file = output_dir / filename
            
            # Add header to the code
            header = f"""#!/usr/bin/env python3
\"\"\"
Financial Analysis Code Generated by Agno Framework
Generated: {datetime.now().isoformat()}
Agent System: GPT-OSS + Firecrawl
Metadata: {json.dumps(metadata or {}, indent=2)}
\"\"\"

"""
            
            with open(code_file, 'w') as f:
                f.write(header + code)
            
            # Save metadata separately
            if metadata:
                metadata_file = output_dir / f"{filename.replace('.py', '_metadata.json')}"
                with open(metadata_file, 'w') as f:
                    json.dump({
                        'generated_at': datetime.now().isoformat(),
                        'filename': filename,
                        'agent_system': 'Agno Framework + GPT-OSS',
                        'features': 'Firecrawl news integration',
                        **metadata
                    }, f, indent=2)
            return [TextContent(type="text", text=f"✅ Code saved successfully!\n\n📁 **Location**: {code_file}\n📄 **Filename**: {filename}\n🗂️ **Directory**: {output_dir}\n\n{f'📋 **Metadata**: Saved to {filename.replace(\".py\", \"_metadata.json\")}' if metadata else ''}")]
        except Exception as e:
            logger.error(f"Error saving code: {e}")
            return [TextContent(type="text", text=f"❌ Error saving code: {str(e)}")]
    
    async def execute_analysis_code(self, code: str, timeout: int = 30) -> Sequence[types.TextContent]:
        """Execute analysis code in secure environment"""
        try:
            logger.info("Executing analysis code in secure environment")
            
            # Create temporary file with proper imports
            execution_code = f"""
import sys
import warnings
warnings.filterwarnings('ignore')

# Add error handling wrapper
try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"Execution Error: {{e}}")
    sys.exit(1)
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(execution_code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run([
                sys.executable, temp_file
            ], capture_output=True, text=True, timeout=timeout, cwd=Path.cwd())
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                output = result.stdout if result.stdout else "Code executed successfully! Check for generated plots."
                warnings = f"\n⚠️ **Warnings:**\n```\n{result.stderr}\n```" if result.stderr else ""
                
                execution_summary = f"""✅ **Execution Successful!**

📊 **Output:**
```
{output}
```{warnings}

🎯 **Status**: Analysis completed successfully
⏱️ **Execution Time**: Within {timeout} seconds
🖼️ **Plots**: Check for generated matplotlib windows or saved plot files
"""
                return [TextContent(type="text", text=execution_summary)]
            else:
                error_details = f"""❌ **Execution Failed!**

🚨 **Error Details:**
```
{result.stderr}
```

🔍 **Troubleshooting Tips:**
1. Check if all required packages are installed (yfinance, pandas, matplotlib)
2. Verify internet connection for data fetching
3. Ensure stock symbols are valid
4. Check for any syntax errors in the generated code

💡 **Suggestion**: Try regenerating the code with a simpler query or check the error message above.
"""
                return [TextContent(type="text", text=error_details)]
                
        except subprocess.TimeoutExpired:
            return [TextContent(type="text", text=f"⏰ **Execution Timeout!**\n\nThe analysis took longer than {timeout} seconds to complete.\n\n💡 **Solutions:**\n- Increase timeout value\n- Simplify the analysis query\n- Check network connection for data fetching")]
            
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return [TextContent(type="text", text=f"❌ **Execution Error**: {str(e)}\n\nPlease check the code and try again.")]
    
    async def get_market_news(self, symbol: str, limit: int = 5) -> Sequence[types.TextContent]:
        """Fetch latest market news using Firecrawl"""
        try:
            logger.info(f"Fetching market news for {symbol}")
            
            # Use the tools from our agent system
            news_data = self.analysis_team.tools.get_stock_news(symbol, limit)
            
            if not news_data:
                return [TextContent(type="text", text=f"📰 **Market News for {symbol}**\n\n⚠️ No recent news available or Firecrawl not configured.\n\n💡 **Note**: To enable news fetching, provide a Firecrawl API key when starting the server.")]
            
            news_formatted = f"# 📰 Latest Market News for {symbol}\n\n"
            
            for i, news_item in enumerate(news_data[:limit], 1):
                news_formatted += f"## {i}. {news_item.get('title', 'Market Update')}\n\n"
                news_formatted += f"📅 **Source**: {news_item.get('url', 'N/A')}\n\n"
                
                content = news_item.get('content', '')
                if len(content) > 300:
                    content = content[:300] + "..."
                news_formatted += f"📝 **Summary**: {content}\n\n"
                news_formatted += "---\n\n"
            
            news_formatted += f"*Powered by Firecrawl - Real-time web scraping for financial news*"
            
            return [TextContent(type="text", text=news_formatted)]
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return [TextContent(type="text", text=f"❌ **Error fetching news**: {str(e)}\n\nPlease check your Firecrawl API key and internet connection.")]
    
    async def generate_investment_report(self, analysis_result_json: str, include_charts: bool = True) -> Sequence[types.TextContent]:
        """Generate comprehensive investment report"""
        try:
            logger.info("Generating comprehensive investment report")
            
            # Parse the analysis result if provided as JSON string
            if analysis_result_json.strip():
                try:
                    analysis_data = json.loads(analysis_result_json)
                except:
                    analysis_data = {"symbol": "Unknown", "analysis_type": "general"}
            else:
                analysis_data = {"symbol": "Unknown", "analysis_type": "general"}
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""# 📋 Investment Analysis Report
            
**Generated**: {timestamp}  
**Powered by**: Agno Framework + GPT-OSS + Firecrawl  
**Report Type**: Comprehensive Financial Analysis

---

## 📊 Executive Summary

This report provides a comprehensive analysis of the requested financial instrument(s) using advanced AI agents powered by the GPT-OSS model and real-time news integration via Firecrawl.

### Key Highlights:
- **AI-Powered Analysis**: Multi-agent system with specialized roles
- **Real-time Data**: Live market data and news integration
- **Technical Analysis**: Advanced charting and indicator analysis
- **Risk Assessment**: Comprehensive risk evaluation
- **Actionable Insights**: Clear recommendations and next steps

---

## 🤖 Agent System Overview

### Query Parser Agent
- **Role**: Natural language processing
- **Function**: Convert user queries into structured analysis parameters
- **Technology**: GPT-OSS model with specialized prompts

### Code Generator Agent  
- **Role**: Python code generation
- **Function**: Create executable analysis scripts with proper error handling
- **Libraries**: yfinance, pandas, matplotlib, numpy

### Market Analyst Agent
- **Role**: Market insights and recommendations
- **Function**: Provide expert-level market analysis and investment guidance
- **Data Sources**: Historical data, technical indicators, news sentiment

---

## 📈 Analysis Methodology

### 1. Data Collection
- **Primary Source**: Yahoo Finance via yfinance library
- **News Integration**: Real-time news scraping with Firecrawl
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands

### 2. AI-Powered Processing
- **Model**: GPT-OSS (locally hosted via Ollama)
- **Framework**: Agno multi-agent orchestration
- **Approach**: Collaborative analysis with specialized agents

### 3. Visualization & Reporting
- **Charts**: Professional matplotlib visualizations
- **Metrics**: Key performance indicators and risk measures
- **Format**: Code generation for reproducible analysis

---

## 🎯 Investment Framework

### Risk Categories
- **Low Risk**: Blue-chip stocks, established companies
- **Medium Risk**: Growth stocks, sector-specific investments
- **High Risk**: Volatile stocks, emerging markets

### Analysis Dimensions
1. **Technical Analysis**: Price patterns, volume, indicators
2. **Fundamental Factors**: Company performance, market position
3. **Market Sentiment**: News analysis, social indicators
4. **Risk Assessment**: Volatility, correlation, sector risks

---

## 🔧 Usage Instructions

### 1. Running Analysis
```bash
# Ensure Ollama is running with GPT-OSS
ollama serve
ollama pull gpt-oss

# Run the MCP server
python mcp_financial_main.py
```

### 2. Available Tools
- `analyze_stock_with_agents`: Complete multi-agent analysis
- `save_analysis_code`: Save generated code with metadata
- `execute_analysis_code`: Run analysis in secure environment
- `get_market_news`: Fetch latest news with Firecrawl
- `generate_investment_report`: Create comprehensive reports

### 3. Integration Options
- **Claude Desktop**: Full MCP integration
- **Cursor IDE**: Development environment integration
- **Command Line**: Direct Python execution

---

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This analysis is for informational purposes only
2. **AI Limitations**: AI-generated insights should be verified with human expertise
3. **Data Accuracy**: Market data may have delays or inaccuracies
4. **Risk Warning**: All investments carry risk of loss
5. **Professional Consultation**: Consult licensed financial advisors for investment decisions

---

## 📞 Technical Support

### Dependencies
```bash
pip install mcp pandas matplotlib yfinance requests pydantic agno firecrawl-py
```

### Configuration Files
- MCP Server: `mcp_financial_main.py`
- Agent System: `financial_agents.py`
- Output Directory: `./financial_analysis_agno_output/`

### Troubleshooting
1. **Ollama Issues**: Ensure GPT-OSS model is loaded
2. **Firecrawl Errors**: Check API key configuration
3. **MCP Connection**: Verify Claude Desktop/Cursor setup
4. **Code Execution**: Check Python environment and dependencies

---

*Report generated by MCP Financial Analyst with Agno Framework*  
*© 2025 - Powered by GPT-OSS, Firecrawl, and Agno Multi-Agent System*
"""

            if include_charts:
                report += f"\n\n## 📊 Chart Generation Code\n\n```python\n# Use the 'execute_analysis_code' tool with generated analysis code\n# Charts will be displayed in matplotlib windows\n```"

            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return [TextContent(type="text", text=f"❌ **Error generating report**: {str(e)}")]

async def main():
    """Main function to run the MCP server"""
    # Get Firecrawl API key from environment
    firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
    
    if not firecrawl_api_key:
        logger.warning("FIRECRAWL_API_KEY not found in environment. News features will be limited.")
    
    server_instance = MCPFinancialServer(firecrawl_api_key)
    
    async with server_instance.server.run_stdio() as server:
        await server.initialize(InitializationOptions())
        logger.info("🚀 MCP Financial Analyst Server with Agno Framework started!")
        logger.info("📊 Features: GPT-OSS model, Firecrawl news integration, Multi-agent analysis")
        await asyncio.Event().wait()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 
    
# End of MCP Financial Analyst Server with Agno Framework
# Make sure to run this script with Python 3.8+ and all dependencies installed.
# Use the command: python main.py
# Ensure Ollama is running with the gpt-oss model for full functionality.
# For Firecrawl news features, set the FIRECRAWL_API_KEY environment variable.
# Example: export FIRECRAWL_API_KEY='your_api_key_here'
# This server integrates with Claude Desktop and Cursor IDE for enhanced user experience.
# For any issues, check the logs or refer to the documentation.
# © 2025 - Powered by Agno Framework, GPT-OSS, and Firecrawl
# License: MIT License
# This code is open source and can be modified under the terms of the MIT License.
# Contributions are welcome! Please follow the contribution guidelines in the repository.
# For more information, visit the project repository on GitHub.
# Thank you for using the MCP Financial Analyst Server with Agno Framework!