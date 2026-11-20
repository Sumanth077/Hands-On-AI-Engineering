"""Streamlit app that runs a multi-agent CrewAI pipeline to analyze a stock and generate a markdown investment report."""
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel
from tools.financial_tools import MarketDataUnavailable, YFinanceStockTool

# Load environment variables
load_dotenv()

# Define Pydantic models for structured output
class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    current_price: float
    market_cap: float
    pe_ratio: float
    recommendation: str
    analysis_summary: str
    risk_assessment: str
    technical_indicators: dict
    fundamental_metrics: dict

# Initialize Orq.ai LLM
@st.cache_resource
def load_llm():
    """Create and cache the Orq.ai LLM client used by the agents."""
    return LLM(
        model="openai/alibaba/qwen3.6-flash",
        base_url="https://api.orq.ai/v3/router",
        api_key=os.getenv("ORQ_API_KEY"),
        temperature=0.3
    )

# Create agents and tasks
def create_agents_and_tasks(symbol: str):
    """Build the stock analysis and report writing agents, their tasks, and the crew that runs them for the given symbol."""
    market_data = YFinanceStockTool().fetch_data(symbol)
    market_data_json = json.dumps(market_data, indent=2)
    llm = load_llm()
    
    # Stock Analysis Agent
    stock_analysis_agent = Agent(
        role="Wall Street Financial Analyst",
        goal=f"Conduct a comprehensive, data-driven analysis of {symbol} stock using the latest available Yahoo Finance data",
        backstory="""You are a seasoned Wall Street analyst with 15+ years of experience in equity research.
                     You're known for your meticulous analysis and data-driven insights.
                     You base price and financial claims only on the source data supplied with the task, never on pre-existing knowledge.
                     You're an expert at interpreting financial metrics, market trends, and providing actionable insights.""",
        llm=llm,
        verbose=True,
        memory=False
    )

    # Report Writing Agent
    report_writer_agent = Agent(
        role="Financial Report Specialist",
        goal="Transform detailed financial analysis into a professional, comprehensive investment report",
        backstory="""You are an expert financial writer with a track record of creating institutional-grade research reports.
                     You excel at presenting complex financial data in a clear, structured format.
                     You always maintain professional standards while making reports accessible and actionable.
                     You're known for your clear data presentation, trend analysis, and risk assessment capabilities.""",
        llm=llm,
        verbose=True
    )

    # Analysis Task
    analysis_task = Task(
        description=f"""Analyze {symbol} using ONLY the successful source-data payload fetched by the application before the LLM is called. Do not use prior model knowledge to fill missing values. If a value is N/A or absent, label it unavailable.

        Source data:
        {market_data_json}

        Your analysis must include:

        1. Latest Trading Information (HIGHEST PRIORITY)
           - Latest stock price with specific date
           - Percentage change
           - Trading volume
           - Market status (open/closed)
           - Highlight if this is from the most recent trading session

        2. 52-Week Performance (CRITICAL)
           - 52-week high with exact date
           - 52-week low with exact date
           - Current price position relative to 52-week range
           - Calculate percentage from highs and lows

        3. Financial Deep Dive
           - Market capitalization
           - P/E ratio and other key metrics
           - Revenue growth and profit margins
           - Dividend information (if applicable)

        4. Technical Analysis
           - Recent price movements
           - Volume analysis
           - Key technical indicators

        5. Market Context
           - Business summary
           - Analyst recommendations
           - Key risk factors

        IMPORTANT: 
        - Begin your analysis with the latest price and 52-week data
        - Include specific dates for all price points
        - Clearly indicate when each price point was recorded
        - Calculate and show percentage changes
        - Use only the supplied source data for price and financial figures
        - Compare current metrics with historical trends""",
        expected_output="A comprehensive analysis report using the supplied source data and its trading date, with clear section breakdowns",
        agent=stock_analysis_agent
    )

    # Report Task
    report_task = Task(
        description=f"""Transform the analysis into a professional investment report for {symbol}. The report must:

        1. Structure:
           - Begin with an executive summary
           - Use clear section headers
           - Include tables for data presentation
           - Add emoji indicators for trends (📈 📉)

        2. Content Requirements:
           - Include timestamps for all data points
           - Present key metrics in tables
           - Use bullet points for key insights
           - Compare metrics to industry averages
           - Explain technical terms
           - Highlight potential risks

        3. Sections:
           - Executive Summary
           - Market Position Overview
           - Financial Metrics Analysis
           - Technical Analysis
           - Risk Assessment
           - Future Outlook

        4. Formatting:
           - Use markdown formatting
           - Create tables for data comparison
           - Include trend emojis
           - Use bold for key metrics
           - Add bullet points for key takeaways

        IMPORTANT:
        - Maintain professional tone
        - Clearly state all data sources
        - Include risk disclaimers
        - Format in clean, readable markdown
        - Do not invent or supplement prices, dates, metrics, or recommendations from model memory
        - If the source-backed analysis marks a value unavailable, keep it unavailable""",
        expected_output="A professionally formatted investment report in markdown, with clear sections, data tables, and visual indicators",
        agent=report_writer_agent
    )

    # Create Crew
    crew = Crew(
        agents=[stock_analysis_agent, report_writer_agent],
        tasks=[analysis_task, report_task],
        process=Process.sequential,
        verbose=True
    )

    return crew

# Streamlit UI
st.set_page_config(page_title="Multi-Agent AI Financial Analyst", layout="wide")

st.title("🎯 Multi-Agent AI Financial Analyst")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # Stock Symbol input
    symbol = st.text_input(
        "Stock Symbol",
        value="",
        help="Enter a stock symbol (for example: TSLA or MSFT). Provide a single ticker symbol to analyze."
    ).upper()

    # Analysis button
    analyze_button = st.button("Analyze Stock", type="primary")

# Main content area
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.report = None

if analyze_button:
    try:
        if not symbol:
            st.error("Please enter a valid stock symbol before running the analysis.")
        else:
            with st.spinner(f'Analyzing {symbol}... This may take a few minutes.'):
                # Create and run the crew
                crew = create_agents_and_tasks(symbol)
                result = crew.kickoff()
            # Convert the CrewOutput to string if needed
            if hasattr(result, 'raw'):
                st.session_state.report = result.raw
            else:
                st.session_state.report = str(result)
            st.session_state.analysis_complete = True

    except MarketDataUnavailable as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if st.session_state.analysis_complete and st.session_state.report:
    st.markdown("### Analysis Report")
    st.markdown(st.session_state.report)
    
    # Download button
    st.download_button(
        label="Download Report",
        data=st.session_state.report,
        file_name=f"stock_analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )

# Footer
st.markdown("---") 
