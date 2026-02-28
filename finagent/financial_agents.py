import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv, find_dotenv

# Load environment variables from root .env
load_dotenv(find_dotenv())

def get_financial_analyst_agent(api_key: str):
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=api_key),
        tools=[YFinanceTools(
            stock_price=False,        # Slow: calls yfinance.info
            technical_indicators=False, # Redundant: calls history() again
            historical_prices=True,     # Fast: calls history()
            company_info=False,
            analyst_recommendations=False
        )],
        instructions=[
            "You are a senior financial analyst team.",
            "Use 'get_historical_stock_prices' to fetch recent price data.",
            "The most recent trading price is the last entry in the historical data.",
            "Analyze the trends and provide a CONCISE, professional report.",
            "Focus on the most important technical levels and risk factors.",
            "Always state a disclaimer that this is not financial advice."
        ],
        show_tool_calls=False,
        markdown=True
    )

def run_financial_analysis(query: str) -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "Error: OPENAI_API_KEY environment variable is not set."
    
    agent = get_financial_analyst_agent(openai_api_key)
    try:
        # Run the agent and collect response
        response = agent.run(query)
        return response.content
    except Exception as e:
        return f"Error during analysis: {e}"
