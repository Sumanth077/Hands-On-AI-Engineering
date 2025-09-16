import re
import json
import yfinance as yf
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from agno import Agent, Tool, ToolConfig, ToolOutput, ToolInput, OpenAIConfig
from agno.memory import ConversationBufferMemory
from agno.prompts import FewShotPromptTemplate, PromptTemplate
from agno.chains import LLMChain
from agno.llms import OpenAI
from agno.team import Team
from agno.workflow import Workflow
from agno.models.openai import OpenAIChat 

from dotenv import load_dotenv
load_dotenv()

# define the output structure for the given query
class queryOutput(BaseModel):
    stock_symbol: str = Field(..., description="The stock symbol to query")
    start_date: str = Field(..., description="The start date for the query in YYYY-MM-DD format")
    end_date: str = Field(..., description="The end date for the query in YYYY-MM-DD format")
    metrics: List[str] = Field(..., description="List of financial metrics to retrieve")
    action: str = Field(..., description="The action to perform: 'fetch_data' or 'analyze_data'")
    timeframe: str = Field(..., description="The timeframe for analysis: 'short_term', 'medium_term', or 'long_term'")


# variable to load the LLM model from openai gpt-oss
llm = Agent(
    model=OpenAIChat(model="gpt-oss-20b", temperature=0.2),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    markdown=True
)

# create the agno query parser agent to parse the user query
query_pasrser_agent = Agent(
    name = "Financial Query Parser",
    model =llm.model,
    memory = llm.memory,
    role = "Extract stock details and fetch required data from this user query: {query}."
)

# define the code writer agent to generate python code for data analysis
code_writer_agent = Agent(
    name = "Financial Code Writer",
    model = llm.model,
    memory = llm.memory,
    role = "Generate python code to analyze stock data based on the user query."
)

# define the code interpreter agent to execute the generated code
code_interpreter_agent = Agent(
    name = "Financial Code Interpreter",
    model = llm.model,
    memory = llm.memory,
    role = "Execute the generated python code and return the results."
)

# develop research team of agents to tackle the user query and generate the final response\
financial_research_team = Team(
    name = "Financial Research Team",
    agents = [query_pasrser_agent, code_writer_agent],
    instructions = "You are a team of financial analysts. Your goal is to analyze stock data based on user queries and provide insights."
)   

# define the workflow to handle the user query and generate the final response
financial_workflow = Workflow(
    name = "Financial Analysis Workflow",
    description = "Workflow to analyze stock data based on user queries. Review and execute the generated Python code by code writer agent to visualize stock data and fix any errors encountered.",
    steps = [financial_research_team, code_interpreter_agent]
)   

# define the funtion to take user query and generate the final response using the workflow
def analyze_stock_data(user_query: str) -> str:
    response = financial_workflow.run(query=user_query)
    return response['output']

if __name__ == "__main__":
    user_query = "Analyze the stock performance of Apple Inc. (AAPL) from January 1, 2022 to December 31, 2022. Provide insights on its closing prices and volume trends. Suggest whether it's a good buy for short-term investment."
    final_response = analyze_stock_data(user_query)
    print(final_response)   

# Example user query: "Analyze the stock performance of Apple Inc. (AAPL) from January 1, 2022 to December 31, 2022. Provide insights on its closing prices and volume trends. Suggest whether it's a good buy for short-term investment."
# Example user query: "Analyze the stock performance of Tesla Inc. (TSLA) from June 1, 2021 to June 1, 2022. Provide insights on its opening prices and market trends. Suggest whether it's a good buy for medium-term investment."
# Example user query: "Analyze the stock performance of Microsoft Corp. (MSFT) from January 1, 2020 to January 1, 2023. Provide insights on its high and low prices. Suggest whether it's a good buy for long-term investment.