# define deep research agent for the agentic application using agno, firecrawl

#importing tools
import agno.agent import Agent 
import agno.models.nebius import Nebius
from dotenv import load_dotenv
from typing import List, Dict, Any, Iterator
from agno.utils.log import logger
from agno.utils.pprint import prrint_run_response
from agno.tools.firecrawl import FirecrawlTools
from agno.workflow import Workflow, RunEvent, RunResponse
from pydantic import BaseModel, Field
import os

load_dotenv()

# define the deep research agent class with agno Workflow
class DeepResearchAgent(Workflow):
    """
    Deep Research Agent for in-depth financial analysis and research
    steps will include data collection and gathering, data analysis to generate insights,
    and detailed report generation using agno agents workflow.
    """

    # search the web from the user query
    searcher: Agent = Agent(
        tools=[FirecrawlTools(api_key=os.getenv("FIRECRAWL_API_KEY"))],
        model= Nebius(
            id="deepseek-ai/DeepSeek-V3-0324", api_key=os.getenv("NEBIUS_API_KEY")),
        show_progress=True,
        show_tool_calls=True,
        markdown=True,
        description=(
            "A deep research agent for in-depth financial analysis and research."
            " It collects data, analyzes it to generate insights, and creates detailed reports."
            " Genuine, reliable, diverse sources of information."
        ),
        Instructions=(
            "1. Start with a clear research question or objective."
            "2. Use the search tools to gather data from various sources."
            "3. Analyze the collected data to identify trends and insights."
            "4. Compile the findings with proper statistics, reports, charts and tables."
            "5. Organize your findings in a clear, structured format (e.g., markdown table or sections by source type).\n"
            "6. If the topic is ambiguous, clarify with the user before proceeding.\n"
            "7. Be as comprehensive and verbose as possible—err on the side of including more detail.\n"
            "8. Mention the References & Sources of the Content. (It's Must)"
        ),
    )

    # analyst: Agent that dissects and finds the insights from the retrieved information
    analyst: Agent = Agent(
        model= Nebius(
            id="deepseek-ai/DeepSeek-V3-0324", api_key=os.getenv("NEBIUS_API_KEY")),
        show_progress=True,
        show_tool_calls=True,
        markdown=True,
        description=(
             "You are AnalystBot-X, a critical thinker who synthesizes research findings "
            "into actionable insights. Your job is to analyze, compare, and interpret the "
            "information provided by the researcher."
        ),
        Instructions=(
            "1. Identify key themes, trends, and contradictions in the research.\n"
            "2. Highlight the most important findings and their implications.\n"
            "3. Suggest areas for further investigation if gaps are found.\n"
            "4. Present your analysis in a structured, easy-to-read format.\n"
            "5. Extract and list ONLY the reference links or sources that were ACTUALLY found and provided by the researcher in their findings. Do NOT create, invent, or hallucinate any links.\n"
            "6. If no links were provided by the researcher, do not include a References section.\n"
            "7. Don't add hallucinations or make up information. Use ONLY the links that were explicitly passed to you by the researcher.\n"
            "8. Verify that each link you include was actually present in the researcher's findings before listing it.\n"
            "9. If there's no Link found from the previous agent then just say, No reference Found."
        ),
    )

# writer: produce polished reports and summaries based on the research findings
    writer: Agent = Agent(
        model= Nebius(
            id="deepseek-ai/DeepSeek-V3-0324", api_key=os.getenv("NEBIUS_API_KEY")),
        show_progress=True,
        show_tool_calls=True,
        markdown=True,
        description=(
            "You are WriterBot-X, a skilled communicator who transforms research findings "
            "into clear, engaging reports. Your job is to write, edit, and polish the content "
            "produced by the researcher and analyst."
        ),
        Instructions=(
            "1. Start with a clear understanding of the research question and objectives.\n"
            "2. Use the findings from the researcher and analyst to create a cohesive narrative.\n"
            "3. Include relevant statistics, quotes, and examples to support your points.\n"
            "4. Organize the report in a logical structure with headings and subheadings.\n"
            "5. Edit for clarity, conciseness, and coherence.\n"
            "6. Ensure proper citation of all sources and references.\n"
            "7. Use a professional tone and style appropriate for the target audience.\n"
            "8. Review and revise the report based on feedback from the team."
        ),
    )

    def run(self, topic: str)  -> Iterator[RunResponse]:
        """
        Run the deep research agent workflow with the given topic.
        This method orchestrates the search, analysis, and writing processes.
        """
        logger.info(f"Starting deep research on topic: {topic}")

        # Step 1: Search for information
        search_response = self.searcher.run(topic)
        prrint_run_response(search_response)

        # Step 2: Analyze the findings
        analysis_response = self.analyst.run(search_response)
        prrint_run_response(analysis_response)

        # Step 3: Write the final report
        write_response = self.writer.run(analysis_response)
        prrint_run_response(write_response)

        yield from write_response

def final_research(query: str) -> str:
    research_agent = DeepResearchAgent()
    results = research_agent.run(query)

    logger.info(f"Deep research completed for query: {query}")

    # collect the final report into single string
    full_report = ""
    for response in results:
        if response.content:
            full_report += response.content

    return full_report


if __name__ == "__main__":
    topic = "Give the detailed guide on how to work with Git/GitHub"
    final_report = final_research(topic)
    print(final_report)
