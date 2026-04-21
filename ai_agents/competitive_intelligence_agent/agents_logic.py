import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import TavilySearchTool

# Gemma 4 via LiteLLM/Gemini API
gemma_llm = LLM(
    model="gemini/gemma-4-26b-a4b-it",
    api_key=os.getenv("GEMINI_API_KEY")
)

def get_research_crew(my_company, competitor, pain_point, goal):
    # Specialized Tools
    search_tool = TavilySearchTool(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)

    # 1. Market Scout: Focuses only on the competitor's strategy vs yours
    scout = Agent(
        role="Market Analyst",
        goal="Identify {competitor}'s current market moves and positioning relative to {my_company}.",
        backstory="You are a market expert. You ignore noise and focus on strategic shifts.",
        llm=gemma_llm, tools=[search_tool], verbose=True
    )

    # 2. Product Strategist: Focuses only on the pain point
    strategist = Agent(
        role="Product Strategist",
        goal="Compare {competitor} features against {my_company} regarding {pain_point}.",
        backstory="You are a technical product strategist. You find the technical gaps that lose deals.",
        llm=gemma_llm, tools=[search_tool], verbose=True
    )

    # 3. Battlecard Author: Synthesizes everything into a "Kill Switch"
    author = Agent(
        role="Battlecard Author",
        goal="Write a punchy, aggressive sales battlecard to help us achieve: {goal}.",
        backstory="You are a sales coach. You provide arguments that win deals. Use plain, human language.",
        llm=gemma_llm, verbose=True
    )

    # Defined Task Workflow
    tasks = [
        Task(
            description=f"Analyze {competitor}. Explain how their market presence challenges {my_company}.",
            expected_output="Positioning summary.", 
            agent=scout
        ),
        Task(
            description=f"Compare {competitor} to {my_company}. Why is {competitor} winning or losing on the issue of: {pain_point}?",
            expected_output="Feature gap analysis.", 
            agent=strategist
        ),
        Task(
            description=f"Draft a Battlecard for {my_company} to help us achieve: {goal}. Focus on beating {competitor}.",
            expected_output="Markdown Battlecard.", 
            agent=author
        )
    ]
    
    return Crew(agents=[scout, strategist, author], tasks=tasks, process=Process.sequential, verbose=True)