from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_nebius import ChatNebius

from .config import Settings, configure_tracing
from .models import ResearchResult, ReviewResult, Source
from .review import JevReviewer
from .validation import validate_citations

ProgressCallback = Callable[[str, str], None]


class TavilyCollector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sources: list[Source] = []
        self.searches: list[str] = []
        self._seen_urls: set[str] = set()

    def search(self, query: str) -> str:
        self.searches.append(query)
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "search_depth": self.settings.tavily_search_depth,
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=45.0,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        result_lines: list[str] = []
        for result in results:
            url = str(result.get("url", ""))
            if not url or url in self._seen_urls:
                continue
            self._seen_urls.add(url)
            source = Source(
                number=len(self.sources) + 1,
                title=str(result.get("title") or url),
                url=url,
                snippet=str(result.get("content", "")),
                published_date=result.get("published_date"),
            )
            self.sources.append(source)
            result_lines.append(
                f"[{source.number}] {source.title}\nURL: {source.url}\nEvidence: {source.snippet}"
            )
        return "\n\n".join(result_lines) or "No usable results were returned."


def _extract_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return "The research agent did not return a final report."


def _extract_plan(result: dict[str, Any]) -> list[str]:
    plan: list[str] = []
    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", []) or []:
            if call.get("name") in {"write_todos", "write_todo"}:
                todos = call.get("args", {}).get("todos", [])
                for todo in todos:
                    item = todo.get("content") if isinstance(todo, dict) else str(todo)
                    if item and item not in plan:
                        plan.append(item)
    return plan


class DeepResearchWorkflow:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _run_research(
        self, question: str, revision_instruction: str = ""
    ) -> tuple[str, list[Source], list[str], list[str]]:
        configure_tracing(self.settings)
        collector = TavilyCollector(self.settings)

        @tool
        def search_web(query: str) -> str:
            """Search the live web using Tavily and return numbered evidence sources."""
            try:
                return collector.search(query)
            except httpx.HTTPError as error:
                return f"Search failed: {error}"

        model = ChatNebius(
            model=self.settings.nebius_model,
            api_key=self.settings.nebius_api_key,
            base_url=self.settings.nebius_base_url,
            temperature=0,
        )
        agent = create_deep_agent(model=model, tools=[search_web])
        prompt = f"""Research the user's question using the search_web tool before answering.

User question: {question}

Today is {date.today().isoformat()}. For a time-sensitive question, search explicitly for current information and inspect publication dates. Do not call a past year "current" unless the sources establish that context. If the available evidence is older, say so plainly instead of presenting it as the latest state.

Return only the finished report for the user in Markdown. Do not reveal your planning, tool use, scratch work, or phrases such as "let me synthesize". Do not address the user as an internal research task.

Every material factual claim must include one or more inline numbered citations such as [1] or [2]. Only cite source numbers returned by search_web. Do not invent URLs, citations, or facts. Prefer primary and recent sources where appropriate.
{revision_instruction}
"""
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return _extract_text(result), collector.sources, _extract_plan(result), collector.searches

    def run(self, question: str, progress: ProgressCallback | None = None) -> ResearchResult:
        def update(label: str, detail: str) -> None:
            if progress:
                progress(label, detail)

        update("Planning and researching", "Creating a plan and searching Tavily.")
        answer, sources, plan, searches = self._run_research(question)
        citation_issues = validate_citations(answer, sources)

        update("Reviewing evidence", "JEV is checking evidence coverage and claim grounding.")
        reviewer = JevReviewer(self.settings)
        review = reviewer.review(question, answer, sources, citation_issues)
        revision_count = 0

        if review.needs_revision:
            revision_count = 1
            update("Researching again", "The review found a gap, so the agent is gathering stronger evidence.")
            instruction = (
                "The first review found weak evidence or unsupported claims. Search again, "
                "resolve the gap, and return a revised report with valid numbered citations."
            )
            answer, sources, new_plan, new_searches = self._run_research(question, instruction)
            plan.extend(item for item in new_plan if item not in plan)
            searches.extend(new_searches)
            citation_issues = validate_citations(answer, sources)
            update("Reviewing revised report", "JEV is checking the revised evidence one final time.")
            review = reviewer.review(question, answer, sources, citation_issues)

        return ResearchResult(
            question=question,
            answer=answer,
            sources=sources,
            plan=plan,
            searches=searches,
            review=review,
            revision_count=revision_count,
            citation_issues=citation_issues,
        )

