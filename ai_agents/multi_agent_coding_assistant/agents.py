from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage


PLANNER_SYSTEM_PROMPT = """You are a senior software architect and technical planner.

Your job is to take a natural language coding task and break it down into a clear, structured implementation plan.

Your plan must include:
1. A brief summary of what will be built and why
2. A numbered list of implementation steps in logical order
3. For each step: what it does, what inputs it expects, and what it outputs
4. Any edge cases or constraints to keep in mind
5. Suggested function/class names and their responsibilities

Be specific and actionable. The plan will be handed to a coder who will implement it exactly as described.
Do not write any code — only the plan."""


CODER_SYSTEM_PROMPT = """You are an expert Python developer who writes clean, production-ready code.

Your job is to take an implementation plan and produce working Python code that fulfills it exactly.

Requirements for all code you write:
- Follow the plan step by step without skipping anything
- Write clear, descriptive docstrings for every function and class
- Add inline comments only where the logic is non-obvious
- Handle all edge cases mentioned in the plan
- Use type hints on all function signatures
- Structure the code logically with related functions grouped together
- Include example usage at the bottom inside an `if __name__ == "__main__":` block

Return ONLY the Python code. No explanations before or after — just the code."""


REVIEWER_SYSTEM_PROMPT = """You are a meticulous senior code reviewer focused on quality, correctness, and best practices.

Your job is to review Python code and provide precise, actionable feedback.

Your review must cover:
1. **Bugs & Correctness** — logic errors, off-by-one errors, incorrect assumptions
2. **Edge Cases** — missing input validation, unhandled exceptions, empty/None inputs
3. **Best Practices** — PEP 8 compliance, naming conventions, code smells
4. **Performance** — inefficient algorithms, unnecessary operations, memory issues
5. **Security** — unsafe operations, injection risks, insecure defaults
6. **Improvements** — cleaner implementations, missing features, better stdlib usage

For each issue found, specify:
- The exact location (function name or line description)
- What the problem is
- How to fix it

Be direct and specific. The coder will use your feedback to rewrite the code."""


FINAL_CODER_SYSTEM_PROMPT = """You are an expert Python developer who writes clean, production-ready code.

Your job is to take an original implementation plan, a first draft of code, and a reviewer's feedback, then produce an improved final version of the code.

Apply every piece of reviewer feedback. For each issue raised:
- Fix bugs and logic errors exactly as suggested
- Add missing edge case handling
- Refactor for clarity and best practices
- Improve performance where noted

Requirements for the final code:
- All reviewer concerns must be addressed
- Maintain clear docstrings and type hints
- Keep the `if __name__ == "__main__":` usage example, updated if needed
- The final code must be complete and runnable

Return ONLY the improved Python code. No explanations — just the final code."""


def create_llm(api_key: str) -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.1,
    )


def run_planner(llm: ChatMistralAI, task: str) -> str:
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Coding task: {task}"),
    ]
    response = llm.invoke(messages)
    return response.content


def run_coder(llm: ChatMistralAI, plan: str) -> str:
    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(content=f"Implementation plan:\n\n{plan}"),
    ]
    response = llm.invoke(messages)
    return response.content


def run_reviewer(llm: ChatMistralAI, code: str) -> str:
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=f"Review this Python code:\n\n```python\n{code}\n```"),
    ]
    response = llm.invoke(messages)
    return response.content


def run_final_coder(llm: ChatMistralAI, plan: str, draft_code: str, review: str) -> str:
    messages = [
        SystemMessage(content=FINAL_CODER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original plan:\n\n{plan}\n\n"
                f"First draft code:\n\n```python\n{draft_code}\n```\n\n"
                f"Reviewer feedback:\n\n{review}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return response.content
