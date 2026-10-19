"""LangGraph workflow that plans a targeted, read-only repository investigation."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from codebase_guide.llm import LLMError, chat, json_object
from codebase_guide.models import GuideAnswer, GuideState, RepositorySnapshot, SourceFile
from codebase_guide.repository import rank_paths


def _tree_preview(paths: list[str]) -> str:
    return "\n".join(f"- {path}" for path in paths[:900])


def _numbered(file: SourceFile) -> str:
    lines = file.content.splitlines()
    trimmed = lines[:500]
    return "\n".join(f"{index + 1}: {line}" for index, line in enumerate(trimmed))


def _select_paths(question: str, snapshot: RepositorySnapshot) -> tuple[str, list[str]]:
    prompt = f"""You are planning a read-only investigation of a software repository.
Question: {question}
Repository: {snapshot.name}
Description: {snapshot.description}
Default branch: {snapshot.default_branch}
Available files:\n{_tree_preview(snapshot.file_paths)}

Return JSON only: {{"plan": "one sentence", "paths": ["path/to/file"]}}.
Choose at most 6 paths. Every selected path must appear exactly in Available files.
Start with manifests, entry points, routes, or tests only when relevant to the question."""
    try:
        data = json_object(chat([{"role": "user", "content": prompt}]))
        proposed = [str(item) for item in data.get("paths", [])]
        selected = [path for path in proposed if path in snapshot.file_paths][:6]
        if selected:
            return str(data.get("plan", "Inspecting the files most relevant to the question.")), selected
    except LLMError:
        pass
    return "Using a deterministic file selection because model planning was unavailable.", rank_paths(
        snapshot.file_paths, question
    )


def _write_answer(question: str, snapshot: RepositorySnapshot, files: list[SourceFile]) -> str:
    evidence = "\n\n".join(f"FILE: {file.path}\n{_numbered(file)}" for file in files)
    prompt = f"""You are a careful codebase onboarding guide. Answer this question about the repository:
{question}

Repository: {snapshot.name}
Description: {snapshot.description}

Use only the supplied source files. Do not infer behaviour from a filename alone. If the evidence does not answer the question, say what is unknown.
Write a concise answer for a developer. Every factual statement about implementation needs one or more inline citations in exactly this format: [path:line]. Cite only a supplied file and valid line number.

SOURCE FILES:
{evidence}"""
    return chat([{"role": "user", "content": prompt}], temperature=0.15)


def _write_spoken_brief(question: str, answer: str) -> str:
    prompt = f"""Turn this codebase answer into a natural spoken response for a developer.

Question: {question}
Written answer: {answer}

Rules:
- Use only facts stated in the written answer. Do not add information.
- Answer the question directly in two to four short sentences, under 90 words.
- Keep the important language, framework, or architecture conclusion when relevant.
- Do not read citations, Markdown, headings, bullet points, URLs, file paths, line numbers, code, or a complete list of dependencies.
- Use normal spoken English. For example, say “The project is mainly written in Python” rather than reciting file extensions.
- Return only the words to be spoken."""
    return chat([{"role": "user", "content": prompt}], temperature=0.2)


def build_graph(reader):
    def plan(state: GuideState) -> GuideState:
        plan_text, paths = _select_paths(state["question"], state["snapshot"])
        return {"plan": plan_text, "selected_paths": paths}

    def collect(state: GuideState) -> GuideState:
        snapshot = state["snapshot"]
        files = reader.read_files(state["selected_paths"], snapshot.default_branch)
        return {"files": files}

    def answer(state: GuideState) -> GuideState:
        files = state["files"]
        if not files:
            return {"answer": "I could not read the selected files. Try another repository or a more specific question."}
        return {"answer": _write_answer(state["question"], state["snapshot"], files)}

    graph = StateGraph(GuideState)
    graph.add_node("plan", plan)
    graph.add_node("collect", collect)
    graph.add_node("answer", answer)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "collect")
    graph.add_edge("collect", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def investigate(reader, snapshot: RepositorySnapshot, question: str) -> GuideAnswer:
    result = build_graph(reader).invoke({"question": question, "snapshot": snapshot})
    answer = result["answer"]
    try:
        spoken_answer = _write_spoken_brief(question, answer)
    except LLMError:
        spoken_answer = answer
    return GuideAnswer(
        question=question,
        answer=answer,
        spoken_answer=spoken_answer,
        files=result.get("files", []),
        selected_paths=result.get("selected_paths", []),
        plan=result.get("plan", ""),
    )

