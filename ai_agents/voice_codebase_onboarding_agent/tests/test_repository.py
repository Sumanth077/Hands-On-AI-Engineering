import tempfile
from pathlib import Path

import pytest

from codebase_guide.repository import LocalRepository, RepositoryError, parse_github_url, rank_paths
from codebase_guide.llm import is_configured
from codebase_guide.fish_audio import narration_script
from codebase_guide import workflow


def test_parse_github_url_accepts_owner_and_repository():
    assert parse_github_url("https://github.com/Sumanth077/Hands-On-AI-Engineering") == (
        "Sumanth077",
        "Hands-On-AI-Engineering",
    )


def test_parse_github_url_rejects_non_repository_url():
    with pytest.raises(RepositoryError):
        parse_github_url("https://example.com/not-a-github-repository")


def test_rank_paths_prioritizes_relevant_entry_point():
    paths = ["docs/guide.md", "src/ui.py", "main.py", "pyproject.toml"]

    selected = rank_paths(paths, "Where is the main application entry point?")

    assert selected[0] == "main.py"


def test_rank_paths_starts_broad_questions_with_root_context():
    paths = ["examples/main.py", "README.md", "pyproject.toml", "src/server.py"]

    selected = rank_paths(paths, "Give me a project architecture overview")

    assert selected[0] == "pyproject.toml"


def test_local_snapshot_skips_virtual_environment_files():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
        root = Path(temporary_directory)
        (root / ".git").mkdir()
        (root / "main.py").write_text("print('hello')", encoding="utf-8")
        (root / ".venv").mkdir()
        (root / ".venv" / "ignored.py").write_text("print('ignore')", encoding="utf-8")

        snapshot = LocalRepository(str(root)).snapshot()

        assert "main.py" in snapshot.file_paths
        assert ".venv/ignored.py" not in snapshot.file_paths


def test_gateway_key_configures_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")

    assert is_configured()


def test_narration_script_removes_markdown_and_source_citations():
    answer = "## Entry point\n\n- Start with **`main.py`** [main.py:12].\n"

    narration = narration_script(answer)

    assert "##" not in narration
    assert "[main.py:12]" not in narration
    assert "Start with main.py." in narration


def test_spoken_brief_requests_conversational_answer(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_chat(messages, *, temperature):
        captured["prompt"] = messages[0]["content"]
        captured["temperature"] = temperature
        return "The project is mainly written in Python."

    monkeypatch.setattr(workflow, "chat", fake_chat)

    spoken = workflow._write_spoken_brief("Which languages are used?", "## Languages\nPython [main.py:1]")

    assert spoken == "The project is mainly written in Python."
    assert "Do not read citations" in str(captured["prompt"])

