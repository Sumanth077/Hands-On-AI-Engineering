"""Offline UI checks: live-only submissions and read-only real-session loading."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from repo_scout import service
from repo_scout.research_models import ResearchResult

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def app_path(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "false")
    monkeypatch.delenv("REPOSCOUT_RESTORE_RESULT", raising=False)
    path = tmp_path / "streamlit_app.py"
    path.write_text(APP.read_text(encoding="utf-8"), encoding="utf-8")

    async def forbidden(*args, **kwargs):
        raise AssertionError("Viewing the UI must never start research")

    monkeypatch.setattr(service, "run_research", forbidden)
    return path


def saved_result(offline=False):
    return ResearchResult(
        requirement="Saved repository research",
        offline=offline,
        created_at=datetime.now(timezone.utc),
        status="complete",
    )


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_ui_has_no_sample_mode_or_automatic_research(app_path):
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert not app.radio
    assert app.text_area[0].value == ""
    assert button(app, "Start research").disabled
    assert not app.get("download_button")
    assert not any("sample data" in element.value.lower() for element in app.warning)
    app.text_area[0].set_value("A requirement not submitted").run()
    assert not app.exception
    assert "research_result" not in app.session_state


def test_explicit_live_submission_and_reruns_preserve_result(app_path, monkeypatch):
    calls = []
    monkeypatch.setattr(service, "live_configuration_ready", lambda: True)

    async def live(*args, **kwargs):
        assert kwargs["offline"] is False
        calls.append(args[0])
        return saved_result()

    monkeypatch.setattr(service, "run_research", live)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.text_area[0].set_value("Find projects with required capabilities")
    button(app, "Start research").click().run()
    assert not app.exception
    assert calls == ["Find projects with required capabilities"]
    assert len(app.get("download_button")) == 2
    saved = app.session_state["research_result"]
    app.run()
    app.text_area[0].set_value("Changed, not submitted").run()
    assert len(calls) == 1
    assert app.session_state["research_result"] == saved


def test_saved_real_session_loads_without_live_credentials_and_filters_fixtures(app_path):
    folder = app_path.parent / "outputs"
    for name, offline in [("live-real", False), ("live-fixture", True)]:
        path = folder / name / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(saved_result(offline).model_dump_json(), encoding="utf-8")
    real = folder / "live-real" / "result.json"
    before = {p: p.read_bytes() for p in folder.glob("*/result.json")}
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert app.selectbox[0].options == ["live-real"]
    app.selectbox[0].set_value(str(real)).run()
    button(app, "Open saved session").click().run()
    assert not app.exception
    assert button(app, "Start research").disabled
    assert app.session_state["research_result"]["offline"] is False
    assert len(app.get("download_button")) == 2
    app.run()
    assert not app.exception
    assert all(p.read_bytes() == content for p, content in before.items())


def test_legacy_fixture_state_and_fixture_restore_are_not_displayed(app_path, monkeypatch):
    path = app_path.parent / "outputs" / "live-fixture" / "result.json"
    path.parent.mkdir(parents=True)
    fixture = saved_result(True).model_dump(mode="json")
    path.write_text(saved_result(True).model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("REPOSCOUT_RESTORE_RESULT", str(path))
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.session_state["research_result"] = fixture
    app.run()
    assert not app.exception
    assert "research_result" not in app.session_state
    assert not app.get("download_button")
    assert path.exists()


def test_real_restore_hook_still_works(app_path, monkeypatch):
    path = app_path.parent / "outputs" / "live-real" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(saved_result().model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("REPOSCOUT_RESTORE_RESULT", str(path))
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.session_state["research_result"]["offline"] is False
    assert len(app.get("download_button")) == 2


def test_evidence_previews_use_markdown_and_keep_original_raw_source(app_path):
    from repo_scout.research_models import (
        Assessment,
        Candidate,
        Citation,
        Criterion,
        Interpretation,
        Project,
        Source,
    )

    raw = '# Before\n```python\nbefore()\nprint("ok")\n```\n## Heading\n**Bold** [guide](guide.md)\n![badge](https://example.org/badge.svg)\n'
    excerpt = raw[raw.index("print") :]
    result = saved_result()
    result.interpretation = Interpretation(
        summary="A saved finding",
        queries=["saved"],
        criteria=[Criterion(id="local", label="Local use", description="Local use", required=True)],
    )
    repo = "https://github.com/example/saved"
    result.sources = [
        Source(
            id="s1",
            repo_url=repo,
            url=repo,
            role="repository",
            retrieval_timestamp=result.created_at,
            outcome="success",
            markdown=raw,
        )
    ]
    result.projects = [
        Project(
            candidate=Candidate(repo_url=repo, reason="Saved lead"),
            name="Saved project",
            assessments=[
                Assessment(
                    criterion_id="local",
                    verdict="Matches",
                    explanation="A saved finding",
                    basis="documented",
                    citations=[Citation(source_id="s1", excerpt=excerpt)],
                )
            ],
        )
    ]
    path = app_path.parent / "outputs" / "live-evidence" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.selectbox[0].set_value(str(path)).run()
    button(app, "Open saved session").click().run()
    assert not app.exception
    preview = next(item for item in app.markdown if "## Heading" in item.value)
    assert '```python\nprint("ok")\n```' in preview.value
    assert "**Bold** [guide](<https://github.com/example/guide.md>)" in preview.value
    assert "![badge]" not in preview.value
    assert any(item.label == "View raw source" and not item.proto.expanded for item in app.expander)
    assert raw.strip() in [item.value for item in app.code]
    assert excerpt.strip() in [item.value for item in app.code]
    assert app.session_state["research_result"]["sources"][0]["markdown"] == raw
    assert (
        app.session_state["research_result"]["projects"][0]["assessments"][0]["citations"][0][
            "excerpt"
        ]
        == excerpt
    )
