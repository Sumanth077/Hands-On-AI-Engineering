"""Application composition. Offline execution needs no credentials and uses only fixture transports."""

import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

from .apify_client import ApifyClient, redact
from .archive import RunArchive
from .config import Settings, require_live_enabled
from .gemini import SELECTED_MODEL, GeminiBackend
from .llm import StructuredModel
from .openai_backend import SELECTED_OPENAI_MODEL, CostLedger, OpenAIBackend
from .research import Researcher
from .research_models import Limits, ResearchResult, Selection

PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"


def configured_limits(candidate_limit: int = 5) -> Limits:
    load_dotenv(dotenv_path=PROJECT_ENV, override=False)
    values = {
        field: os.environ[f"REPOSCOUT_{field.upper()}"]
        for field in Limits.model_fields
        if field != "candidate_limit" and f"REPOSCOUT_{field.upper()}" in os.environ
    }
    return Limits(candidate_limit=candidate_limit, **values)


def live_configuration_ready() -> bool:
    load_dotenv(dotenv_path=PROJECT_ENV, override=False)
    return not live_setup_issues()


def active_provider_label() -> str:
    load_dotenv(dotenv_path=PROJECT_ENV, override=False)
    provider = os.getenv("REPOSCOUT_LLM_PROVIDER", "openai").strip().lower()
    return {"openai": "OpenAI · Responses API", "gemini": "Gemini Developer API"}.get(
        provider, "Provider configuration needs review"
    )


def live_setup_issues() -> list[str]:
    """Return setting NAMES only; never echo unknown model/provider values or keys."""
    load_dotenv(dotenv_path=PROJECT_ENV, override=False)
    issues = []
    provider = os.getenv("REPOSCOUT_LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "gemini"}:
        return ["REPOSCOUT_LLM_PROVIDER must be openai or gemini"]
    required = ["APIFY_TOKEN", "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"]
    for name in required:
        value = os.getenv(name, "").strip()
        if not value or value.startswith("replace_with_"):
            issues.append(name + " missing")
    if (
        provider == "openai"
        and os.getenv("OPENAI_MODEL", SELECTED_OPENAI_MODEL).strip() != SELECTED_OPENAI_MODEL
    ):
        issues.append("OPENAI_MODEL requires verified pricing")
    if provider == "gemini" and os.getenv("GEMINI_MODEL") != SELECTED_MODEL:
        issues.append("GEMINI_MODEL must use the verified model")
    if os.getenv("REPOSCOUT_ENABLE_LIVE", "false").lower() != "true":
        issues.append("REPOSCOUT_ENABLE_LIVE is disabled")
    return issues


async def run_research(
    requirement: str,
    limits: Limits,
    *,
    offline: bool = False,
    progress: Callable[[str, float], None] | None = None,
    resume_result: ResearchResult | None = None,
    selection_cache: Selection | None = None,
) -> ResearchResult:
    if offline:
        # Synthetic fixtures are used only by automated tests.
        from .demo import DemoModel, DemoWeb

        if selection_cache is not None:
            raise ValueError("Cached live selection cannot enter fixture mode")
        if resume_result is not None:
            raise ValueError("Live results cannot be resumed through fixture mode")
        fixture = DemoWeb()
        async with fixture.client() as web:
            result = await Researcher(
                web, StructuredModel(DemoModel(), limits), limits, offline=True, progress=progress
            ).run(requirement)
        result.diagnostics["fixture_peak_concurrency"] = fixture.peak_concurrency
        result.diagnostics["fixture_label"] = "Synthetic overlap check, not a benchmark"
        return result
    load_dotenv(dotenv_path=PROJECT_ENV, override=False)
    require_live_enabled()
    provider = os.getenv("REPOSCOUT_LLM_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", SELECTED_OPENAI_MODEL).strip()
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OpenAI provider setup pending: add OPENAI_API_KEY locally")
        backend = OpenAIBackend(model, SecretStr(key), ledger=CostLedger())
    elif provider == "gemini":
        model, key = (os.getenv(name, "").strip() for name in ("GEMINI_MODEL", "GEMINI_API_KEY"))
        if not all((model, key)):
            raise ValueError(
                "Gemini provider setup pending: add GEMINI_API_KEY and GEMINI_MODEL locally"
            )
        backend = GeminiBackend(model, SecretStr(key))
    else:
        raise ValueError(
            "Unsupported LLM provider; choose openai or gemini explicitly. No fallback."
        )
    settings = Settings.from_env()
    archive = RunArchive([key, settings.token.get_secret_value()])
    backend.recorder = archive.record
    archive.record(
        "started",
        {
            "requirement": requirement,
            "limits": limits.model_dump(),
            "model": model,
            "provider": provider,
            "benchmark": False,
            "account_plan": settings.account_plan,
            "resumed_from": resume_result.diagnostics.get("saved_result_dir")
            if resume_result
            else None,
            "reused_saved_selection": selection_cache.model_dump() if selection_cache else None,
        },
    )
    try:
        async with ApifyClient(settings, recorder=archive.record) as web:
            researcher = Researcher(
                web, StructuredModel(backend, limits), limits, progress=progress
            )
            result = (
                (
                    await researcher.resume(resume_result)
                    if resume_result.projects
                    else await researcher.resume_discovery(
                        resume_result, selection_cache=selection_cache
                    )
                )
                if resume_result
                else await researcher.run(requirement)
            )
    except Exception as exc:
        archive.record("implementation_error", {"type": type(exc).__name__})
        raise
    data = result.model_dump(mode="json")
    # Redact either provider's reflected credential across all results and diagnostics.
    for token in (key, settings.token.get_secret_value()):
        data = redact(data, token)
    return archive.save(ResearchResult.model_validate(data))
