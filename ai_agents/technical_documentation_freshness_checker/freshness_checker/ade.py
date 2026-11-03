from __future__ import annotations

import os
from pathlib import Path

from landingai_ade import LandingAIADE

from freshness_checker.models import TechnicalClaim, TechnicalClaimSet


class ADEError(RuntimeError):
    pass


def _client() -> LandingAIADE:
    environment = os.getenv("LANDINGAI_ENVIRONMENT", "us").strip().lower()
    # The ADE client uses the US endpoint by default. Its `environment` argument
    # is only needed for an EU-region key.
    if environment == "eu":
        return LandingAIADE(environment="eu")
    return LandingAIADE()


def parse_and_extract(document_path: Path) -> tuple[str, list[TechnicalClaim], dict]:
    """Parse a PDF, then extract only claims that need live-web verification."""
    client = _client()
    parse_job = client.v2.parse_jobs.create(
        document=document_path,
        model="dpt-3-pro-latest",
        service_tier="standard",
    )
    parsed = client.v2.parse_jobs.wait(parse_job.job_id, timeout=600, raise_on_failure=True)
    markdown = parsed.result.markdown

    extract_job = client.v2.extract_jobs.create(
        markdown=markdown,
        schema=TechnicalClaimSet,
        service_tier="standard",
    )
    extracted = client.v2.extract_jobs.wait(
        extract_job.job_id, timeout=600, raise_on_failure=True
    )

    payload = extracted.result.extraction
    if isinstance(payload, str):
        claim_set = TechnicalClaimSet.model_validate_json(payload)
    else:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        claim_set = TechnicalClaimSet.model_validate(payload)

    metadata = extracted.result.extraction_metadata
    if hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump(mode="json")
    return markdown, claim_set.claims, metadata or {}
