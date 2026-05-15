import re
import time
from collections.abc import Callable

import httpx

from nvidia_client import NvidiaAPIError, chat_completion

SECTION_NAMES = ("COMPANY RESEARCH", "ROLE REQUIREMENTS", "JOB POSTING")
MAX_OUTPUT_TOKENS = 1500


class PipelineResult:
    def __init__(
        self,
        company_research: str,
        role_requirements: str,
        job_posting: str,
    ):
        self.company_research = company_research
        self.role_requirements = role_requirements
        self.job_posting = job_posting


def _build_prompt(company: str, role: str) -> str:
    return f"""
You are a job posting expert. Complete these three tasks for {company} hiring a {role}:

1. COMPANY RESEARCH: Based on your knowledge of {company}, describe their culture, values, and work environment in 2-3 sentences.

2. ROLE REQUIREMENTS: List the top 5 technical skills and 3 soft skills required for a {role} at {company}.

3. JOB POSTING: Write a complete professional job posting for a {role} at {company} including: job summary, responsibilities (5 bullet points), requirements (5 bullet points), and how to apply.

Format each section clearly with the headers: COMPANY RESEARCH, ROLE REQUIREMENTS, JOB POSTING.
""".strip()


def _normalize_header(line: str) -> str | None:
    cleaned = line.strip().upper()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"^\*+|\*+$", "", cleaned).strip()
    cleaned = cleaned.rstrip(":").strip()
    if cleaned in SECTION_NAMES:
        return cleaned
    return None


def parse_sections(response: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {name: [] for name in SECTION_NAMES}
    current: str | None = None

    for line in response.splitlines():
        header = _normalize_header(line)
        if header:
            current = header
            continue
        if current:
            sections[current].append(line)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def run_pipeline(
    company: str,
    role: str,
    on_stage: Callable[[str], None] | None = None,
) -> PipelineResult:
    pipeline_start = time.perf_counter()

    def notify(stage: str) -> None:
        if on_stage:
            on_stage(stage)

    notify("Generating with DeepSeek V4 Flash (single request)...")
    print(f"[Pipeline] Calling NVIDIA NIM for {company} / {role}", flush=True)

    try:
        response = chat_completion(
            [{"role": "user", "content": _build_prompt(company, role)}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except (NvidiaAPIError, TimeoutError, ValueError, httpx.HTTPError) as exc:
        raise ValueError(
            f"Could not complete NVIDIA API request: {exc}\n\n"
            "Wait 1–2 minutes and try again. Confirm NVIDIA_API_KEY in .env "
            "from https://build.nvidia.com/ (keys start with nvapi-)."
        ) from exc

    parsed = parse_sections(response)
    company_research = parsed["COMPANY RESEARCH"] or response
    role_requirements = parsed["ROLE REQUIREMENTS"] or ""
    job_posting = parsed["JOB POSTING"] or response

    elapsed = time.perf_counter() - pipeline_start
    print(f"[Pipeline] Finished in {elapsed:.1f}s", flush=True)

    notify("Complete")
    return PipelineResult(
        company_research=company_research,
        role_requirements=role_requirements,
        job_posting=job_posting,
    )
