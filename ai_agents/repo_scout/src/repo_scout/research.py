"""Bounded evidence workflow shared by demo and live modes; never uses classic runs."""

import asyncio
import html
import re
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from pydantic import ValidationError

from .apify_client import ApifyClient
from .evidence import passages, select_passages
from .llm import ModelAuthenticationError, ModelError, ModelQuotaError, StructuredModel
from .models import FetchInput, SearchInput, SearchResponse
from .research_models import (
    Assessment,
    BatchExtraction,
    Candidate,
    Citation,
    Discovered,
    Extraction,
    Interpretation,
    Limits,
    Project,
    ResearchPlan,
    ResearchResult,
    Selection,
    SelectionDecision,
    Source,
    SourceRequest,
    Synthesis,
)


def normalize_repo(url: str) -> str | None:
    """Normalize GitHub repository/deep links, excluding site navigation and credentials."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"https", "http"}
        or parsed.hostname not in {"github.com", "www.github.com"}
        or parsed.username
        or parsed.password
        or port not in {None, 80, 443}
    ):
        return None
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 2 or parts[0].lower() in {
        "topics",
        "search",
        "collections",
        "marketplace",
        "orgs",
        "users",
        "settings",
        "features",
        "sponsors",
        "login",
        "signup",
        "about",
        "security",
    }:
        return None
    owner, repo = parts[:2]
    repo = re.sub(r"\.git$", "", repo, flags=re.I)
    if not re.fullmatch(r"[A-Za-z0-9-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        return None
    if repo in {".", ".."}:
        return None
    return f"https://github.com/{owner.lower()}/{repo.lower()}"


def discovered_links(markdown: str, base: str) -> list[str]:
    # Fetch only links actually present in the evidence; strip anchors, never guess paths.
    targets = re.findall(r"(?<!!)\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", markdown)
    targets += re.findall(r"<((?:https?://)[^>\s]+)>", markdown)
    links = []
    for target in targets:
        try:
            parsed = urlsplit(urljoin(base, target.strip("<>")))
            url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
            FetchInput(url=url)
        except (ValidationError, ValueError):
            continue
        if url not in links:
            links.append(url)
    # GitHub navigation can otherwise crowd actual project docs out of the cap.
    if len(links) <= 100:
        return links
    return sorted(
        links,
        key=lambda link: (
            not bool(
                re.search(
                    r"readme|licen[cs]e|/docs/|quickstart|install|usage|/releases|/commits/",
                    link,
                    re.I,
                )
            )
        ),
    )[:100]


def merge_links(content: str, base: str, returned: list[str] | None) -> list[str]:
    links = discovered_links(content, base)
    for target in returned or []:
        try:
            parsed = urlsplit(urljoin(base, target))
            url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
            FetchInput(url=url)
        except (ValidationError, ValueError):
            continue
        if url not in links:
            links.append(url)
    return links[:200]


def supported(citations: list[Citation], sources: list[Source], repo_url: str) -> list[Citation]:
    pages = {
        source.id: source
        for source in sources
        if source.repo_url == repo_url and source.outcome == "success"
    }
    valid = []
    for cite in citations:
        source = pages.get(cite.source_id)
        if source is None:
            continue
        if cite.passage_id:
            passage = next(
                (p for p in passages(source.id, source.markdown) if p["id"] == cite.passage_id),
                None,
            )
            if passage:
                valid.append(cite.model_copy(update={"excerpt": passage["text"]}))
            continue
        excerpt = canonical_excerpt(cite.excerpt)
        urls = re.findall(r"https?://[^\s)\]>]+", cite.excerpt)
        if any(url not in source.markdown for url in urls):
            continue
        if excerpt and (
            " ".join(cite.excerpt.split()) in " ".join(source.markdown.split())
            or excerpt in canonical_excerpt(source.markdown)
        ):
            valid.append(cite)
    return valid


def canonical_excerpt(text: str) -> str:
    """Ignore punctuation escapes and bold delimiters, not words or link targets."""
    text = re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~])", r"\1", text)
    text = html.unescape(text).replace("’", "'").replace("‘", "'").replace("\u00ad", "")
    text = re.sub(r"(?<!!)\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"`+([^`]+)`+", r"\1", text)
    return " ".join(text.split())


def evidence_window(markdown: str, limit: int) -> str:
    """Bounded original passages, retaining relevant sections beyond a page prefix.

    Never paraphrase evidence. Full Actor bodies remain in the response journal.
    """
    if len(markdown) <= limit:
        return markdown
    spans = [(0, min(500, limit // 4))]
    pattern = r"licen[cs]|permission is hereby|associated documentation|table extraction|read_pdf|find_tables|pip install|local mode|deterministic local|commits on|hybrid|model.weight|command.line|download[^\n]{0,100}(?:jar|binary)|java -jar"
    matches = sorted(
        re.finditer(pattern, markdown, re.I),
        key=lambda match: (
            not bool(
                re.search(
                    r"table extraction|read_pdf|find_tables|pip install|commits on|command.line|download|java -jar",
                    match.group(),
                    re.I,
                )
            ),
            match.start(),
        ),
    )
    for match in matches:
        start = markdown.rfind("\n\n", 0, match.start()) + 2
        end = markdown.find("\n\n", match.end())
        end = len(markdown) if end < 0 else end
        if end - start > 2000:
            start = markdown.rfind("\n", 0, match.start()) + 1
            line_end = markdown.find("\n", match.end() + 600)
            end = min(end, line_end if line_end >= 0 else len(markdown))
        spans.append((max(0, start), end))
    chosen, used = [], 0
    for start, end in spans:
        if any(a <= start and end <= b for a, b in chosen):
            continue
        if end - start + used + 9 <= limit:
            chosen.append((start, end))
            used += end - start + 9
    return "\n\n[...]\n\n".join(markdown[a:b] for a, b in sorted(chosen))[:limit]


def unknown(
    criterion_id: str, explanation: str = "No validated evidence; manual review required"
) -> Assessment:
    return Assessment(
        criterion_id=criterion_id,
        verdict="Needs review",
        explanation=explanation,
        citations=[],
        basis="unknown",
    )


def enforce_required_conditions(requirement: str, interpretation: Interpretation) -> Interpretation:
    """A preference needs explicit, verbatim user authorization; default is required.

    Generic requirement policy, never a candidate verdict or a topic-specific rule.
    """
    interpretation = interpretation.model_copy(deep=True)
    for criterion in interpretation.criteria:
        quote = criterion.preference_quote.strip()
        explicit = (
            quote
            and quote in requirement
            and re.search(
                r"\b(prefer(?:red|ably|ence)?|optional(?:ly)?|nice.to.have|ideally|if possible)\b",
                quote,
                re.I,
            )
        )
        if not criterion.required and not explicit:
            criterion.required = True
            criterion.description = re.sub(
                r"(?i)^preference\s*:", "Required:", criterion.description
            )
    return interpretation


class Researcher:
    def __init__(
        self,
        web: ApifyClient,
        model: StructuredModel,
        limits: Limits,
        *,
        offline: bool = False,
        progress: Callable[[str, float], None] | None = None,
    ):
        self.web, self.model, self.limits = web, model, limits
        self.offline, self.progress = offline, progress or (lambda label, fraction: None)
        self.web_calls = 0
        self.web_seconds = 0.0
        self.raw: list[dict] = []
        self.context_coverage: dict = {}
        self.citation_rejections: list[dict] = []

    async def reassess_saved(self, previous: ResearchResult) -> ResearchResult:
        """Replay assessment with real production decisions, never reviewed verdicts.

        Discovery identity and already retrieved sources are inputs; interpretation,
        facts, assessments and shortlist are newly generated by the selected backend.
        No discovery, planner, or web request is made by this mode.
        """
        if previous.offline:
            raise ValueError("Production replay requires archived live evidence")
        result = ResearchResult(
            requirement=previous.requirement,
            offline=False,
            created_at=previous.created_at,
            discovered=previous.discovered,
            sources=previous.sources,
            projects=[Project(candidate=p.candidate, name=p.name) for p in previous.projects],
        )
        try:
            result.interpretation = await self.model.ask(
                "interpret",
                {
                    "requirement": result.requirement,
                    "as_of": result.created_at.isoformat(),
                    "instruction": "Keep all user conditions required unless explicitly optional. Do not add obligations for unrelated optional assets. If recency is requested without a cutoff, disclose a 12-month screening assumption in the summary and criterion description. Emit at most 3 focused repository queries.",
                },
                Interpretation,
            )
            result.interpretation = enforce_required_conditions(
                result.requirement, result.interpretation
            )
            for project in result.projects:
                project.assessments = [unknown(c.id) for c in result.interpretation.criteria]
            await self._extract_projects(result, result.projects)
            await self._synthesize(result)
        except ModelError as exc:
            result.warnings.append(str(exc))
        result = self._finish(result)
        result.diagnostics["execution_mode"] = "automated_saved_evidence_replay"
        result.diagnostics["replay_source"] = previous.diagnostics.get("saved_result_dir")
        result.diagnostics["assessment_origin"] = (
            "Production model-generated interpretation/extraction/shortlist; no manual overrides"
        )
        result.diagnostics["historical_usage_not_recharged"] = {
            "web_requests": previous.diagnostics.get("web_requests"),
            "model_calls": previous.diagnostics.get("model_calls"),
        }
        return result

    async def recover_assessments(self, previous: ResearchResult) -> ResearchResult:
        """Reassess archived sources after a shared assessment defect, without web calls.

        Keep interpretation, discovery, retrieval history and cumulative run limits.
        Prior model charges remain in the backend's persistent ledger.
        """
        if previous.offline or previous.interpretation is None:
            raise ValueError("Recovery requires an interpreted live result")
        result = previous.model_copy(deep=True)
        self.web_calls = previous.diagnostics["web_requests"]
        self.web_seconds = previous.diagnostics.get("retrieval_elapsed_seconds") or 0.0
        self.raw = list(previous.diagnostics.get("retrievals", []))
        self.model.calls = previous.diagnostics["model_calls"]
        self.model.traces = list(previous.diagnostics.get("model_traces", []))
        for attr, key in [
            ("usage_records", "model_usage"),
            ("count_records", "model_input_counts"),
        ]:
            if hasattr(self.model.backend, attr):
                getattr(self.model.backend, attr).extend(previous.diagnostics.get(key, []))
        result.shortlist = []
        try:
            await self._extract_projects(result, result.projects)
            await self._synthesize(result)
        except ModelError as exc:
            result.warnings.append(str(exc))
        result = self._finish(result)
        result.diagnostics["execution_mode"] = "live_assessment_recovery"
        result.diagnostics["recovery_source"] = previous.diagnostics.get("saved_result_dir")
        result.diagnostics["new_web_requests"] = 0
        result.diagnostics["observed_peak_retrieval_concurrency"] = previous.diagnostics.get(
            "observed_peak_retrieval_concurrency", 0
        )
        return result

    async def run(self, requirement: str) -> ResearchResult:
        if not requirement.strip() or len(requirement) > 4000:
            raise ValueError("Requirement must contain 1..4000 characters")
        result = ResearchResult(
            requirement=requirement, offline=self.offline, created_at=datetime.now(timezone.utc)
        )
        if self.offline:
            result.warnings.append(
                "Offline demo — sample data. This scripted PDF scenario is not live research; "
                "editing the requirement does not change the fixture's sample criteria."
            )
        self.progress("Interpreting requirements", 0.05)
        try:
            result.interpretation = await self.model.ask(
                "interpret",
                {
                    "requirement": requirement,
                    "as_of": result.created_at.isoformat(),
                    "instruction": "Keep all user conditions required unless explicitly optional. Do not add obligations for unrelated optional assets. If recency is requested without a cutoff, disclose a 12-month screening assumption in the summary and criterion description. Emit at most 3 focused repository queries.",
                },
                Interpretation,
            )
            if not self.offline:
                result.interpretation = enforce_required_conditions(
                    requirement, result.interpretation
                )
            ids = [item.id for item in result.interpretation.criteria]
            if len(ids) != len(set(ids)):
                raise ModelError("Interpretation contains duplicate criterion IDs")
            await self._discover(result)
            if not result.discovered:
                result.warnings.append("No relevant public repository results were available.")
                return self._finish(result)
            await self._select_candidates(result)
            if result.projects:
                await self._inspect_selected(result)
        except ModelError as exc:
            result.warnings.append(str(exc))
            # Existing evidence and validated assessments survive an exhausted budget or model failure.
        return self._finish(result)

    async def resume_discovery(
        self, previous: ResearchResult, *, selection_cache: Selection | None = None
    ) -> ResearchResult:
        if (
            previous.offline
            or previous.projects
            or not previous.discovered
            or previous.interpretation is None
        ):
            raise ValueError(
                "Discovery resume requires a live discovery checkpoint with no selected projects"
            )
        result = previous.model_copy(deep=True)
        self.raw = list(previous.diagnostics.get("retrievals", []))
        self.web_calls = previous.diagnostics["web_requests"]
        self.web_seconds = previous.diagnostics.get("retrieval_elapsed_seconds") or 0.0
        self.model.calls = previous.diagnostics["model_calls"]
        self.model.traces = list(previous.diagnostics.get("model_traces", []))
        for attr, key in [
            ("usage_records", "model_usage"),
            ("count_records", "model_input_counts"),
        ]:
            if hasattr(self.model.backend, attr):
                getattr(self.model.backend, attr).extend(previous.diagnostics.get(key, []))
        result.warnings = [
            w for w in result.warnings if not w.startswith("Candidate selection returned")
        ]
        try:
            await self._select_candidates(result, selection_cache=selection_cache)
            if selection_cache is not None:
                result.warnings = [
                    w
                    for w in result.warnings
                    if not w.startswith("Selection must explain exactly one decision")
                ]
            if result.projects:
                await self._inspect_selected(result)
        except ModelError as exc:
            result.warnings.append(str(exc))
        result = self._finish(result)
        result.diagnostics["resumed_from_discovery"] = previous.diagnostics.get("saved_result_dir")
        result.diagnostics["saved_selection_response_reused"] = selection_cache is not None
        if selection_cache is not None:
            result.diagnostics["historical_selection_warnings"] = previous.warnings
        result.diagnostics["previous_discovery_requests_reused"] = previous.diagnostics[
            "web_requests"
        ]
        return result

    async def _select_candidates(self, result, *, selection_cache: Selection | None = None):
        # Inspect broad plausible leads using root evidence first, reserving five or
        # more requests for agent-chosen gaps. The ceiling is not a quota.
        inspection_capacity = min(
            self.limits.candidate_limit,
            max(1, self.limits.max_web_requests - self.web_calls - 5),
        )
        selection = selection_cache or await self.model.ask(
            "select",
            {
                "interpretation": result.interpretation.model_dump(),
                "discovered": [item.model_dump() for item in result.discovered],
                "candidate_limit": inspection_capacity,
                "instruction": "Select plausible relevant discovered repositories for inspection, "
                "not only already-proven matches. Search snippets are leads: missing licence or "
                "maintenance evidence must be checked after retrieval, not treated as failure "
                "at selection. Never invent repositories or claim eligibility from snippets. "
                "Candidate limit is a ceiling, not quota. Preserve a selection reason; return "
                "empty only if no supplied repository is plausibly relevant. "
                "For EVERY discovered repository emit one decision: inspect, exclude (clearly irrelevant), "
                "or defer (relevant but capacity-limited). Use available capacity for plausible leads; "
                "defer only when capacity is reached. All reasons, including candidate reasons, "
                "must be under 120 characters. "
                "The candidates array must equal the inspect decisions, within candidate_limit. "
                "Unknown licence, maintenance, deployment or table details are questions for retrieval, "
                "not grounds for exclusion. Search publication dates are not repository activity.",
            },
            Selection,
        )
        allowed = {item.repo_url for item in result.discovered}
        decisions = {normalize_repo(d.repo_url): d for d in selection.decisions}
        if len(decisions) != len(selection.decisions):
            raise ModelError("Duplicate selection decisions; no retry")
        # candidates already declares inspect + reason. Reuse that explicit model
        # decision if omitted from the redundant list; never invent a decision for
        # an unselected repository or overwrite a conflicting exclusion.
        for candidate in selection.candidates:
            canonical = normalize_repo(candidate.repo_url)
            if canonical in allowed and canonical not in decisions:
                decisions[canonical] = SelectionDecision(
                    repo_url=canonical, action="inspect", reason=candidate.reason
                )
        if set(decisions) != allowed:
            raise ModelError(
                "Selection must explain exactly one decision for every discovered repository; no retry"
            )
        proposed = {normalize_repo(c.repo_url) for c in selection.candidates} & allowed
        inspected = {url for url, d in decisions.items() if d.action == "inspect"}
        if inspected != proposed or len(inspected) > inspection_capacity:
            raise ModelError("Selection decisions disagree with candidates or capacity; no retry")
        result.selection_decisions = list(decisions.values())
        self.selection_decisions = [d.model_dump() for d in result.selection_decisions]

        seen = set()
        for candidate in selection.candidates:
            canonical = normalize_repo(candidate.repo_url)
            if canonical not in allowed or canonical in seen:
                result.warnings.append(
                    "Ignored a duplicate or undiscovered candidate from model selection."
                )
                continue
            candidate = Candidate(repo_url=canonical, reason=candidate.reason)
            result.projects.append(
                Project(
                    candidate=candidate,
                    name=canonical.rsplit("/", 1)[1],
                    assessments=[
                        unknown(criterion.id) for criterion in result.interpretation.criteria
                    ],
                )
            )
            seen.add(canonical)
            if len(result.projects) >= inspection_capacity:
                break
        if not result.projects:
            result.warnings.append(
                "Candidate selection returned an empty list. Discovered repositories were not "
                "inspected; this does not establish that none meet the requirement."
            )
            return

    async def _inspect_selected(self, result):
        self.progress("Inspecting repository evidence", 0.3)
        await self._retrieve(
            result,
            [
                SourceRequest(
                    repo_url=p.candidate.repo_url,
                    url=p.candidate.repo_url,
                    role="repository",
                    parent_id="discovery",
                    reason="Selected discovery result; inspect README and outgoing evidence links",
                )
                for p in result.projects
            ],
        )
        await self._extract_projects(result, result.projects)
        for iteration in range(self.limits.max_iterations):
            self.progress(
                f"Inspecting sources · round {iteration + 1} of {self.limits.max_iterations}",
                0.35 + 0.4 * iteration / self.limits.max_iterations,
            )
            before = len(result.sources)
            plan = await self._plan(result, iteration)
            await self._retrieve(result, plan)
            changed_repos = {s.repo_url for s in result.sources[before:]}
            await self._extract_projects(
                result, [p for p in result.projects if p.candidate.repo_url in changed_repos]
            )
            if (
                not plan
                or len(result.sources) == before
                or self.web_calls >= self.limits.max_web_requests
            ):
                break
        self.progress("Comparing criteria and preparing shortlist", 0.9)
        await self._synthesize(result)

    async def _discover(self, result):
        self.progress("Discovering public repositories", 0.15)
        queries = list(dict.fromkeys(result.interpretation.queries))[
            : min(3, self.limits.max_web_requests)
        ]
        requests = [SearchInput(query=query, maxResults=20) for query in queries]
        self.web_calls += len(requests)
        gate = asyncio.Semaphore(self.limits.concurrency)

        async def one(request):
            async with gate:
                return await self.web.search(request)

        start = perf_counter()
        outcomes = await asyncio.gather(*(one(request) for request in requests))
        self.web_seconds += perf_counter() - start
        found = {}
        for outcome in outcomes:
            self.raw.append(outcome.model_dump(mode="json"))
        self._check_terminal(outcomes)
        for outcome in outcomes:
            if outcome.measurement.outcome not in {"success", "empty"}:
                result.warnings.append(
                    "A discovery search failed or timed out; results may be incomplete."
                )
                continue
            try:
                search = SearchResponse.model_validate(outcome.raw[-1]["body"])
            except (ValidationError, KeyError, IndexError):
                result.warnings.append("A discovery response could not be parsed.")
                continue
            for item in search.results:
                try:
                    canonical = normalize_repo(str(item.url))
                except ValueError:
                    canonical = None
                if canonical:
                    if canonical not in found:
                        found[canonical] = Discovered(
                            repo_url=canonical,
                            title=item.title[:200],
                            snippet=item.snippet[:1000] if item.snippet is not None else None,
                            search_urls=[str(item.url)],
                        )
                    else:
                        if str(item.url) not in found[canonical].search_urls:
                            found[canonical].search_urls.append(str(item.url))
                        if found[canonical].snippet is None and item.snippet is not None:
                            found[canonical].snippet = item.snippet[:1000]
        result.discovered = list(found.values())[:60]

    async def _retrieve(self, result, plan):
        seen = {(s.repo_url, s.url) for s in result.sources}
        remaining = self.limits.max_web_requests - self.web_calls
        queued = []
        # Round robin prevents the first candidate from consuming the whole source budget.
        projects = [project.candidate.repo_url for project in result.projects]
        groups = {
            repo: [request for request in plan if request.repo_url == repo] for repo in projects
        }
        while any(groups.values()) and len(queued) < remaining:
            for repo in projects:
                if groups[repo] and len(queued) < remaining:
                    request = groups[repo].pop(0)
                    if (repo, request.url) not in seen:
                        queued.append(request)
                        seen.add((repo, request.url))
        if not queued:
            return
        self.web_calls += len(queued)
        start = perf_counter()
        # Existing helper caps each invocation at 20; maintain that bound for larger plans.
        outcomes = []
        for offset in range(0, len(queued), 20):
            outcomes.extend(
                await self.web.fetch_many(
                    [
                        FetchInput(url=request.url, formats=["markdown", "text", "links"])
                        for request in queued[offset : offset + 20]
                    ],
                    concurrency=self.limits.concurrency,
                    stop_on_auth_quota=True,
                )
            )
            if any(out.measurement.actor_http_status in {401, 403, 429} for out in outcomes):
                break
        self.web_seconds += perf_counter() - start
        self.web_calls -= len(queued) - len(outcomes)
        self.web_calls -= sum(
            bool(out.measurement.error and out.measurement.error.startswith("Not requested:"))
            for out in outcomes
        )
        for request, outcome in zip(queued[: len(outcomes)], outcomes, strict=True):
            self.raw.append(outcome.model_dump(mode="json"))
            full_markdown = outcome.content or ""
            markdown = full_markdown  # Archive original text; only model context is bounded.
            body = outcome.raw[-1].get("body", {}) if outcome.raw else {}
            fetch_details = body.get("fetch") if isinstance(body, dict) else None
            loaded = fetch_details.get("loadedUrl") if isinstance(fetch_details, dict) else None
            # Validate redirect URL before using it as a base; unsafe redirects cannot seed new requests.
            try:
                FetchInput(url=loaded or request.url)
            except (ValidationError, ValueError):
                loaded = request.url
            result.sources.append(
                Source(
                    id=f"s{len(result.sources) + 1}",
                    repo_url=request.repo_url,
                    url=request.url,
                    loaded_url=loaded,
                    role=request.role,
                    parent_id=None if request.parent_id == "discovery" else request.parent_id,
                    retrieval_timestamp=outcome.measurement.timestamp,
                    outcome=outcome.measurement.outcome,
                    error=outcome.measurement.error,
                    markdown=markdown,
                    links=merge_links(
                        full_markdown,
                        loaded or request.url,
                        body.get("links") if isinstance(body, dict) else None,
                    ),
                    truncated=False,
                )
            )
        self._check_terminal(outcomes)

    @staticmethod
    def _check_terminal(outcomes):
        if any(out.measurement.actor_http_status in {401, 403} for out in outcomes):
            raise ModelAuthenticationError(
                "Apify authentication/permission failed. Research stopped; verify APIFY_TOKEN and account access."
            )
        if any(out.measurement.actor_http_status == 429 for out in outcomes):
            raise ModelQuotaError(
                "Apify quota or rate limit reached (429). Research stopped without retry; completed evidence is preserved."
            )

    def _source_context(self, result, repo=None):
        sources = [source for source in result.sources if repo is None or source.repo_url == repo]
        context, coverage = select_passages(
            sources,
            min(self.limits.max_source_chars, self.limits.max_input_chars // 4),
            " ".join(c.label for c in result.interpretation.criteria)
            if result.interpretation
            else "",
        )
        self.context_coverage[repo or "all"] = coverage
        return context

    def _planning_sources(self, result):
        """Planner needs available links and unresolved criteria, not duplicate quoted facts."""
        pages = []
        seen_urls = {source.url for source in result.sources}
        for source in result.sources:
            ranked = sorted(
                [link for link in source.links if link not in seen_urls],
                key=lambda link: (
                    not bool(
                        re.search(
                            r"readme|licen[cs]e|release|commit|doc|install|usage|guide", link, re.I
                        )
                    )
                ),
            )
            pages.append(
                {
                    "id": source.id,
                    "repo_url": source.repo_url,
                    "url": source.url,
                    "role": source.role,
                    "outcome": source.outcome,
                    "markdown": source.markdown[:300],
                    "links": ranked[:8],
                }
            )
        return pages

    async def _plan(self, result, iteration):
        try:
            plan = await self.model.ask(
                "plan",
                {
                    "interpretation": result.interpretation.model_dump(),
                    "iteration": iteration,
                    "projects": [
                        {
                            "candidate": project.candidate.model_dump(),
                            "name": project.name,
                            "assessments": [
                                {
                                    "criterion_id": a.criterion_id,
                                    "verdict": a.verdict,
                                    "explanation": a.explanation[:350],
                                }
                                for a in project.assessments
                            ],
                        }
                        for project in result.projects
                    ],
                    "sources": self._planning_sources(result),
                    "remaining_web_requests": self.limits.max_web_requests - self.web_calls,
                    "instruction": "Request README, licence, documentation, releases, and activity evidence "
                    "from supplied links only. Seek additional evidence for unresolved criteria. "
                    "Use parent_id of the source containing the exact URL. Never request an already "
                    "retrieved URL again; read its evidence or use an unseen relevant linked page. "
                    "Return no requests if done.",
                },
                ResearchPlan,
            )
        except (ModelQuotaError, ModelAuthenticationError):
            raise
        except ModelError as exc:
            result.warnings.append(f"Research planning stopped: {exc}")
            return []
        parents = {source.id: source for source in result.sources if source.outcome == "success"}
        valid = []
        for request in plan.requests:
            parent = parents.get(request.parent_id)
            if parent and request.repo_url == parent.repo_url and request.url in parent.links:
                valid.append(request)
            else:
                result.warnings.append(
                    "Rejected a research URL absent from its cited parent evidence."
                )
        return valid

    async def resume(self, previous: ResearchResult) -> ResearchResult:
        """Resume evidence rounds from saved successful work without rediscovery/refetching."""
        if previous.offline or not previous.interpretation or not previous.projects:
            raise ValueError("Resume requires a live result with selected candidates")
        if len(previous.projects) > self.limits.candidate_limit:
            raise ValueError("Saved candidate count exceeds the authorized limit")
        result = previous.model_copy(deep=True)
        self.raw = list(previous.diagnostics.get("retrievals", []))
        self.web_calls = previous.diagnostics["web_requests"]
        self.web_seconds = previous.diagnostics.get("retrieval_elapsed_seconds") or 0.0
        self.model.calls = previous.diagnostics["model_calls"]
        self.model.traces = list(previous.diagnostics.get("model_traces", []))
        if hasattr(self.model.backend, "usage_records"):
            self.model.backend.usage_records.extend(previous.diagnostics.get("model_usage", []))
        start_round = sum(trace["stage"] == "plan" for trace in self.model.traces)
        result.warnings = [
            warning
            for warning in result.warnings
            if "Research planning stopped: Model input exceeds" not in warning
        ]
        try:
            changed = False
            for iteration in range(start_round, self.limits.max_iterations):
                if self.web_calls >= self.limits.max_web_requests:
                    break
                self.progress(f"Resuming unresolved evidence · round {iteration + 1}", 0.4)
                before = len(result.sources)
                plan = await self._plan(result, iteration)
                await self._retrieve(result, plan)
                changed_repos = {source.repo_url for source in result.sources[before:]}
                if changed_repos:
                    changed = True
                    result.shortlist = []
                for project in result.projects:
                    if project.candidate.repo_url in changed_repos:
                        await self._extract(result, project)
                if not changed_repos:
                    break
            if changed or not result.shortlist:
                self.progress("Updating comparison and shortlist", 0.9)
                synthesis_completed = await self._synthesize(result)
                if synthesis_completed:
                    result.warnings = [
                        warning
                        for warning in result.warnings
                        if "OpenAI $0.25 cost cap cannot reserve" not in warning
                    ]
        except ModelError as exc:
            result.warnings.append(str(exc))
        return self._finish(result)

    async def _extract(self, result, project):
        try:
            extracted = await self.model.ask(
                "extract",
                {
                    "repo_url": project.candidate.repo_url,
                    "interpretation": result.interpretation.model_dump(),
                    "sources": self._source_context(result, project.candidate.repo_url),
                    "as_of": result.created_at.isoformat(),
                    "instruction": "Extract project-code licence and scoped separate terms/conflicts. "
                    "For licence_issues identify component, applicable execution path, and actual conflicting "
                    "terms; absence of conflicting evidence is not a conflict. Optional OCR/hybrid weights "
                    "must not block a documented non-model table-extraction path. Local installation plus "
                    "a local-file library example supports self-hosting, but explicit local execution is also sufficient. Extract capabilities, latest "
                    "release and repository activity separately. Return an assessment for every criterion. "
                    "Use supplied passage IDs with excerpt empty; the app resolves verbatim text. "
                    "At most 3 facts, one strong citation per assessment (two only if needed), "
                    "concise explanations under 220 characters. Missing evidence is Needs review, "
                    "not Does not match. Partial context is not evidence of absent functionality.",
                },
                Extraction,
            )
        except (ModelQuotaError, ModelAuthenticationError):
            raise
        except ModelError as exc:
            result.warnings.append(f"Evidence synthesis unavailable for {project.name}: {exc}")
            return
        self._apply_extraction(result, project, extracted)

    async def _extract_projects(self, result, projects):
        for offset in range(0, len(projects), 2):
            batch = projects[offset : offset + 2]
            self.progress(
                f"Assessing candidates {offset + 1}–{offset + len(batch)} of {len(projects)}",
                0.35 + 0.3 * (offset + len(batch)) / len(projects),
            )
            if len(batch) == 1:
                await self._extract(result, batch[0])
                continue
            contexts = [
                {
                    "repo_url": p.candidate.repo_url,
                    "sources": self._source_context(result, p.candidate.repo_url),
                }
                for p in batch
            ]
            try:
                response = await self.model.ask(
                    "extract",
                    {
                        "interpretation": result.interpretation.model_dump(),
                        "as_of": result.created_at.isoformat(),
                        "projects": contexts,
                        "instruction": "Assess both repositories independently; return exactly one item for each repo_url. "
                        "Keep every criterion required as specified. Cite supplied source_id and passage_id with excerpt empty. "
                        "At most 3 short facts and one strong passage per criterion, explanations under 220 characters. "
                        "Do not turn missing/partial context into a negative finding. Explicit local processing is sufficient "
                        "for self-hosting, no particular invocation syntax is mandatory. Licence means relevant project code; "
                        "only applicable separate terms block it. Separate release dates from activity. "
                        "Report dated activity against the interpretation's disclosed recency window.",
                    },
                    BatchExtraction,
                )
            except (ModelQuotaError, ModelAuthenticationError):
                raise
            except ModelError as exc:
                result.warnings.append(f"Assessment batch stopped: {exc}")
                continue
            found = {}
            for item in response.items:
                canonical = normalize_repo(item.repo_url)
                if canonical in found:
                    raise ModelError("Duplicate repository in assessment batch")
                if canonical in {p.candidate.repo_url for p in batch}:
                    found[canonical] = item.extraction
            for project in batch:
                if project.candidate.repo_url in found:
                    self._apply_extraction(result, project, found[project.candidate.repo_url])
                else:
                    result.warnings.append(
                        f"Model omitted assessment for {project.name}; retained as unresolved"
                    )

    @staticmethod
    def _comparison_context(project):
        data = project.model_dump()
        data["candidate"]["reason"] = data["candidate"]["reason"][:150]
        for item in data["assessments"] + data["facts"]:
            if "explanation" in item:
                item["explanation"] = item["explanation"][:250]
            for cite in item["citations"]:
                if cite.get("passage_id"):
                    cite["excerpt"] = ""
        return data

    def _apply_extraction(self, result, project, extracted):
        # An omission from an earlier batch is historical once this project is assessed.
        result.warnings = [
            w
            for w in result.warnings
            if w != f"Model omitted assessment for {project.name}; retained as unresolved"
        ]
        # Repository name is known from discovery; never trust an uncited model-supplied name.
        facts = []
        for fact in extracted.facts:
            citations = supported(fact.citations, result.sources, project.candidate.repo_url)
            if citations and len(citations) == len(fact.citations):
                facts.append(fact.model_copy(update={"citations": citations}))
            else:
                result.warnings.append("Dropped a fact with unverifiable supporting excerpts.")
        project.facts = facts
        assessments = {}
        for assessment in extracted.assessments:
            if assessment.criterion_id in assessments:
                assessments[assessment.criterion_id] = unknown(
                    assessment.criterion_id,
                    "Conflicting or duplicate assessments; manual review required",
                )
                continue
            citations = supported(assessment.citations, result.sources, project.candidate.repo_url)
            if assessment.citations and len(citations) == len(assessment.citations):
                assessments[assessment.criterion_id] = assessment.model_copy(
                    update={"citations": citations, "basis": "documented", "processing_note": ""}
                )
            else:
                rejected = {
                    "repo_url": project.candidate.repo_url,
                    "criterion_id": assessment.criterion_id,
                    "proposed_verdict": assessment.verdict,
                    "citations": [c.model_dump() for c in assessment.citations],
                }
                if assessment.citations:
                    self.citation_rejections.append(rejected)
                revised = unknown(
                    assessment.criterion_id,
                    "Citation validation rejected evidence references; this is a processing issue, not proof the requirement fails."
                    if assessment.citations
                    else assessment.explanation,
                )
                revised.citations = citations
                revised.processing_note = (
                    "citation_rejected" if assessment.citations else "unresolved"
                )
                assessments[assessment.criterion_id] = revised
        project.assessments = [
            assessments.get(criterion.id, unknown(criterion.id))
            for criterion in result.interpretation.criteria
        ]
        coverage = self.context_coverage.get(project.candidate.repo_url, {})
        for assessment in project.assessments:
            if assessment.verdict == "Needs review" and not assessment.processing_note:
                assessment.processing_note = (
                    "context_limited" if coverage.get("context_limited") else "unresolved"
                )
        # A fact label alone cannot prove a conflict or its applicability. Older outputs
        # even labelled 'No conflicting licence is documented' as a conflict fact.
        applicable = [
            issue
            for issue in extracted.licence_issues
            if issue.scope != "optional_unrelated"
            and issue.conflicting_terms
            and len(supported(issue.citations, result.sources, project.candidate.repo_url))
            == len(issue.citations)
        ]
        if applicable:
            for assessment in project.assessments:
                criterion = next(
                    c for c in result.interpretation.criteria if c.id == assessment.criterion_id
                )
                if re.search(
                    r"licen[cs]|terms|permissive", criterion.label + criterion.description, re.I
                ):
                    if assessment.verdict == "Does not match":
                        continue
                    assessment.verdict = "Needs review"
                    assessment.explanation = (
                        "Applicable conflicting licence terms require manual review: "
                        + "; ".join(issue.explanation for issue in applicable)[:1000]
                    )
                    assessment.citations = [
                        cite for issue in applicable for cite in issue.citations
                    ][:10]
                    assessment.citations = supported(
                        assessment.citations, result.sources, project.candidate.repo_url
                    )

    async def _synthesize(self, result):
        eligible = [
            project
            for project in result.projects
            if not any(
                assessment.verdict == "Does not match"
                for assessment in project.assessments
                if next(
                    c.required
                    for c in result.interpretation.criteria
                    if c.id == assessment.criterion_id
                )
            )
            and any(a.verdict == "Matches" for a in project.assessments)
        ]
        if not eligible:
            result.warnings.append(
                "No evidence-backed shortlist is available; inspect candidates needing review."
            )
            return
        try:
            synthesis = await self.model.ask(
                "shortlist",
                {
                    "interpretation": result.interpretation.model_dump(),
                    "eligible_projects": [
                        self._comparison_context(project) for project in eligible
                    ],
                    "instruction": "Rank fit to required criteria before preferences. Explain documented "
                    "tradeoffs and manual review needs with existing excerpts. Never equate activity with quality. "
                    "Return at most 3 concise recommendations. Each should use 1 or 2 short existing citations; "
                    "keep each explanation and tradeoff below 300 characters.",
                },
                Synthesis,
            )
        except (ModelQuotaError, ModelAuthenticationError):
            raise
        except ModelError as exc:
            result.warnings.append(f"Shortlist unavailable: {exc}")
            return
        allowed = {project.candidate.repo_url for project in eligible}
        used = set()
        for item in synthesis.shortlist:
            citations = supported(item.citations, result.sources, item.repo_url)
            if (
                item.repo_url in allowed
                and item.repo_url not in used
                and citations
                and len(citations) == len(item.citations)
            ):
                result.shortlist.append(item.model_copy(update={"citations": citations}))
                used.add(item.repo_url)
            else:
                result.warnings.append("Dropped an unsupported shortlist recommendation.")
        # Required fit takes priority even if the model overweights a preferred criterion.
        required = {
            criterion.id for criterion in result.interpretation.criteria if criterion.required
        }
        by_repo = {project.candidate.repo_url: project for project in eligible}
        result.shortlist.sort(
            key=lambda item: (
                -sum(
                    a.verdict == "Matches"
                    for a in by_repo[item.repo_url].assessments
                    if a.criterion_id in required
                ),
                -sum(
                    a.verdict == "Matches"
                    for a in by_repo[item.repo_url].assessments
                    if a.criterion_id not in required
                ),
            )
        )
        return True  # An empty, validated response also completed successfully.

    def _finish(self, result):
        if self.web_calls >= self.limits.max_web_requests:
            result.warnings.append(
                "Web request budget reached; unresolved criteria need manual review."
            )
        if any(source.outcome != "success" for source in result.sources):
            result.warnings.append(
                "Some sources failed or were empty; partial evidence was preserved."
            )
        if any(source.truncated for source in result.sources):
            result.warnings.append("Some source content was truncated by the input-size limit.")
        unresolved = any(
            a.verdict == "Needs review" for p in result.projects for a in p.assessments
        )
        result.status = (
            "failed"
            if result.interpretation is None
            else ("partial" if unresolved or not result.projects or result.warnings else "complete")
        )
        result.diagnostics = {
            "model_budget": self.model.backend.ledger.summary()
            if hasattr(self.model.backend, "ledger")
            else None,
            "model_usage": getattr(self.model.backend, "usage_records", []),
            "model_input_counts": getattr(self.model.backend, "count_records", []),
            "web_requests": self.web_calls,
            "observed_peak_retrieval_concurrency": getattr(self.web, "peak_concurrency", 0),
            "model_calls": self.model.calls,
            "selection_decisions": getattr(self, "selection_decisions", []),
            "context_coverage": self.context_coverage,
            "citation_rejections": self.citation_rejections,
            "retrieval_elapsed_seconds": None if self.offline else self.web_seconds,
            "model_generation_seconds": None
            if self.offline
            else sum(t["elapsed_seconds"] for t in self.model.traces),
            "timing_note": "No benchmark measurements from fixtures"
            if self.offline
            else "Separate phase totals; not a benchmark",
            "limits": self.limits.model_dump(),
            "model_traces": [
                {k: v for k, v in trace.items() if k != "elapsed_seconds"}
                for trace in self.model.traces
            ]
            if self.offline
            else self.model.traces,
            "retrievals": self.raw,
        }
        if self.offline:
            # Fixture executions are not measurements: remove synthetic timing/usage everywhere.
            for raw in result.diagnostics["retrievals"]:
                raw["measurement"].pop("elapsed_seconds", None)
                raw["measurement"].pop("request_started_at", None)
                raw["measurement"].pop("request_finished_at", None)
                raw["measurement"]["usage"] = None
                raw["measurement"]["usage_note"] = "Synthetic fixture; no live usage or benchmark"
        self.progress("Research finished · review the evidence", 1.0)
        return result
