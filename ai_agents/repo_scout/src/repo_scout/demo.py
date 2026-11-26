"""Explicit fixtures, served through the real Apify parser using HTTPX MockTransport."""

import asyncio
import json
from importlib.resources import files

import httpx
from pydantic import SecretStr

from .apify_client import ApifyClient
from .config import Settings
from .research_models import Limits

DATA = json.loads(files("repo_scout").joinpath("fixtures/demo.json").read_text(encoding="utf-8"))
SAMPLE_REQUIREMENT = DATA["requirement"]
PREFIX = "https://github.com/reposcout-samples/"


def pages() -> dict:
    output = {}
    for item in DATA["projects"]:
        repo = PREFIX + item["slug"]
        output[repo] = {
            "role": "repository",
            "markdown": "\n".join(
                [
                    f"# {item['name']}",
                    "SYNTHETIC TEST DATA — not an actual project.",
                    item["hosting"],
                    item["tables"],
                    item["terms"] or "",
                    f"[README]({repo}/blob/main/README.md)",
                    f"[Licence]({repo}/blob/main/LICENSE)",
                    f"[Documentation](https://docs.example.com/{item['slug']})",
                    f"[Releases]({repo}/releases)",
                    f"[Repository activity]({repo}/commits/main)",
                ]
            ),
        }
        for path, role, content in [
            ("/blob/main/README.md", "readme", item["hosting"] + "\n" + item["tables"]),
            ("/blob/main/LICENSE", "licence", item["licence"]),
            ("/releases", "release", item["release"]),
            ("/commits/main", "activity", item["activity"]),
        ]:
            output[repo + path] = {
                "role": role,
                "markdown": content,
                "failed": role == "licence" and item["licence_failure"],
            }
        output[f"https://docs.example.com/{item['slug']}"] = {
            "role": "documentation",
            "markdown": item["docs"],
        }
    return output


class DemoWeb:
    def __init__(self):
        self.pages = pages()
        self.active = self.peak_concurrency = 0
        self.requests: list[dict] = []

    async def handle(self, request: httpx.Request):
        self.active += 1
        self.peak_concurrency = max(self.peak_concurrency, self.active)
        try:
            await asyncio.sleep(0.005)  # Exercises overlap, never treated as a benchmark.
            payload = json.loads(request.content)
            self.requests.append(payload)
            if request.url.host == "web-search.apify.actor":
                results = [
                    {
                        "url": PREFIX + item["slug"],
                        "title": item["name"],
                        "snippet": "Synthetic self-hosted PDF parsing project; inspect table and licensing evidence.",
                        "position": index + 1,
                    }
                    for index, item in enumerate(DATA["projects"])
                ]
                # A deep-link duplicate and non-repository result exercise normalization.
                results.extend(
                    [
                        {**results[0], "url": results[0]["url"] + "/blob/main/README.md"},
                        {
                            "url": "https://github.com/topics/pdf",
                            "title": "Topic",
                            "snippet": "Not a repository",
                            "position": 7,
                        },
                    ]
                )
                return httpx.Response(
                    200,
                    json={
                        "query": payload["query"],
                        "search": {"provider": "synthetic-fixture"},
                        "results": results,
                    },
                )
            page = self.pages.get(payload["url"])
            if not page or page.get("failed"):
                return httpx.Response(
                    200,
                    json={
                        "url": payload["url"],
                        "metadata": {},
                        "fetch": {"httpStatusCode": 404, "loadedUrl": payload["url"]},
                        "markdown": "Synthetic missing page — no usable licence evidence.",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "url": payload["url"],
                    "metadata": {},
                    "fetch": {"httpStatusCode": 200, "loadedUrl": payload["url"]},
                    "markdown": page["markdown"],
                },
            )
        finally:
            self.active -= 1

    def client(self) -> ApifyClient:
        return ApifyClient(
            Settings(token=SecretStr("synthetic-demo-token"), account_plan="offline sample"),
            transport=httpx.MockTransport(self.handle),
        )


class DemoModel:
    """Scripted structured decisions for the sample scenario, not a language model."""

    def __init__(self):
        self.pages = pages()

    async def generate(self, stage: str, context: dict, schema: dict, limits: Limits) -> str:
        if stage == "interpret":
            return json.dumps(DATA["interpretation"])
        if stage == "select":
            candidates = [
                {
                    "repo_url": item["repo_url"],
                    "reason": "Discovery describes PDF parsing; inspect capabilities and terms.",
                }
                for item in context["discovered"]
                if "PDF parsing" in (item["snippet"] or "")
            ][: context["candidate_limit"]]
            selected = {c["repo_url"] for c in candidates}
            return json.dumps(
                {
                    "candidates": candidates,
                    "decisions": [
                        {
                            "repo_url": item["repo_url"],
                            "action": "inspect" if item["repo_url"] in selected else "defer",
                            "reason": "Plausible PDF lead for inspection"
                            if item["repo_url"] in selected
                            else "Deferred under inspection capacity",
                        }
                        for item in context["discovered"]
                    ],
                }
            )
        if stage == "plan":
            requests = []
            fetched = {(source["repo_url"], source["url"]) for source in context["sources"]}
            for source in context["sources"]:
                for link in source["links"]:
                    page = self.pages.get(link)
                    if not page or (source["repo_url"], link) in fetched:
                        continue
                    if (page["role"] == "documentation") != (context["iteration"] > 0):
                        continue
                    requests.append(
                        {
                            "repo_url": source["repo_url"],
                            "url": link,
                            "role": page["role"],
                            "parent_id": source["id"],
                            "reason": "Inspect linked evidence for a documented criterion or separate release/activity signal.",
                        }
                    )
            return json.dumps({"requests": requests})
        if stage == "extract":
            if "projects" in context:
                return json.dumps(
                    {
                        "items": [
                            {"repo_url": p["repo_url"], "extraction": self.extract(p)}
                            for p in context["projects"]
                        ]
                    }
                )
            return json.dumps(self.extract(context))
        if stage == "shortlist":
            ranked = sorted(
                context["eligible_projects"],
                key=lambda p: (
                    -sum(a["verdict"] == "Matches" for a in p["assessments"]),
                    p["name"],
                ),
            )
            output = []
            for project in ranked:
                citations = [cite for a in project["assessments"] for cite in a["citations"]][:8]
                unresolved = [
                    a["criterion_id"]
                    for a in project["assessments"]
                    if a["verdict"] == "Needs review"
                ]
                output.append(
                    {
                        "repo_url": project["candidate"]["repo_url"],
                        "explanation": "The cited sample documentation supports the matched criteria shown in the comparison.",
                        "tradeoffs": "Manual review: " + ", ".join(unresolved)
                        if unresolved
                        else "Review documented scope and source dates; no capabilities were execution-tested and recency does not prove quality.",
                        "citations": citations,
                    }
                )
            return json.dumps({"shortlist": output})
        raise ValueError("Unknown fixture stage")

    @staticmethod
    def extract(context):
        item = next(
            project
            for project in DATA["projects"]
            if context["repo_url"] == PREFIX + project["slug"]
        )
        sources = [source for source in context["sources"] if source["outcome"] == "success"]

        def cites(text):
            return [
                {"source_id": source["id"], "excerpt": text}
                for source in sources
                if text
                and text
                in source.get("markdown", "\n".join(p["text"] for p in source.get("passages", [])))
            ][:1]

        facts = []
        for kind, text in [
            ("self_hosting", item["hosting"]),
            ("capability", item["tables"]),
            ("licence", item["licence"]),
            ("latest_release", item["release"]),
            ("repository_activity", item["activity"]),
            ("separate_terms", item["docs"] if item["terms"] else None),
        ]:
            citations = cites(text)
            if citations:
                facts.append({"kind": kind, "value": text, "citations": citations})
        conflict = cites(item["terms"]) + cites(item["licence"]) if item["terms"] else []
        if len(conflict) == 2:
            facts.append(
                {
                    "kind": "conflict",
                    "value": "README and code licence conflict; model-weight terms need separate review.",
                    "citations": conflict,
                }
            )
        assessments = []
        for criterion, text, matches in [
            ("hosting", item["hosting"], True),
            ("tables", item["tables"], item["slug"] != "slate-pdf"),
            ("licence", item["licence"], item["licence_match"]),
            ("activity", item["activity"], item["recent"]),
        ]:
            citations = cites(text)
            verdict = (
                "Needs review" if not citations else "Matches" if matches else "Does not match"
            )
            if criterion == "licence" and conflict:
                verdict, citations = "Needs review", conflict
            assessments.append(
                {
                    "criterion_id": criterion,
                    "verdict": verdict,
                    "explanation": text if citations else "No licence evidence was retrieved.",
                    "citations": citations,
                    "basis": "documented" if citations else "unknown",
                }
            )
        return {"project_name": item["name"], "facts": facts, "assessments": assessments}
