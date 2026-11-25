"""Explicit single-attempt requests. No retries, background polling, or model calls."""

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from .config import Settings, require_live_enabled
from .models import FetchInput, FetchResponse, Measurement, Outcome, SearchInput, SearchResponse

API = "https://api.apify.com/v2"
SERVERS = {
    "web-fetch": "https://web-fetch.apify.actor/",
    "web-search": "https://web-search.apify.actor/",
}


def redact(value: Any, token: str) -> Any:
    """Sanitize even reflected secrets in response bodies and nested URL strings."""
    if isinstance(value, dict):
        return {
            redact(str(key), token): "[REDACTED]"
            if re.search(r"token|authorization|password|secret|cookie|api.?key", str(key), re.I)
            and not (
                type(item) is int
                and str(key).replace("_", "").lower()
                in {
                    "prompttokencount",
                    "candidatestokencount",
                    "thoughtstokencount",
                    "cachedcontenttokencount",
                    "totaltokencount",
                    "tooluseprompttokencount",
                    "maxoutputtokens",
                    "maxcompletiontokens",
                    "inputtokens",
                    "outputtokens",
                    "totaltokens",
                    "cachedtokens",
                    "reasoningtokens",
                    "cachewritetokens",
                    "countedinputtokens",
                }
                or isinstance(item, dict)
                and str(key).replace("_", "").lower()
                in {"inputtokensdetails", "outputtokensdetails"}
            )
            else redact(item, token)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, token) for item in value]
    if isinstance(value, str):
        if token:
            for secret in {token, quote(token, safe="")}:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)Bearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", value)
        return re.sub(
            r"(?i)([?&](?:token|api_key|key|secret|password)=)[^&#\s\"']+", r"\1[REDACTED]", value
        )
    return value


class RetrievalError(Exception):
    pass


class ApifyClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        warm_idle_status: str = "unknown",
        deadline: float = 180,
        recorder=None,
    ):
        self.token = settings.token.get_secret_value()
        self.plan = settings.account_plan
        self.warm_idle_status = warm_idle_status
        self.deadline = deadline
        self.recorder = recorder
        self.is_mock = isinstance(transport, httpx.MockTransport)
        self.active_requests = 0
        self.peak_concurrency = 0
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(75, connect=10),
            follow_redirects=False,
            transport=transport,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def _request(self, out: Outcome, method: str, url: str, **kwargs) -> Any:
        if not self.is_mock:
            require_live_enabled()
        self.active_requests += 1
        self.peak_concurrency = max(self.peak_concurrency, self.active_requests)
        out.measurement.request_started_at = datetime.now(timezone.utc)
        out.measurement.concurrent_requests_at_start = self.active_requests
        try:
            response = await self.http.request(method, url, **kwargs)
        finally:
            out.measurement.request_finished_at = datetime.now(timezone.utc)
            self.active_requests -= 1
        out.measurement.actor_http_status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = {"non_json_body": response.text}
        out.raw.append(redact({"http_status": response.status_code, "body": body}, self.token))
        if not response.is_success:
            raise RetrievalError(
                f"Apify HTTP {response.status_code}; inspect sanitized raw response"
            )
        if isinstance(body, dict) and "non_json_body" in body:
            raise RetrievalError("Apify returned non-JSON content")
        return body

    async def search(self, request: SearchInput) -> Outcome:
        return await self._execute("web-search", "server", request.model_dump())

    async def fetch(self, request: FetchInput, *, classic: bool = False) -> Outcome:
        return await self._execute(
            "web-fetch", "classic" if classic else "server", request.model_dump()
        )

    async def _classic(self, out: Outcome, payload: dict) -> dict:
        # One POST, at most two bounded long-wait GETs, then one dataset read.
        body = await self._request(
            out,
            "POST",
            f"{API}/acts/apify~web-fetch/runs",
            params={"waitForFinish": 60, "timeout": 180},
            json=payload,
        )
        for attempt in range(3):
            run = body.get("data") if isinstance(body, dict) else None
            if not isinstance(run, dict) or not isinstance(run.get("id"), str):
                raise RetrievalError("Malformed classic run response; execution status unknown")
            out.measurement.run_id = run["id"]
            usage = {
                key: run[key]
                for key in ("usage", "usageUsd", "usageTotalUsd", "chargedEventCounts")
                if key in run
            }
            if usage:
                out.measurement.usage = usage
                out.measurement.usage_note = "Apify run response; preliminary, not a final invoice"
            status = run.get("status")
            if status == "SUCCEEDED":
                dataset = run.get("defaultDatasetId")
                if not isinstance(dataset, str) or not dataset:
                    raise RetrievalError("Successful run missing dataset ID")
                items = await self._request(
                    out,
                    "GET",
                    f"{API}/datasets/{quote(dataset, safe='')}/items",
                    params={"format": "json", "limit": 2},
                )
                if not isinstance(items, list) or len(items) != 1:
                    raise RetrievalError("Expected exactly one Web Fetch dataset item")
                return items[0]
            if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
                raise RetrievalError(f"Classic run ended with {status}")
            if status not in {"READY", "RUNNING", "TIMING-OUT", "ABORTING"}:
                raise RetrievalError("Unknown classic run status; execution status unknown")
            if attempt == 2:
                raise TimeoutError
            body = await self._request(
                out,
                "GET",
                f"{API}/actor-runs/{quote(run['id'], safe='')}",
                params={"waitForFinish": 60},
            )
        raise AssertionError("unreachable")

    async def _execute(self, actor: str, interface: str, payload: dict) -> Outcome:
        out = Outcome(
            measurement=Measurement(
                timestamp=datetime.now(timezone.utc),
                actor=f"apify/{actor}",
                interface=interface,
                sanitized_input=redact(payload, self.token),
                account_plan=self.plan,
                warm_idle_status=self.warm_idle_status,
            )
        )
        start = perf_counter()
        try:
            async with asyncio.timeout(self.deadline):
                if interface == "classic":
                    body = await self._classic(out, payload)
                else:
                    body = await self._request(out, "POST", SERVERS[actor], json=payload)
                if actor == "web-search":
                    result = SearchResponse.model_validate(body)
                    out.measurement.outcome = "success" if result.results else "empty"
                else:
                    # Capture target status even when another required field is malformed.
                    if isinstance(body, dict) and isinstance(body.get("fetch"), dict):
                        status = body["fetch"].get("httpStatusCode")
                        if type(status) is int:
                            out.measurement.target_status = status
                    page = FetchResponse.model_validate(body)
                    if not 200 <= page.fetch.httpStatusCode < 300:
                        raise RetrievalError(
                            f"Target HTTP {page.fetch.httpStatusCode}; unusable evidence"
                        )
                    requested = payload.get("formats", ["markdown"])
                    content = next(
                        (
                            value
                            for name in ("markdown", "text")
                            if name in requested
                            and isinstance(value := getattr(page, name), str)
                            and value.strip()
                        ),
                        None,
                    )
                    if content is None and all(
                        getattr(page, name) is None
                        for name in ("markdown", "text")
                        if name in requested
                    ):
                        raise RetrievalError(
                            "Requested Markdown/text format missing or unsupported"
                        )
                    if content is None:
                        out.measurement.outcome = "empty"
                        out.measurement.error = "Empty Markdown; no usable evidence"
                    else:
                        out.content = redact(content, self.token)
                        out.measurement.outcome = "success"
        except (TimeoutError, httpx.TimeoutException):
            out.measurement.outcome = "uncertain"
            out.measurement.error = (
                "Client wait timed out; execution/billing may continue. Do not retry blindly."
            )
        except httpx.RequestError:
            out.measurement.outcome = "uncertain"
            out.measurement.error = "Transport failure; execution/billing status unknown"
        except ValidationError:
            out.measurement.error = "Malformed Actor response; inspect sanitized raw response"
        except RetrievalError as exc:
            out.measurement.error = str(exc)
            if "execution status unknown" in str(exc):
                out.measurement.outcome = "uncertain"
        finally:
            out.measurement.elapsed_seconds = perf_counter() - start
        out = Outcome.model_validate(redact(out.model_dump(mode="json"), self.token))
        if self.recorder:
            self.recorder("apify_response", out.model_dump(mode="json"))
        return out

    async def fetch_many(
        self, requests: list[FetchInput], *, concurrency: int = 3, stop_on_auth_quota: bool = False
    ) -> list[Outcome]:
        if not 1 <= concurrency <= 20 or len(requests) > 20:
            raise ValueError("Concurrency must be 1..20; at most 20 URLs")
        gate = asyncio.Semaphore(concurrency)
        stopped = asyncio.Event()

        async def fetch_one(request):
            async with gate:
                if stopped.is_set():
                    return Outcome(
                        measurement=Measurement(
                            timestamp=datetime.now(timezone.utc),
                            actor="apify/web-fetch",
                            interface="server",
                            sanitized_input=redact(request.model_dump(), self.token),
                            error="Not requested: batch stopped after authentication or quota failure",
                            account_plan=self.plan,
                            warm_idle_status=self.warm_idle_status,
                        )
                    )
                outcome = await self.fetch(request)
                if stop_on_auth_quota and outcome.measurement.actor_http_status in {401, 403, 429}:
                    stopped.set()
                return outcome

        return await asyncio.gather(*(fetch_one(request) for request in requests))


def compare_content(outcomes: list[Outcome]) -> dict:
    """Compare Markdown only, ignoring timestamps and fetch/metadata objects."""
    normalized = [" ".join(out.content.split()) if out.content else None for out in outcomes]
    valid = len(outcomes) == 2 and all(out.measurement.outcome == "success" for out in outcomes)
    return {
        "comparable": valid,
        "normalized_content_equal": normalized[0] == normalized[1] if valid else None,
        "content_sha256": [
            hashlib.sha256(text.encode()).hexdigest() if text else None for text in normalized
        ],
        "note": "Whitespace-normalized Markdown equality; no model generation. One pair proves no speed claim.",
    }
