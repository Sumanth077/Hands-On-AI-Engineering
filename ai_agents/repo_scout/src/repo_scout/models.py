import ipaddress
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


def public_url(value: str) -> str:
    parsed = urlsplit(value)
    # Accessing .port rejects malformed/out-of-range port values before scheduling work.
    _ = parsed.port
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise ValueError("Use a public HTTP(S) URL without credentials")
    if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local")):
        raise ValueError("Local URLs are not supported")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host:
            raise ValueError("Use a public hostname") from None
    else:
        if not address.is_global:
            raise ValueError("Private/reserved IP addresses are not supported")
    # Domain resolution and redirect destinations remain the remote Actor's responsibility.
    if parsed.query or parsed.fragment:
        raise ValueError("This public-evidence prototype accepts URLs without query or fragment")
    return value


class FetchInput(BaseModel):
    url: str
    formats: list[Literal["markdown", "text", "links"]] = Field(
        default_factory=lambda: ["markdown"], min_length=1, max_length=3
    )
    _public = field_validator("url")(public_url)


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    maxResults: int = Field(default=5, ge=1, le=20)
    allowedDomains: list[str] = Field(default_factory=lambda: ["github.com"], max_length=10)
    provider: Literal["google", "duckduckgo"] = "google"

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Search query must not be blank")
        return value


class FetchDetails(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    httpStatusCode: int = Field(ge=100, le=599)
    loadedUrl: HttpUrl


class FetchResponse(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    url: HttpUrl
    fetch: FetchDetails
    metadata: dict[str, Any]
    markdown: str | None = None
    text: str | None = None
    links: list[str] | None = None


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    url: HttpUrl
    title: str
    snippet: str | None = None
    position: int = Field(ge=1)


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    query: str
    search: dict[str, Any]
    results: list[SearchResult]


class Measurement(BaseModel):
    timestamp: datetime
    actor: str
    interface: Literal["server", "classic"]
    sanitized_input: dict[str, Any]
    elapsed_seconds: float = 0
    request_started_at: datetime | None = None
    request_finished_at: datetime | None = None
    concurrent_requests_at_start: int | None = None
    outcome: Literal["success", "empty", "failed", "uncertain"] = "failed"
    error: str | None = None
    target_status: int | None = None
    actor_http_status: int | None = None
    run_id: str | None = None
    account_plan: str = "unknown"
    warm_idle_status: str = "unknown"
    usage: dict[str, Any] | None = None
    usage_note: str | None = None
    model_generation_seconds: float | None = None


class Outcome(BaseModel):
    measurement: Measurement
    raw: list[dict[str, Any]] = Field(default_factory=list)
    content: str | None = None
