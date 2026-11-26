"""Official OpenAI Responses SDK with a persistent, pre-request cumulative reservation cap."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import SecretStr

from .config import require_live_enabled
from .llm import SYSTEM, ModelAuthenticationError, ModelBudgetError, ModelError, ModelQuotaError
from .research_models import Limits

SELECTED_OPENAI_MODEL = "gpt-6-luna"
# Official global Standard pricing/model/caching docs verified 2026-11-25.
# Nanodollars/token: $0.10/M input, $0.50/M output. Upper bound uses 2x/1.5x
# long-context rates for EVERY call, ignores cache discounts, and disables cache writes.
INPUT_RATE_NANO = 100
OUTPUT_RATE_NANO = 500
BOUND_INPUT_RATE_NANO = 200
BOUND_OUTPUT_RATE_NANO = 750
CONTEXT_TOKEN_CEILING = 1_050_000
CAP_NANO = 500_000_000
DEFAULT_LEDGER = Path("outputs/openai-integration-budget.sqlite3")


def strict_schema(schema: dict) -> dict:
    """Require defaulted fields on the wire while retaining saved-output compatibility."""
    schema = json.loads(json.dumps(schema))

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


class CostLedger:
    """Reservations survive timeouts, process restarts, new UI runs, and manual retries.

    No automatic reset. An operator must explicitly authorize a new integration budget.
    SQLite IMMEDIATE transactions prevent concurrent callers from oversubscribing it.
    """

    def __init__(self, path: Path = DEFAULT_LEDGER):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, stage TEXT, "
                "reserved INTEGER NOT NULL, charged INTEGER, status TEXT NOT NULL)"
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
            if "input_bound" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN input_bound INTEGER")

    def reserve(
        self, stage: str, max_output_tokens: int, *, input_tokens: int = CONTEXT_TOKEN_CEILING
    ) -> tuple[str, int]:
        if type(input_tokens) is not int or not 0 < input_tokens <= CONTEXT_TOKEN_CEILING:
            raise ModelBudgetError("Invalid authoritative input count; no generation sent")
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            committed = db.execute(
                "SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM attempts"
            ).fetchone()[0]
            input_bound = input_tokens * BOUND_INPUT_RATE_NANO
            output_allowance = min(
                max_output_tokens, (CAP_NANO - committed - input_bound) // BOUND_OUTPUT_RATE_NANO
            )
            if output_allowance < 256:
                raise ModelBudgetError(
                    f"OpenAI ${CAP_NANO / 1e9:.2f} cost cap cannot reserve the next request. Research stopped; no retry or provider fallback."
                )
            reservation = input_bound + output_allowance * BOUND_OUTPUT_RATE_NANO
            attempt = uuid4().hex
            db.execute(
                "INSERT INTO attempts (id,stage,reserved,charged,status,input_bound) VALUES (?,?,?,NULL,'reserved',?)",
                (attempt, stage, reservation, input_tokens),
            )
        return attempt, reservation

    def settle(self, attempt: str, input_tokens: int, output_tokens: int, max_output_tokens: int):
        if not (
            type(input_tokens) is int
            and type(output_tokens) is int
            and 0 <= input_tokens <= CONTEXT_TOKEN_CEILING
            and 0 <= output_tokens <= max_output_tokens
        ):
            raise ModelBudgetError(
                "OpenAI usage exceeded the verified reservation bounds; stop and inspect billing"
            )
        upper_cost = input_tokens * BOUND_INPUT_RATE_NANO + output_tokens * BOUND_OUTPUT_RATE_NANO
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT reserved,input_bound FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
            if (
                row is None
                or input_tokens > (row[1] or CONTEXT_TOKEN_CEILING)
                or upper_cost > row[0]
            ):
                raise ModelBudgetError(
                    "OpenAI usage exceeded the counted reservation; hold retained"
                )
            db.execute(
                "UPDATE attempts SET charged=?,status='settled' WHERE id=?", (upper_cost, attempt)
            )

    def summary(self) -> dict:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT reserved,charged,status FROM attempts").fetchall()
        return {
            "cap_usd": CAP_NANO / 1e9,
            "attempts": len(rows),
            "committed_upper_bound_usd": sum(
                row[1] if row[1] is not None else row[0] for row in rows
            )
            / 1e9,
            "unsettled_attempts": sum(row[1] is None for row in rows),
            "settled_attempts": sum(row[1] is not None for row in rows),
            "uncertain_reserved_usd": sum(row[0] for row in rows if row[1] is None) / 1e9,
            "ledger_path": str(self.path.resolve()),
            "pricing_verified_on": "2026-11-25",
        }


class OpenAIBackend:
    def __init__(
        self,
        model: str,
        key: SecretStr,
        *,
        ledger: CostLedger,
        transport: httpx.MockTransport | None = None,
        recorder=None,
    ):
        if model != SELECTED_OPENAI_MODEL:
            raise ValueError(
                "OPENAI_MODEL must be gpt-6-luna; other IDs need verified pricing before live use"
            )
        if not key.get_secret_value().strip() or key.get_secret_value().startswith("replace_with_"):
            raise ValueError("OpenAI provider setup pending: add OPENAI_API_KEY locally")
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError("Only MockTransport may bypass the live gate")
        self.model, self.key, self.ledger = model, key, ledger
        self.transport, self.recorder = transport, recorder
        self.usage_records: list[dict] = []
        self.count_records: list[dict] = []

    async def generate(self, stage: str, context: dict, schema: dict, limits: Limits) -> str:
        if self.transport is None:
            require_live_enabled()
        input_body = {
            "model": self.model,
            "instructions": SYSTEM,
            "input": json.dumps({"stage": stage, "data": context}),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": stage,
                    "strict": True,
                    "schema": strict_schema(schema),
                }
            },
            "tools": [],
            "tool_choice": "none",
            "reasoning": {"effort": "low"},
            "truncation": "disabled",
        }
        counted = await self._count_input(stage, input_body)
        attempt, reserved = self.ledger.reserve(
            stage, limits.max_output_tokens, input_tokens=counted
        )
        output_allowance = (reserved - counted * BOUND_INPUT_RATE_NANO) // BOUND_OUTPUT_RATE_NANO
        record = {
            "provider": "openai_responses",
            "model": self.model,
            "stage": stage,
            "attempt_id": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reserved_cost_usd": reserved / 1e9,
            "max_output_tokens": output_allowance,
            "usage": None,
            "estimated_cost_usd": None,
            "outcome": "uncertain",
        }
        self.usage_records.append(record)
        record["counted_input_tokens"] = counted
        record["input_payload_sha256"] = hashlib.sha256(
            json.dumps(input_body, sort_keys=True).encode()
        ).hexdigest()
        body = {
            **input_body,
            "max_output_tokens": output_allowance,
            "store": False,
            "background": False,
            "service_tier": "default",
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        }
        # Explicit caching with no breakpoints disables caching/writes per official docs.
        # max_retries=0 ensures every possible generation has its own reservation.
        if self.recorder:
            self.recorder(
                "openai_request",
                {"attempt_id": attempt, "request": body, "budget": self.ledger.summary()},
            )
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=httpx.Timeout(50, connect=10),
                follow_redirects=False,
            ) as http:
                async with AsyncOpenAI(
                    api_key=self.key.get_secret_value(),
                    base_url="https://api.openai.com/v1",
                    max_retries=0,
                    timeout=50,
                    http_client=http,
                ) as client:
                    response = await client.responses.create(**body)
            if self.recorder:
                self.recorder(
                    "openai_response",
                    {
                        "attempt_id": attempt,
                        "response": response.model_dump(mode="json", exclude_none=True),
                    },
                )
            usage = response.usage
            record["reported_model"] = response.model
            record["service_tier"] = response.service_tier
            if not re.fullmatch(
                re.escape(self.model) + r"(?:-\d{4}-\d{2}-\d{2})?", response.model or ""
            ):
                raise ModelBudgetError(
                    "OpenAI reported an unverified model; reservation retained and research stopped"
                )
            if response.service_tier not in {None, "default"}:
                raise ModelBudgetError(
                    "OpenAI returned an unexpected processing tier; reservation retained"
                )
            if usage is not None:
                if getattr(usage.input_tokens_details, "cache_write_tokens", 0):
                    raise ModelBudgetError(
                        "Unexpected OpenAI cache write despite disabled caching; stop and inspect billing"
                    )
                self.ledger.settle(
                    attempt, usage.input_tokens, usage.output_tokens, output_allowance
                )
                record["usage"] = usage.model_dump(mode="json")
                # Standard global uncached estimate, discounts ignored; actual invoice unknown.
                long = usage.input_tokens > 272_000
                record["estimated_cost_usd"] = (
                    usage.input_tokens * (BOUND_INPUT_RATE_NANO if long else INPUT_RATE_NANO)
                    + usage.output_tokens * (BOUND_OUTPUT_RATE_NANO if long else OUTPUT_RATE_NANO)
                ) / 1e9
            if response.status != "completed":
                raise ModelError("OpenAI returned incomplete/failed output; no automatic retry")
            if any(item.type not in {"message", "reasoning"} for item in response.output):
                raise ModelError(
                    "Unexpected OpenAI tool output rejected; hosted tools are disabled"
                )
            if any(
                part.type == "refusal"
                for item in response.output
                if item.type == "message"
                for part in item.content
            ):
                raise ModelError("OpenAI refused structured output; no retry")
            text = response.output_text
            if not text or len(text) > limits.max_output_chars:
                raise ModelError("OpenAI output missing or exceeds configured size")
            record["outcome"] = "response_received"
            return text
        except APIStatusError as exc:
            record["outcome"] = "failed"
            if self.recorder:
                self.recorder(
                    "openai_error",
                    {"attempt_id": attempt, "http_status": exc.status_code, "body": exc.body},
                )
            if exc.status_code in {401, 403}:
                raise ModelAuthenticationError(
                    "OpenAI authentication/permission failed; verify OPENAI_API_KEY and model access. Research stopped."
                ) from None
            error_body = exc.body if isinstance(exc.body, dict) else {}
            error_body = error_body.get("error", error_body)
            code = str(error_body.get("code", "")) if isinstance(error_body, dict) else ""
            if exc.status_code in {402, 429} or code.startswith(
                ("billing", "insufficient_quota", "credit_balance")
            ):
                raise ModelQuotaError(
                    "OpenAI billing/quota limit reached; research stopped without retry. Check API credits and project limits."
                ) from None
            raise ModelError(
                f"OpenAI HTTP {exc.status_code}; no retry or provider fallback"
            ) from None
        except APIConnectionError:
            raise ModelError(
                "OpenAI transport/timeout failure; execution and billing uncertain. Reservation retained; no retry."
            ) from None
        finally:
            record["budget"] = self.ledger.summary()
            if self.recorder:
                self.recorder("openai_usage", record)

    async def _count_input(self, stage: str, body: dict) -> int:
        """Official free complete-input count; same immutable input fields as generation.

        Source: help.openai.com/en/articles/4936856-understanding-and-counting-tokens.
        No files/history/tools, no guessed framing allowance or local character heuristic.
        A failed count has no inference reservation because no generation has been sent.
        """
        record = {
            "stage": stage,
            "endpoint": "responses.input_tokens.count",
            "cost_usd": 0,
            "input_payload_sha256": hashlib.sha256(
                json.dumps(body, sort_keys=True).encode()
            ).hexdigest(),
            "outcome": "failed",
        }
        self.count_records.append(record)
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=httpx.Timeout(50, connect=10),
                follow_redirects=False,
            ) as http:
                async with AsyncOpenAI(
                    api_key=self.key.get_secret_value(),
                    base_url="https://api.openai.com/v1",
                    max_retries=0,
                    http_client=http,
                ) as client:
                    response = await client.responses.input_tokens.count(**body)
            count = response.input_tokens
            if type(count) is not int or not 0 < count <= CONTEXT_TOKEN_CEILING:
                raise ModelBudgetError("Invalid complete-input count; generation not sent")
            record.update(outcome="success", input_tokens=count)
            return count
        except APIStatusError as exc:
            record["http_status"] = exc.status_code
            if exc.status_code in {401, 403}:
                raise ModelAuthenticationError(
                    "OpenAI token-count authentication failed; stopped"
                ) from None
            if exc.status_code in {402, 429}:
                raise ModelQuotaError("OpenAI token-count billing/quota error; stopped") from None
            raise ModelBudgetError(
                "OpenAI complete-input count unavailable; no generation sent"
            ) from None
        except APIConnectionError:
            raise ModelBudgetError(
                "OpenAI token-count transport failed; no generation sent"
            ) from None
        finally:
            if self.recorder:
                self.recorder("openai_input_count", record)
