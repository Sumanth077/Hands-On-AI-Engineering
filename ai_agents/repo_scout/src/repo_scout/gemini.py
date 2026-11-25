"""Official Google Gen AI SDK adapter, Gemini Developer API only; no web tools."""

import asyncio
import json

import httpx
from google import genai
from google.genai import errors, types
from pydantic import SecretStr

from .config import require_live_enabled
from .llm import SYSTEM, ModelAuthenticationError, ModelError, ModelQuotaError
from .research_models import Limits

# Verified against Google's model/pricing docs on 2026-11-25. Not an account-tier check.
SELECTED_MODEL = "gemini-3.5-flash-lite"


class GeminiBackend:
    def __init__(
        self,
        model: str,
        key: SecretStr,
        *,
        transport: httpx.MockTransport | None = None,
        recorder=None,
    ):
        if model != SELECTED_MODEL:
            raise ValueError(
                "GEMINI_MODEL must be gemini-3.5-flash-lite, the verified free-tier model. No fallback."
            )
        if not key.get_secret_value().strip():
            raise ValueError("Set GEMINI_API_KEY locally")
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError("Only a MockTransport can bypass the live gate")
        self.model, self.key, self.transport = model, key, transport
        self.usage_records: list[dict] = []
        self.recorder = recorder

    async def generate(self, stage: str, context: dict, schema: dict, limits: Limits) -> str:
        if self.transport is None:
            require_live_enabled()
        record = {
            "provider": "gemini_developer_api",
            "model": self.model,
            "stage": stage,
            "outcome": "failed",
            "usage": None,
            "cost_usd": None,
            "account_tier": "unknown",
        }
        self.usage_records.append(record)
        if self.recorder:
            self.recorder(
                "gemini_request",
                {"stage": stage, "context": context, "schema": schema, "model": self.model},
            )
        # Explicit endpoint/key/Developer API mode override ambient SDK auth/routing settings.
        options = types.HttpOptions(
            base_url="https://generativelanguage.googleapis.com",
            api_version="v1beta",
            timeout=50000,
            retry_options=types.HttpRetryOptions(attempts=1),
            async_client_args={"transport": self.transport} if self.transport else None,
        )
        client = genai.Client(
            api_key=self.key.get_secret_value(), enterprise=False, http_options=options
        )
        try:
            async with client.aio as aclient:
                response = await aclient.models.generate_content(
                    model=self.model,
                    contents=json.dumps({"stage": stage, "data": context}),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        max_output_tokens=limits.max_output_tokens,
                        candidate_count=1,
                        tools=[],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                        automatic_continuation=False,
                    ),
                )
            if self.recorder:
                self.recorder(
                    "gemini_response",
                    {
                        "stage": stage,
                        "response": response.model_dump(mode="json", exclude_none=True),
                    },
                )
            # Preserve authoritative token metadata even if the output is blocked/malformed.
            metadata = response.usage_metadata
            if metadata is not None:
                record["usage"] = {
                    name: getattr(metadata, name)
                    for name in (
                        "prompt_token_count",
                        "candidates_token_count",
                        "thoughts_token_count",
                        "cached_content_token_count",
                        "total_token_count",
                        "tool_use_prompt_token_count",
                    )
                    if type(getattr(metadata, name, None)) is int
                }
            record["reported_model_version"] = response.model_version
            candidates = response.candidates or []
            if len(candidates) != 1:
                raise ModelError("Gemini blocked or omitted structured output; no retry")
            candidate = candidates[0]
            if candidate.finish_reason != types.FinishReason.STOP:
                raise ModelError("Gemini output blocked, truncated, or incomplete; no retry")
            parts = candidate.content.parts if candidate.content else []
            if any(part.function_call is not None for part in (parts or [])):
                raise ModelError(
                    "Unexpected Gemini tool call rejected; Apify alone performs web retrieval"
                )
            text = "".join(part.text for part in (parts or []) if part.text and not part.thought)
            if not text:
                raise ModelError("Gemini omitted structured JSON; no retry")
            if len(text) > limits.max_output_chars:
                raise ModelError("Gemini output exceeds configured character limit; no retry")
            record["outcome"] = "response_received"
            # StructuredModel applies the exact stage's Pydantic model and citation validation follows.
            return text
        except errors.APIError as exc:
            if self.recorder:
                self.recorder(
                    "gemini_error", {"stage": stage, "http_status": exc.code, "body": exc.details}
                )
            if exc.code in {401, 403}:
                record["outcome"] = "authentication_failed"
                raise ModelAuthenticationError(
                    "Gemini authentication/permission failed. Research stopped; verify the local key and project access before retrying."
                ) from None
            if exc.code == 429 or exc.status == "RESOURCE_EXHAUSTED":
                record["outcome"] = "quota_exhausted"
                raise ModelQuotaError(
                    "Gemini quota exhausted (429 RESOURCE_EXHAUSTED). Research stopped; "
                    "completed evidence is preserved. Check project quotas in AI Studio before a new run. "
                    "No automatic retry, paid-model switch, or provider fallback."
                ) from None
            raise ModelError(f"Gemini API error (HTTP {exc.code}); no retry or fallback") from None
        except (TimeoutError, httpx.RequestError, asyncio.CancelledError):
            record["outcome"] = "uncertain"
            raise
        finally:
            client.close()
            if self.recorder:
                self.recorder("gemini_usage", record)
