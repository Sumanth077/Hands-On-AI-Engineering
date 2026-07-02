#!/usr/bin/env python3
"""SaaS Customer Support Voice Agent -- Dynamic Variables Webhook.

Telnyx AI Assistant Builder calls this webhook at the start of each call.
We return real-time context that gets injected into the assistant's system
prompt as {{variable}} placeholders -- no static hardcoded values needed.

In production, replace the stub values with real database/API lookups.
"""

import os
import time

import telnyx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TELNYX_API_KEY: str = os.getenv("TELNYX_API_KEY", "")
TELNYX_PUBLIC_KEY: str = os.getenv("TELNYX_PUBLIC_KEY", "")

telnyx_client = telnyx.Telnyx(api_key=TELNYX_API_KEY, public_key=TELNYX_PUBLIC_KEY)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SaaS Customer Support -- Dynamic Variables Webhook",
    description="Injects real-time SaaS context into a Telnyx AI Assistant at call start.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Dynamic Variables webhook
# ---------------------------------------------------------------------------


def get_live_context() -> dict:
    """Return real-time context for the AI assistant system prompt.

    These values replace {{variable}} placeholders in the assistant's
    Instructions field (configured in the Telnyx portal).

    In production, replace these stubs with real lookups:
      - system_status    -> query your status page API
      - current_queue_wait -> query your support queue
      - todays_incidents -> query PagerDuty, OpsGenie, etc.
    """
    return {
        "system_status": "All systems operational",
        "current_queue_wait": "Under 2 minutes",
        "business_hours": "Monday to Friday, 9AM to 5PM EST; Saturday, 9AM to 1PM EST",
        "todays_incidents": "None",
        "support_email": "support@bank.com",
    }


@app.post("/webhooks/dynamic-variables")
async def dynamic_variables(request: Request) -> JSONResponse:
    """Telnyx calls this endpoint at the start of each call.

    Telnyx sends a POST with call metadata. We respond with a dict of
    key/value pairs that are injected into the assistant's system prompt
    via {{variable}} placeholders.

    Docs: https://developers.telnyx.com/docs/inference/ai-assistants/dynamic-variables
    """
    # Verify Ed25519 signature before trusting the request
    raw_body = await request.body()
    try:
        telnyx_client.webhooks.unwrap(raw_body.decode("utf-8"), dict(request.headers))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Return live context -- Telnyx injects these as {{variable}} in the prompt
    return JSONResponse(get_live_context())


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "timestamp": int(time.time())})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
