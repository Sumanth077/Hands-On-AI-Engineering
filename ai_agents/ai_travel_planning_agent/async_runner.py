"""Run async agent calls from the synchronous Streamlit entrypoint."""

import asyncio
from typing import Any


def run_coroutine(coroutine: Any) -> Any:
    """Run a coroutine, falling back to ``asyncio.run`` only without a loop.

    Runtime errors from the coroutine itself must propagate. Retrying those
    errors through a second loop could replay provider-backed agent calls.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    return loop.run_until_complete(coroutine)
